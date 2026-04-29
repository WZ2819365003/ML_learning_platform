"""TS Service — V3 executor for ts family training runs.

Responsibilities:
  * Load Dataset → DataFrame via ModelingTask.dataset_id
  * Resolve ts config from ExperimentRun.params["time_series"] or
    ModelingTask.config["time_series"]
  * Slice by validation strategy (holdout only in M6 scope)
  * Instantiate trainer via get_ts_trainer(model_type)
  * fit → predict → compute FORECAST_EVAL_METRICS
  * Persist model file under storage/models/{run_id}{ext}
  * Register executor("ts_train", ...) for V3 dispatch chain

Domain lookup:
  The executor receives ``platform_task_id`` and uses it to locate the
  ExperimentRun via ``ExperimentRun.task_id == platform_task_id``.
  ``domain_id`` is accepted for API symmetry but is not used for the lookup;
  it reflects the TrainingTask stub id written by ``_persist_trials``.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy import select

from app.config import get_settings
from app.core.ts_metrics import coverage, mae, mape, mase, rmse, smape
from app.core.ts_trainer import TSMeta, get_ts_trainer
from app.models.database import (
    Dataset,
    ExperimentRun,
    ModelingTask,
    PlatformExperiment,
    async_session_factory,
)
from app.scheduler.executors import register_executor

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _slice_holdout(
    df: pd.DataFrame, target_col: str, test_size: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if len(df) <= test_size:
        raise ValueError(
            f"Dataset has {len(df)} rows but test_size={test_size}; "
            "need at least test_size + 1 rows."
        )
    train = df.iloc[:-test_size].copy()
    test = df.iloc[-test_size:].copy()
    return train, test


def _build_meta(ts_cfg: dict) -> TSMeta:
    """Construct a TSMeta from the time_series config dict."""
    return TSMeta(
        timestamp_col=ts_cfg["timestamp_col"],
        target_col=ts_cfg["target_col"],
        series_id_col=ts_cfg.get("series_id_col"),
        exogenous_cols=ts_cfg.get("exogenous_cols") or [],
        freq=ts_cfg.get("freq", "D"),
        horizon=int(ts_cfg["horizon"]),
        lookback=int(ts_cfg.get("lookback", 28)),
        interval_levels=ts_cfg.get("interval_levels") or [80, 95],
    )


def _resolve_extension(token: str) -> str:
    """Map a ts registry token to the appropriate model file extension."""
    if token in ("arima", "ets"):
        return ".joblib"
    if token in ("lstm_forecaster", "tcn_forecaster"):
        return ".pt"
    if token == "timesfm_1":
        return ".json"
    raise ValueError(f"Unknown ts token {token!r}; cannot determine file extension.")


def _resolve_ts_config(run: ExperimentRun, mtask: ModelingTask | None) -> dict:
    """
    Extract the time_series config dict from available sources in priority order:

    1. run.params["time_series"]  — set by the API / test fixture directly on the run
    2. mtask.config["time_series"] — set on the ModelingTask at creation time
    3. mtask.training_plan_snapshot["payload"]["time_series"] — frozen snapshot

    Raises RuntimeError if no config is found.
    """
    # 1. Directly on run params
    run_params = run.params or {}
    if run_params.get("time_series"):
        return dict(run_params["time_series"])

    if mtask is None:
        raise RuntimeError(
            f"No time_series config in ExperimentRun {run.id!r} params, "
            "and no ModelingTask available to fall back to."
        )

    # 2. On ModelingTask.config
    mtask_config = mtask.config or {}
    if mtask_config.get("time_series"):
        return dict(mtask_config["time_series"])

    # 3. On ModelingTask.training_plan_snapshot
    snapshot = mtask.training_plan_snapshot or {}
    snap_payload = snapshot.get("payload") or {}
    if snap_payload.get("time_series"):
        return dict(snap_payload["time_series"])

    raise RuntimeError(
        f"No time_series config found for ExperimentRun {run.id!r}. "
        "Expected one of: run.params['time_series'], "
        "ModelingTask.config['time_series'], or "
        "ModelingTask.training_plan_snapshot['payload']['time_series']."
    )


def _resolve_dataset_id(run: ExperimentRun, mtask: ModelingTask | None) -> str:
    """
    Resolve the Dataset.id from the run params or the ModelingTask.

    Priority:
    1. run.params["dataset_id"]
    2. mtask.dataset_id
    """
    run_params = run.params or {}
    if run_params.get("dataset_id"):
        return str(run_params["dataset_id"])
    if mtask is not None and mtask.dataset_id:
        return str(mtask.dataset_id)
    raise RuntimeError(
        f"Cannot resolve dataset_id for ExperimentRun {run.id!r}. "
        "Set run.params['dataset_id'] or ModelingTask.dataset_id."
    )


# ---------------------------------------------------------------------------
# Executor
# ---------------------------------------------------------------------------

async def run_ts_executor(domain_id: str, platform_task_id: str) -> dict[str, Any]:
    """V3 executor for ts family training runs.

    Parameters
    ----------
    domain_id:
        ID of the stub TrainingTask row written by ``_persist_trials`` (kept
        for API symmetry with ml/dl executors; not used for DB lookups here).
    platform_task_id:
        ID of the PlatformTask.  Used to locate the associated ExperimentRun
        via ``ExperimentRun.task_id == platform_task_id``.
    """
    settings = get_settings()

    # ------------------------------------------------------------------
    # 1. Load run + related rows
    # ------------------------------------------------------------------
    async with async_session_factory() as session:
        run = (
            await session.execute(
                select(ExperimentRun).where(ExperimentRun.task_id == platform_task_id)
            )
        ).scalar_one_or_none()

        if run is None:
            # Fallback: try domain_id directly as ExperimentRun.id (test fixture path)
            run = (
                await session.execute(
                    select(ExperimentRun).where(ExperimentRun.id == domain_id)
                )
            ).scalar_one_or_none()

        if run is None:
            raise RuntimeError(
                f"No ExperimentRun found for platform_task_id={platform_task_id!r} "
                f"or domain_id={domain_id!r}."
            )

        # Load the parent experiment
        exp = (
            await session.execute(
                select(PlatformExperiment).where(PlatformExperiment.id == run.experiment_id)
            )
        ).scalar_one_or_none()

        # Load ModelingTask (may be None for synthetic fixtures)
        mtask: ModelingTask | None = None
        if exp is not None and exp.modeling_task_id:
            mtask = (
                await session.execute(
                    select(ModelingTask).where(ModelingTask.id == exp.modeling_task_id)
                )
            ).scalar_one_or_none()

        # Resolve ts config + dataset id (reads from run params, mtask, or snapshot)
        ts_cfg = _resolve_ts_config(run, mtask)
        dataset_id = _resolve_dataset_id(run, mtask)

        ds = (
            await session.execute(
                select(Dataset).where(Dataset.id == dataset_id)
            )
        ).scalar_one_or_none()

        if ds is None:
            raise RuntimeError(f"Dataset {dataset_id!r} not found in database.")

        file_path = ds.file_path
        run_id = run.id
        run_params = dict(run.params or {})

    # ------------------------------------------------------------------
    # 2. Build TSMeta + load data
    # ------------------------------------------------------------------
    meta = _build_meta(ts_cfg)

    df = pd.read_csv(file_path)
    df = df.sort_values(meta.timestamp_col).reset_index(drop=True)

    # ------------------------------------------------------------------
    # 3. Validation split
    # ------------------------------------------------------------------
    val_cfg = ts_cfg.get("validation") or {}
    val_method = val_cfg.get("method", "holdout")
    if val_method != "holdout":
        raise NotImplementedError(
            f"validation.method={val_method!r} is not implemented in M6 scope. "
            "Only 'holdout' is supported."
        )
    test_size = int(val_cfg.get("test_size") or meta.horizon)
    train_df, test_df = _slice_holdout(df, meta.target_col, test_size)

    # ------------------------------------------------------------------
    # 4. Fit
    # ------------------------------------------------------------------
    model_type = run_params.get("model_type", "arima")
    hyperparameters = run_params.get("hyperparameters") or {}

    trainer = get_ts_trainer(model_type)
    trainer.fit(train_df, meta, params=hyperparameters)

    # ------------------------------------------------------------------
    # 5. Predict
    # ------------------------------------------------------------------
    horizon = len(test_df)
    result = trainer.predict(horizon=horizon)

    # ------------------------------------------------------------------
    # 6. Compute metrics
    # ------------------------------------------------------------------
    y_true = test_df[meta.target_col].to_numpy(dtype=float)
    y_pred = np.asarray(result.mean, dtype=float)
    train_y = train_df[meta.target_col].to_numpy(dtype=float)

    metrics: dict[str, float] = {
        "mae":   mae(y_true, y_pred),
        "rmse":  rmse(y_true, y_pred),
        "mape":  mape(y_true, y_pred),
        "smape": smape(y_true, y_pred),
        "mase":  mase(y_true, y_pred, train_y, season=1),
    }

    if result.intervals:
        for level, (lo, hi) in result.intervals.items():
            metrics[f"coverage_{level}"] = coverage(
                y_true,
                np.asarray(lo, dtype=float),
                np.asarray(hi, dtype=float),
            )

    # ------------------------------------------------------------------
    # 7. Save model artifact
    # ------------------------------------------------------------------
    ext = _resolve_extension(model_type)
    settings.storage_models.mkdir(parents=True, exist_ok=True)
    model_path = settings.storage_models / f"{run_id}{ext}"
    trainer.save(model_path)

    logger.info(
        "TS run %s (%s) completed. horizon=%d metrics=%s",
        run_id, model_type, horizon, metrics,
    )

    return {
        "metrics": metrics,
        "artifacts": {
            "model_path": str(model_path),
            "horizon": horizon,
            "model_type": model_type,
        },
    }


register_executor("ts_train", run_ts_executor)

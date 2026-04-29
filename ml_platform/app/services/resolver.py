"""
Task / Dataset / Model resolver — shared by viz_service, shap_service and
model management routes.

The platform has three legitimate ways to refer to a trained model:

  * **Legacy TrainingTask** — `TrainingTask.id` directly, a real row in `training_tasks`.
  * **V3 ExperimentRun** — `ExperimentRun.id` (or the linked `PlatformTask.id`);
    model+metrics artifacts live on disk keyed by the legacy id hidden inside
    `PlatformTask.payload_ref = "train:<legacy_id>"`.
  * **Orphan legacy task** — TrainingTask row was purged but `storage/models/{id}.joblib`
    and `storage/logs/{id}.log` still exist; we can recover dataset + target from
    the log file.

This module hides that branching behind a single `resolve_task_and_dataset` call
so every downstream consumer (chart endpoints, SHAP, model detail pages) sees a
uniform `TaskFacade` view with `model_path`, `target_column`, `model_type` etc.

Task-type awareness is built in: `TaskFacade.task_kind` is always either
`"classification"` or `"regression"`, derived from `model_type`. Callers should
branch on this field rather than string-matching model_type themselves — it's
the single source of truth for "which metrics/charts apply to this task".
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any

import joblib
import numpy as np
from fastapi import HTTPException
from sklearn.model_selection import train_test_split
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.models.database import (
    Dataset,
    ExperimentRun,
    ModelingTask,
    PlatformExperiment,
    PlatformTask,
    TrainingTask,
)
from app.services.prediction_service import load_dataframe, prepare_training_frame
from app.utils.storage_paths import resolve_runtime_path

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# TaskFacade — uniform view of a trained model regardless of backing row
# ---------------------------------------------------------------------------


LEGACY_REGRESSOR_MODEL_TYPES: set[str] = {
    "linear_regression",
    "random_forest_regressor",
    "xgboost_regressor",
    "lightgbm_regressor",
    "mlp_dl_regressor",
    "lstm_regressor",
    "tcn_regressor",
}

# Classifier tokens whose names collide with the `_regression` suffix heuristic.
# Without this override `logistic_regression` (a classifier) would be flagged as
# a regressor and downstream classification-only viz endpoints (per_class /
# pr_curve / calibration / threshold / confusion_matrix) would 400-error.
CLASSIFIER_NAME_OVERRIDES: set[str] = {
    "logistic_regression",
}


def is_regressor(model_type: str | None) -> bool:
    """Decide if a trained model is a regressor based on its `model_type` string.

    Order:
      1. explicit regressor allow-list
      2. classifier override list (defeats the suffix trap)
      3. conventional `_regressor` / `_regression` suffix heuristic
    """
    if not model_type:
        return False
    lower = model_type.lower()
    if lower in LEGACY_REGRESSOR_MODEL_TYPES:
        return True
    if lower in CLASSIFIER_NAME_OVERRIDES:
        return False
    return lower.endswith("_regressor") or lower.endswith("_regression")


FORECASTER_TOKENS: set[str] = {
    "arima", "ets", "lstm_forecaster", "tcn_forecaster", "timesfm_1",
}


def is_forecaster(model_type: str | None) -> bool:
    """True if model_type is one of the ts family tokens."""
    if not model_type:
        return False
    return model_type.lower() in FORECASTER_TOKENS


def task_kind_for(model_type: str | None) -> str:
    """Return 'classification' | 'regression' | 'forecasting'."""
    if is_forecaster(model_type):
        return "forecasting"
    return "regression" if is_regressor(model_type) else "classification"


@dataclass
class TaskFacade:
    """Duck-typed subset of TrainingTask used by downstream viz / shap code."""

    id: str
    model_type: str | None
    model_path: str | None
    target_column: str | None
    test_size: float
    status: str
    dataset_id: str | None
    result_metrics: dict | None = None

    @property
    def task_kind(self) -> str:
        return task_kind_for(self.model_type)


# ---------------------------------------------------------------------------
# Raw on-disk helpers — reused across resolver + shap_service
# ---------------------------------------------------------------------------

_MODEL_EXTENSIONS = (".joblib", ".pt", ".json")


def _find_model_file(models_dir, base_id: str):
    """Search storage/models/ for {base_id}.{joblib,pt,json} — first match wins.

    Returns (Path, ext) or (None, None) if no candidate exists.
    """
    for ext in _MODEL_EXTENSIONS:
        candidate = models_dir / f"{base_id}{ext}"
        if candidate.exists():
            return candidate, ext
    return None, None


def load_model(model_path: str):
    """Load a saved model artifact; raise 404 if the file is missing."""
    path = resolve_runtime_path(model_path)
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Model file not found: {model_path}")
    return joblib.load(path)


def load_and_split_data_stratified(
    file_path: str, target_column: str, test_size: float = 0.2
):
    """Classification-only helper — stratify on y so class balance is preserved.

    Fails on regression targets (continuous y) because stratify requires a
    discrete variable. Use `load_and_split_data_no_stratify` for regression.
    """
    df = load_dataframe(file_path)
    X, y, _, target_encoder = prepare_training_frame(df, target_column)

    X_train, X_test, y_train, y_test = train_test_split(
        X.values, y.values, test_size=test_size, random_state=42, stratify=y.values
    )

    feature_names = list(X.columns)
    class_labels = (
        target_encoder.classes_.tolist()
        if target_encoder is not None
        else sorted([str(c) for c in np.unique(y.values)])
    )

    return X_train, X_test, y_train, y_test, feature_names, class_labels


def load_and_split_data_no_stratify(
    file_path: str, target_column: str, test_size: float = 0.2
):
    """Regression-safe helper — no stratify, continuous y is fine."""
    df = load_dataframe(file_path)
    X, y, _, _ = prepare_training_frame(df, target_column)

    X_train, X_test, y_train, y_test = train_test_split(
        X.values, y.values, test_size=test_size, random_state=42
    )
    feature_names = list(X.columns)
    return X_train, X_test, y_train, y_test, feature_names


# ---------------------------------------------------------------------------
# Log / metrics side-channels — for orphan recovery and context inference
# ---------------------------------------------------------------------------


def pick_test_size(container: Any, default: float = 0.2) -> float:
    """Coerce test_size out of a params/snapshot dict, clamped to [0.05, 0.5]."""
    if not container:
        return default
    try:
        v = container.get("test_size") if hasattr(container, "get") else None
    except Exception:
        return default
    if v is None:
        return default
    try:
        f = float(v)
        if 0.05 <= f <= 0.5:
            return f
    except (TypeError, ValueError):
        pass
    return default


def pick_model_type_from_metrics(task_id: str) -> str | None:
    """Peek at storage/logs/{task_id}_metrics.json for the trainer-recorded model_type."""
    settings = get_settings()
    mf = settings.storage_logs / f"{task_id}_metrics.json"
    if not mf.exists():
        return None
    try:
        with open(mf, "r") as f:
            data = json.load(f)
        return data.get("model_type") or None
    except Exception:
        return None


def parse_legacy_log_context(task_id: str) -> dict:
    """Fallback: extract dataset= / target= from storage/logs/{task_id}.log.

    Lets an orphaned task (TrainingTask row gone, model+log files still around)
    self-describe enough to render charts.
    """
    settings = get_settings()
    log_path = settings.storage_logs / f"{task_id}.log"
    if not log_path.exists():
        return {}
    try:
        text = log_path.read_text(errors="ignore")
    except Exception:
        return {}
    out: dict[str, str] = {}
    m = re.search(r"dataset=([^\s|]+)", text)
    if m:
        out["dataset_path"] = m.group(1).strip()
    m = re.search(r"target=([^\s|]+)", text)
    if m:
        out["target_column"] = m.group(1).strip()
    return out


def parse_legacy_final_metrics(task_id: str) -> dict[str, float] | None:
    """Parse the final `Training completed | metric=value` line from logs.

    Recovered/orphan tasks often have no DB row but the trainer's final log
    line still carries the metrics; we surface those so UI cards render.
    """
    settings = get_settings()
    log_path = settings.storage_logs / f"{task_id}.log"
    if not log_path.exists():
        return None
    try:
        lines = log_path.read_text(errors="ignore").splitlines()
    except Exception:
        return None
    final_lines = [line for line in lines if "Training completed" in line and "=" in line]
    if not final_lines:
        return None
    metrics: dict[str, float] = {}
    for key, value in re.findall(r"\b([A-Za-z_][A-Za-z0-9_]*)=([-+]?\d+(?:\.\d+)?)", final_lines[-1]):
        try:
            metrics[key] = float(value)
        except ValueError:
            continue
    return metrics or None


async def resolve_dataset_by_path(db: AsyncSession, file_path: str) -> Dataset | None:
    """Look up a Dataset by exact file_path match (used for orphan recovery)."""
    result = await db.execute(select(Dataset).where(Dataset.file_path == file_path))
    return result.scalar_one_or_none()


# ---------------------------------------------------------------------------
# Id-candidate walker — log / metrics files are keyed by the legacy id
# ---------------------------------------------------------------------------


async def resolve_legacy_id_candidates(task_id: str, db: AsyncSession) -> list[str]:
    """Return every id worth probing when looking for on-disk artifacts.

    Chains: TrainingTask id | ExperimentRun id | ExperimentRun.task_id
    (PlatformTask id) | PlatformTask.payload_ref legacy id. Deduplicated,
    input-order preserved so the caller's primary key is tried first.
    """
    candidates: list[str] = [task_id]

    run = (await db.execute(
        select(ExperimentRun).where(ExperimentRun.id == task_id)
    )).scalar_one_or_none()
    if run is None:
        pt_by_id = (await db.execute(
            select(PlatformTask).where(PlatformTask.id == task_id)
        )).scalar_one_or_none()
        if pt_by_id is not None:
            run = (await db.execute(
                select(ExperimentRun).where(ExperimentRun.task_id == pt_by_id.id)
            )).scalar_one_or_none()
            if pt_by_id.payload_ref and ":" in pt_by_id.payload_ref:
                candidates.append(pt_by_id.payload_ref.split(":", 1)[1])
    if run is not None and run.task_id:
        candidates.append(run.task_id)
        pt = (await db.execute(
            select(PlatformTask).where(PlatformTask.id == run.task_id)
        )).scalar_one_or_none()
        if pt and pt.payload_ref and ":" in pt.payload_ref:
            candidates.append(pt.payload_ref.split(":", 1)[1])

    pt_ref = (await db.execute(
        select(PlatformTask).where(PlatformTask.payload_ref.in_(
            [f"train:{task_id}", f"dl_train:{task_id}"]
        ))
    )).scalar_one_or_none()
    if pt_ref is not None:
        candidates.append(pt_ref.id)

    seen: set[str] = set()
    ordered: list[str] = []
    for c in candidates:
        if c and c not in seen:
            seen.add(c)
            ordered.append(c)
    return ordered


# ---------------------------------------------------------------------------
# Facade synthesis — V3 run + orphan
# ---------------------------------------------------------------------------


async def synthesize_facade_from_run(
    run: ExperimentRun, db: AsyncSession
) -> tuple[TaskFacade, Dataset | None]:
    """Build a TaskFacade for a V3 ExperimentRun.

    V3 runs don't carry model_path/target/test_size on the ORM row — those live
    on the parent PlatformExperiment → ModelingTask + the plan snapshot. We
    reconstitute them so downstream code can run unchanged.
    """
    params = run.params or {}
    settings = get_settings()
    candidate_ids: list[str] = [run.id]
    if run.task_id:
        candidate_ids.append(run.task_id)

    legacy_id: str | None = None
    if run.task_id:
        pt = (await db.execute(
            select(PlatformTask).where(PlatformTask.id == run.task_id)
        )).scalar_one_or_none()
        if pt and pt.payload_ref and ":" in pt.payload_ref:
            legacy_id = pt.payload_ref.split(":", 1)[1]
            if legacy_id:
                candidate_ids.append(legacy_id)

    model_path: str | None = None
    for cid in candidate_ids:
        candidate, ext = _find_model_file(settings.storage_models, cid)
        if candidate:
            model_path = f"storage/models/{cid}{ext}"
            break

    exp = (await db.execute(
        select(PlatformExperiment).where(PlatformExperiment.id == run.experiment_id)
    )).scalar_one_or_none()
    dataset: Dataset | None = None
    target_column: str | None = None
    if exp is not None:
        if exp.dataset_id:
            dataset = (await db.execute(
                select(Dataset).where(Dataset.id == exp.dataset_id)
            )).scalar_one_or_none()
        if exp.modeling_task_id:
            mt = (await db.execute(
                select(ModelingTask).where(ModelingTask.id == exp.modeling_task_id)
            )).scalar_one_or_none()
            if mt:
                target_column = mt.target_column
                if dataset is None and mt.dataset_id:
                    dataset = (await db.execute(
                        select(Dataset).where(Dataset.id == mt.dataset_id)
                    )).scalar_one_or_none()

    model_type: str | None = None
    if isinstance(params, dict):
        model_type = params.get("model_type") or params.get("model")
    if not model_type:
        for cid in candidate_ids:
            mt_guess = pick_model_type_from_metrics(cid)
            if mt_guess:
                model_type = mt_guess
                break

    test_size = pick_test_size(params)
    return TaskFacade(
        id=run.id,
        model_type=model_type,
        model_path=model_path,
        target_column=target_column,
        test_size=test_size,
        status="SUCCESS" if run.status == "SUCCESS" else (run.status or "UNKNOWN"),
        dataset_id=(exp.dataset_id if exp else None),
        result_metrics=run.metrics or next(
            (m for m in (parse_legacy_final_metrics(cid) for cid in candidate_ids) if m),
            None,
        ),
    ), dataset


async def synthesize_facade_from_orphan(
    task_id: str, db: AsyncSession
) -> tuple[TaskFacade, Dataset | None] | None:
    """Recover from on-disk artifacts when the DB row is gone.

    Returns None if the orphan reconstruction doesn't work (no model file,
    unparseable log, or dataset no longer registered).
    """
    settings = get_settings()
    model_file, ext = _find_model_file(settings.storage_models, task_id)
    if not model_file:
        return None
    ctx = parse_legacy_log_context(task_id)
    dataset_path = ctx.get("dataset_path")
    target_column = ctx.get("target_column")
    if not dataset_path or not target_column:
        return None

    dataset = await resolve_dataset_by_path(db, dataset_path)
    if dataset is None:
        return None

    model_type = pick_model_type_from_metrics(task_id)
    return TaskFacade(
        id=task_id,
        model_type=model_type,
        model_path=f"storage/models/{task_id}{ext}",
        target_column=target_column,
        test_size=0.2,
        status="SUCCESS",
        dataset_id=dataset.id,
        result_metrics=parse_legacy_final_metrics(task_id),
    ), dataset


# ---------------------------------------------------------------------------
# Primary entry — resolve any task/run/orphan id to (TaskFacade, Dataset)
# ---------------------------------------------------------------------------


async def resolve_task_and_dataset(task_id: str, db: AsyncSession):
    """Multi-source resolver: TrainingTask → V3 ExperimentRun → on-disk orphan.

    The caller gets a `TaskFacade` and `Dataset` back in all three cases,
    so downstream code is oblivious to which data path was taken.
    Raises HTTPException(404) when no tier matches.
    """
    # --- Tier 1: legacy TrainingTask direct ---------------------------------
    result = await db.execute(select(TrainingTask).where(TrainingTask.id == task_id))
    task = result.scalar_one_or_none()
    if task is not None:
        if task.status != "SUCCESS":
            raise HTTPException(status_code=400, detail=f"Task not completed (status={task.status})")
        if not task.model_path:
            raise HTTPException(status_code=400, detail="No model saved for this task")
        dataset = (await db.execute(
            select(Dataset).where(Dataset.id == task.dataset_id)
        )).scalar_one_or_none()
        if dataset is None:
            raise HTTPException(status_code=404, detail="Dataset not found")
        return task, dataset

    # --- Tier 2: V3 ExperimentRun synthesis ---------------------------------
    run = (await db.execute(
        select(ExperimentRun).where(ExperimentRun.id == task_id)
    )).scalar_one_or_none()
    if run is None:
        pt = (await db.execute(
            select(PlatformTask).where(PlatformTask.id == task_id)
        )).scalar_one_or_none()
        if pt is not None:
            run = (await db.execute(
                select(ExperimentRun).where(ExperimentRun.task_id == pt.id)
            )).scalar_one_or_none()
            if run is None and pt.payload_ref and pt.payload_ref.startswith(("train:", "dl_train:")):
                legacy_id = pt.payload_ref.split(":", 1)[1]
                legacy_tt = (await db.execute(
                    select(TrainingTask).where(TrainingTask.id == legacy_id)
                )).scalar_one_or_none()
                if legacy_tt is not None:
                    return await resolve_task_and_dataset(legacy_id, db)

    if run is not None:
        if run.status != "SUCCESS":
            raise HTTPException(
                status_code=400,
                detail=f"Run not completed (status={run.status or 'UNKNOWN'})",
            )
        facade, dataset = await synthesize_facade_from_run(run, db)
        if not facade.model_path:
            raise HTTPException(status_code=404, detail="Model artifact not found for this run")
        if not facade.target_column:
            raise HTTPException(status_code=400, detail="Target column unavailable for this run")
        if dataset is None:
            raise HTTPException(status_code=404, detail="Dataset not found for this run")
        return facade, dataset

    # --- Tier 3: orphan recovery from on-disk artifacts ---------------------
    orphan = await synthesize_facade_from_orphan(task_id, db)
    if orphan is not None:
        facade, dataset = orphan
        if dataset is None:
            raise HTTPException(status_code=404, detail="Dataset not found for this task")
        return facade, dataset

    raise HTTPException(status_code=404, detail="Training task not found")


# ---------------------------------------------------------------------------
# Combined helper — resolve + load model + split data
# ---------------------------------------------------------------------------


async def resolve_and_load(task_id: str, db: AsyncSession, *, stratified: bool | None = None):
    """Resolve a task then load model + split dataset.

    `stratified` auto-picks based on task_kind when None (False for regression,
    True for classification). Override when the caller explicitly wants a
    non-stratified split even for classification (SHAP sampling, for example).
    """
    task, dataset = await resolve_task_and_dataset(task_id, db)
    if stratified is None:
        stratified = task.task_kind == "classification" if isinstance(task, TaskFacade) \
            else not is_regressor(getattr(task, "model_type", None))

    if stratified:
        X_train, X_test, y_train, y_test, feature_names, class_labels = load_and_split_data_stratified(
            dataset.file_path, task.target_column, task.test_size
        )
    else:
        X_train, X_test, y_train, y_test, feature_names = load_and_split_data_no_stratify(
            dataset.file_path, task.target_column, task.test_size
        )
        class_labels = None

    model = load_model(task.model_path)
    return {
        "task": task,
        "dataset": dataset,
        "model": model,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "feature_names": feature_names,
        "class_labels": class_labels,
    }

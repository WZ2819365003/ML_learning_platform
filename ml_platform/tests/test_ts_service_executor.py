"""Integration test: end-to-end ARIMA run through ts_service.run_ts_executor.

Schema discoveries (verified against database.py):

ExperimentRun:
  - Has ``task_id`` (FK → platform_tasks.id), NOT ``platform_task_id``.
  - No direct model_type / task_type columns; those live in ``params`` (JSON).

PlatformExperiment:
  - Has ``modeling_task_id``, ``strategy_type``, ``status``, ``name``.
  - No ``model_family`` or ``task_type`` direct columns.

TrainingPlan:
  - Has ``selected_models``, ``search_space``, ``dl_config``, ``budget_config``.
  - No ``payload`` column and no ``dataset_id`` column (plans are dataset-agnostic).

Executor lookup strategy (ts_service.run_ts_executor):
  1. Finds ExperimentRun via ExperimentRun.task_id == platform_task_id.
  2. Resolves ts config from run.params["time_series"].
  3. Resolves dataset_id from run.params["dataset_id"].
  4. Falls back to ExperimentRun.id == domain_id for synthetic test fixtures.

Production propagation path (M6 Batch A fix):
  tuning_service._persist_trials copies ModelingTask.config["time_series"] →
  ExperimentRun.params["time_series"] for every ts trial, so the executor's
  primary resolution path is always populated.  test_ts_executor_arima_end_to_end
  mirrors both layers (mtask.config AND run.params["time_series"]) for clarity;
  test_persist_trials_injects_ts_config_from_task exercises the propagation itself.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from app.models.database import (
    Dataset,
    ExperimentRun,
    ModelingTask,
    PlatformExperiment,
    PlatformTask,
    TrainingPlan,
    async_session_factory,
)
from app.services.ts_service import run_ts_executor


@pytest.mark.asyncio
async def test_ts_executor_arima_end_to_end(tmp_path: Path) -> None:
    """Build a minimal DB fixture, dispatch run_ts_executor, assert metrics returned."""
    n = 120
    csv_path = tmp_path / "sin.csv"
    pd.DataFrame(
        {
            "ds": pd.date_range("2024-01-01", periods=n, freq="D"),
            "y": np.sin(np.arange(n) / 10),
        }
    ).to_csv(csv_path, index=False)

    async with async_session_factory() as session:
        # ── 1. Dataset ──────────────────────────────────────────────────────
        ds = Dataset(
            name="sin_test_executor.csv",
            file_path=str(csv_path),
            file_size=csv_path.stat().st_size,
            row_count=n,
            column_count=2,
            columns_info={
                "ds": {"dtype": "object"},
                "y": {"dtype": "float64"},
            },
        )
        session.add(ds)
        await session.flush()

        # ── 2. TrainingPlan ─────────────────────────────────────────────────
        # Plans have no payload / dataset_id columns.  The ts config travels
        # inside ExperimentRun.params["time_series"] (set below).
        plan = TrainingPlan(
            name="ts-arima-executor-test-plan",
            task_type="forecasting",
            model_family="ts",
            strategy_type="baseline",
            selected_models=["arima"],
        )
        session.add(plan)
        await session.flush()

        # ── 3. ModelingTask ─────────────────────────────────────────────────
        # In production, _persist_trials copies config["time_series"] into
        # each ExperimentRun.params["time_series"].  We set both here so the
        # fixture mirrors the production state after propagation.
        _ts_config: dict[str, Any] = {
            "timestamp_col": "ds",
            "target_col": "y",
            "series_id_col": None,
            "exogenous_cols": [],
            "freq": "D",
            "horizon": 7,
            "lookback": 14,
            "validation": {
                "method": "holdout",
                "test_size": 7,
                "step": 1,
            },
            "interval_levels": [80, 95],
        }
        mtask = ModelingTask(
            name="ts-arima-executor-test-task",
            dataset_id=ds.id,          # plain String column, not FK
            task_type="forecasting",
            training_plan_id=plan.id,
            config={"time_series": _ts_config},
        )
        session.add(mtask)
        await session.flush()

        # ── 4. PlatformExperiment ───────────────────────────────────────────
        exp = PlatformExperiment(
            name="exp-ts-arima-executor-test",
            modeling_task_id=mtask.id,
            strategy_type="baseline",
            status="QUEUED",
        )
        session.add(exp)
        await session.flush()

        # ── 5. PlatformTask ─────────────────────────────────────────────────
        pt = PlatformTask(
            kind="ts_train",
            status="QUEUED",
            payload_ref="ts_train:placeholder",  # updated after run flush
        )
        session.add(pt)
        await session.flush()

        # ── 6. ExperimentRun ────────────────────────────────────────────────
        # task_id links to PlatformTask (column name is task_id, not platform_task_id).
        # In production, _persist_trials populates params["time_series"] from
        # mtask.config["time_series"].  This fixture mirrors that propagated state.
        run = ExperimentRun(
            experiment_id=exp.id,
            task_id=pt.id,
            params={
                "family": "ts",
                "model_type": "arima",
                "hyperparameters": {"p": 1, "d": 1, "q": 1, "seasonal_periods": 0},
                "dataset_id": ds.id,
                "target_column": "y",
                "task_type": "forecasting",
                # Propagated from mtask.config["time_series"] by _persist_trials.
                "time_series": _ts_config,
            },
            status="QUEUED",
        )
        session.add(run)
        await session.flush()

        # Update payload_ref to ts_train:{run.id} so the executor can locate
        # the run via ExperimentRun.task_id == platform_task_id OR via domain_id.
        pt.payload_ref = f"ts_train:{run.id}"
        await session.flush()
        await session.commit()

        run_id = run.id
        pt_id = pt.id

    # ── Call executor ────────────────────────────────────────────────────────
    # domain_id = run_id (test-fixture path used as fallback inside executor)
    # platform_task_id = pt_id (primary lookup: ExperimentRun.task_id == pt_id)
    result = await run_ts_executor(domain_id=run_id, platform_task_id=pt_id)

    # ── Assert metrics ───────────────────────────────────────────────────────
    assert "metrics" in result, f"Expected 'metrics' key, got: {list(result.keys())}"
    m = result["metrics"]

    for key in ("mae", "rmse", "mape", "smape"):
        assert key in m, f"Missing metric {key!r} in {list(m.keys())}"
        assert m[key] is not None, f"Metric {key!r} is None"
        assert isinstance(m[key], (int, float)), f"Metric {key!r} not numeric: {m[key]}"

    # ARIMA has supports_intervals=True so coverage should be present
    assert "coverage_80" in m, (
        f"Expected ARIMA interval coverage_80 in metrics, got: {list(m.keys())}"
    )

    assert "artifacts" in result
    assert "model_path" in result["artifacts"]

    # Model file should exist on disk
    model_path = Path(result["artifacts"]["model_path"])
    assert model_path.exists(), f"Model file not found at {model_path}"


# ---------------------------------------------------------------------------
# Unit test: _persist_trials propagates ts config from ModelingTask.config
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_persist_trials_injects_ts_config_from_task_config() -> None:
    """Verify _persist_trials copies task.config['time_series'] → run.params['time_series'].

    Production gap (M6 Batch A): before the fix, ts trials were dispatched
    without run.params['time_series'], causing ts_service.run_ts_executor to
    raise RuntimeError('No ts config found') in production.

    This test patches the DB helpers so it can call _persist_trials directly
    and inspect the ExperimentRun that gets constructed.
    """
    from app.services.tuning_service import _persist_trials

    ts_config: dict[str, Any] = {
        "timestamp_col": "ds",
        "target_col": "y",
        "series_id_col": None,
        "exogenous_cols": [],
        "freq": "D",
        "horizon": 7,
        "lookback": 14,
        "validation": {"method": "holdout", "test_size": 7, "step": 1},
        "interval_levels": [80, 95],
    }

    # Build a minimal ModelingTask stub (no DB needed).
    task = MagicMock(spec=ModelingTask)
    task.dataset_id = "ds-unit-test"
    task.target_column = "y"
    task.task_type = "forecasting"
    task.config = {"time_series": ts_config}

    # Build a minimal PlatformExperiment stub.
    exp = MagicMock(spec=PlatformExperiment)
    exp.id = "exp-unit-test"
    exp.strategy_type = "baseline"

    trial: dict[str, Any] = {
        "family": "ts",
        "model_type": "arima",
        "hyperparameters": {"p": 1, "d": 1, "q": 1, "seasonal_periods": 0},
        "trial_no": 1,
        "search_meta": {},
    }

    # Stub domain task and platform task returned by helpers.
    stub_domain_task = MagicMock()
    stub_domain_task.id = "domain-task-unit-test"

    stub_platform_task = MagicMock()
    stub_platform_task.id = "platform-task-unit-test"

    # Capture the ExperimentRun that gets add()ed to the session.
    captured_runs: list[Any] = []

    db = AsyncMock()

    async def _fake_flush() -> None:
        # After the first flush (after db.add(run)), we need run.id to be set.
        for run in captured_runs:
            if not hasattr(run, "_id_set"):
                run.id = "run-unit-test"
                run._id_set = True  # type: ignore[attr-defined]

    db.flush = _fake_flush

    async def _fake_refresh(obj: Any) -> None:
        pass

    db.refresh = _fake_refresh

    def _fake_add(obj: Any) -> None:
        if isinstance(obj, ExperimentRun):
            captured_runs.append(obj)

    db.add = _fake_add

    with (
        patch(
            "app.services.tuning_service._create_ts_training_task_record",
            new=AsyncMock(return_value=stub_domain_task),
        ),
        patch(
            "app.services.tuning_service.register_domain_task",
            new=AsyncMock(return_value=stub_platform_task),
        ),
    ):
        await _persist_trials(
            db=db,
            exp=exp,
            task=task,
            trials=[trial],
            eval_metrics=["mae", "rmse"],
        )

    assert len(captured_runs) == 1, "Expected exactly one ExperimentRun to be created"
    run = captured_runs[0]

    assert "time_series" in run.params, (
        "_persist_trials must inject time_series into run.params for ts family"
    )
    assert run.params["time_series"] == ts_config, (
        "run.params['time_series'] must equal task.config['time_series']"
    )
    assert run.params["family"] == "ts"
    assert run.params["model_type"] == "arima"


@pytest.mark.asyncio
async def test_persist_trials_raises_422_when_ts_config_missing() -> None:
    """_persist_trials must raise HTTPException(422) if task.config has no time_series."""
    from fastapi import HTTPException

    from app.services.tuning_service import _persist_trials

    task = MagicMock(spec=ModelingTask)
    task.dataset_id = "ds-unit-test"
    task.target_column = "y"
    task.task_type = "forecasting"
    task.config = {}  # no "time_series" key — should trigger 422

    exp = MagicMock(spec=PlatformExperiment)
    exp.id = "exp-unit-test"
    exp.strategy_type = "baseline"

    trial: dict[str, Any] = {
        "family": "ts",
        "model_type": "arima",
        "hyperparameters": {"p": 1, "d": 1, "q": 1, "seasonal_periods": 0},
        "trial_no": 1,
        "search_meta": {},
    }

    stub_domain_task = MagicMock()
    stub_domain_task.id = "domain-task-unit-test"

    db = AsyncMock()
    db.add = MagicMock()
    db.flush = AsyncMock()
    db.refresh = AsyncMock()

    with (
        patch(
            "app.services.tuning_service._create_ts_training_task_record",
            new=AsyncMock(return_value=stub_domain_task),
        ),
        patch(
            "app.services.tuning_service.register_domain_task",
            new=AsyncMock(return_value=MagicMock(id="pt-unit-test")),
        ),
        pytest.raises(HTTPException) as exc_info,
    ):
        await _persist_trials(
            db=db,
            exp=exp,
            task=task,
            trials=[trial],
            eval_metrics=["mae"],
        )

    assert exc_info.value.status_code == 422
    assert "time_series" in exc_info.value.detail

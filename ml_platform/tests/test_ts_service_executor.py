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
"""
from __future__ import annotations

from pathlib import Path

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
        mtask = ModelingTask(
            name="ts-arima-executor-test-task",
            dataset_id=ds.id,          # plain String column, not FK
            task_type="forecasting",
            training_plan_id=plan.id,
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
        # ts config lives in params["time_series"]; executor reads it from there.
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
                # ts config so the executor can build TSMeta without TrainingPlan.payload
                "time_series": {
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
                },
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

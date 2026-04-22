"""
Phase 2 — ModelingTask ↔ TrainingPlan binding tests.

Covers:
  * Create with ``training_plan_id`` writes a frozen snapshot (plan_id,
    plan_version, captured_at, payload, dag_shape).
  * Snapshot isolation: editing the plan AFTER bind does NOT rewrite the
    task's snapshot.
  * Plan version auto-increments on substantive edits.
  * task_type mismatch rejected with 422.
  * Unknown ``training_plan_id`` rejected with 404.
  * Create without plan works (backward-compat).
  * ``mark_used`` side-effect fires on bind (use_count += 1, last_used_at set).
  * dispatch_experiment_batch falls back to the snapshot payload when the
    caller omits selected_models / search_space / budget_config.
"""
from __future__ import annotations

from unittest.mock import patch

import pytest
from fastapi import HTTPException
from sqlalchemy import select

from app.models.database import Dataset, ModelingTask
from app.services import modeling_task_service as svc
from app.services import training_plan_service as plan_svc


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
async def dataset(db):
    ds = Dataset(name="iris_x", file_path="/tmp/iris_x.csv", file_size=2048, row_count=150)
    db.add(ds)
    await db.flush()
    return ds


@pytest.fixture
async def plan(db):
    return await plan_svc.create_plan(db, {
        "name": "baseline-classification",
        "task_type": "classification",
        "strategy_type": "baseline",
        "model_family": "ml",
        "selected_models": ["random_forest", "xgboost"],
        "budget_config": {"max_trials": 10},
    })


# ---------------------------------------------------------------------------
# Snapshot capture on bind
# ---------------------------------------------------------------------------

async def test_bind_writes_snapshot_with_version(db, dataset, plan):
    task = await svc.create_modeling_task(
        db,
        name="bound-task",
        dataset_id=dataset.id,
        target_column="y",
        task_type="classification",
        training_plan_id=plan["id"],
    )
    assert task["training_plan_id"] == plan["id"]
    snap = task["training_plan_snapshot"]
    assert snap is not None
    assert snap["plan_id"] == plan["id"]
    assert snap["plan_version"] == 1
    assert snap["plan_name"] == "baseline-classification"
    assert snap["captured_at"]  # ISO8601 string
    assert snap["payload"]["selected_models"] == ["random_forest", "xgboost"]
    assert snap["dag_shape"]["num_models"] == 2
    assert snap["dag_shape"]["strategy"] == "baseline"
    assert snap["dag_shape"]["model_family"] == "ml"


async def test_bind_no_plan_still_creates_task(db, dataset):
    """Backward-compat — omitting training_plan_id still works."""
    task = await svc.create_modeling_task(
        db,
        name="plain-task",
        dataset_id=dataset.id,
        target_column="y",
        task_type="classification",
    )
    assert task["training_plan_id"] is None
    assert task["training_plan_snapshot"] is None


async def test_bind_unknown_plan_404(db, dataset):
    with pytest.raises(HTTPException) as exc:
        await svc.create_modeling_task(
            db,
            name="bad-plan-ref",
            dataset_id=dataset.id,
            target_column="y",
            task_type="classification",
            training_plan_id="does-not-exist",
        )
    assert exc.value.status_code == 404


async def test_bind_task_type_mismatch_422(db, dataset, plan):
    """Plan.task_type=classification vs ModelingTask.task_type=regression → 422."""
    with pytest.raises(HTTPException) as exc:
        await svc.create_modeling_task(
            db,
            name="mismatch",
            dataset_id=dataset.id,
            target_column="y",
            task_type="regression",  # plan is classification
            training_plan_id=plan["id"],
        )
    assert exc.value.status_code == 422
    assert "task_type" in exc.value.detail


async def test_bind_bumps_use_count_and_last_used_at(db, dataset, plan):
    assert plan["use_count"] == 0
    assert plan["last_used_at"] is None
    await svc.create_modeling_task(
        db,
        name="using-plan",
        dataset_id=dataset.id,
        target_column="y",
        task_type="classification",
        training_plan_id=plan["id"],
    )
    fresh = await plan_svc.get_plan(db, plan["id"])
    assert fresh["use_count"] == 1
    assert fresh["last_used_at"] is not None


# ---------------------------------------------------------------------------
# Snapshot isolation + version tracking
# ---------------------------------------------------------------------------

async def test_snapshot_does_not_change_when_plan_is_edited(db, dataset, plan):
    task_payload = await svc.create_modeling_task(
        db,
        name="isolated",
        dataset_id=dataset.id,
        target_column="y",
        task_type="classification",
        training_plan_id=plan["id"],
    )
    frozen_models = task_payload["training_plan_snapshot"]["payload"]["selected_models"]
    frozen_version = task_payload["training_plan_snapshot"]["plan_version"]

    # Substantive edit — bumps plan.version and changes selected_models
    await plan_svc.update_plan(db, plan["id"], {
        "selected_models": ["logistic_regression"],
    })

    # Task snapshot must be unaffected
    fresh = await svc.get_modeling_task(db, task_payload["id"])
    snap = fresh["training_plan_snapshot"]
    assert snap["payload"]["selected_models"] == frozen_models  # still RF+XGB
    assert snap["plan_version"] == frozen_version  # still 1

    # …but the staleness indicator flips
    status = fresh["training_plan_status"]
    assert status["plan_exists"] is True
    assert status["snapshot_version"] == 1
    assert status["current_version"] == 2
    assert status["stale"] is True


async def test_plan_version_bumps_only_on_substantive_edits(db, plan):
    # Cosmetic edit — version stays at 1
    await plan_svc.update_plan(db, plan["id"], {"description": "just a note"})
    p = await plan_svc.get_plan(db, plan["id"])
    assert p["version"] == 1

    # Substantive edit — version bumps
    await plan_svc.update_plan(db, plan["id"], {"budget_config": {"max_trials": 20}})
    p = await plan_svc.get_plan(db, plan["id"])
    assert p["version"] == 2


# ---------------------------------------------------------------------------
# dispatch_experiment_batch reads snapshot when caller omits fields
# ---------------------------------------------------------------------------

async def test_dispatch_falls_back_to_snapshot(db, dataset, plan, session_factory):
    from app.services import tuning_service

    task_payload = await svc.create_modeling_task(
        db,
        name="snap-dispatch",
        dataset_id=dataset.id,
        target_column="y",
        task_type="classification",
        training_plan_id=plan["id"],
    )
    await db.commit()  # tuning_service opens its own sessions

    captured: dict = {}

    # Short-circuit _execute_single_trial so we don't try to actually train —
    # we only want to verify the batch is expanded from the snapshot.
    async def noop_trial(*args, **kwargs):
        return {"run_id": kwargs.get("run_id") or (args[2] if len(args) > 2 else None),
                "status": "SUCCESS", "metrics": {}}

    with patch("app.models.database.async_session_factory", session_factory), \
         patch.object(tuning_service, "_execute_single_trial", noop_trial):
        out = await tuning_service.dispatch_experiment_batch(
            db,
            modeling_task_id=task_payload["id"],
            name="batch-from-snapshot",
            strategy_type="",            # snapshot.strategy_type = baseline
            selected_models=[],          # snapshot has RF + XGB
            search_space={},
            budget_config={},
        )
        captured["out"] = out

    # Snapshot-driven expansion yields 2 trials (one per selected model)
    assert captured["out"]["strategy_type"] == "baseline"
    assert captured["out"]["trials_planned"] == 2

"""Tests for ModelingTask CRUD + aggregated views + tuning-space registry."""

from __future__ import annotations

import pytest
from fastapi import HTTPException

from app.models.database import (
    Dataset,
    ExperimentRun,
    ModelingTask,
    PlatformExperiment,
)
from app.services import modeling_task_service as svc


# ---------------------------------------------------------------------------
# Fixtures — minimal dataset + task seed
# ---------------------------------------------------------------------------

@pytest.fixture
async def dataset(db):
    ds = Dataset(name="iris_test", file_path="/tmp/iris.csv", file_size=1024, row_count=150)
    db.add(ds)
    await db.flush()
    return ds


@pytest.fixture
async def modeling_task(db, dataset):
    payload = await svc.create_modeling_task(
        db,
        name="churn-prediction-v1",
        dataset_id=dataset.id,
        target_column="y",
        task_type="classification",
        objective_metric="accuracy",
        objective_direction="max",
        description="smoke-test task",
    )
    return await svc._get_task_or_404(db, payload["id"])


# ---------------------------------------------------------------------------
# CRUD
# ---------------------------------------------------------------------------

async def test_create_task_denormalises_dataset_name(db, dataset):
    result = await svc.create_modeling_task(
        db,
        name="churn-v1",
        dataset_id=dataset.id,
        target_column="label",
        task_type="classification",
    )
    assert result["dataset_name"] == "iris_test"
    assert result["status"] == "CREATED"
    assert result["objective_metric"] == "accuracy"
    assert result["objective_direction"] == "max"


async def test_create_task_rejects_bad_task_type(db):
    with pytest.raises(HTTPException) as exc_info:
        await svc.create_modeling_task(
            db,
            name="x",
            dataset_id=None,
            target_column=None,
            task_type="clustering",  # invalid
        )
    assert exc_info.value.status_code == 422


async def test_create_task_rejects_bad_direction(db):
    with pytest.raises(HTTPException):
        await svc.create_modeling_task(
            db,
            name="x",
            dataset_id=None,
            target_column=None,
            objective_direction="neutral",
        )


async def test_list_returns_pagination_and_counts(db, modeling_task):
    result = await svc.list_modeling_tasks(db, page=1, page_size=10)
    assert result["total"] == 1
    assert result["items"][0]["experiment_count"] == 0
    assert result["items"][0]["successful_run_count"] == 0


async def test_get_returns_experiments_and_run_stats(db, modeling_task):
    exp = PlatformExperiment(
        modeling_task_id=modeling_task.id,
        name="baseline",
        strategy_type="baseline",
        dataset_id=modeling_task.dataset_id,
        objective_metric="accuracy",
        objective_direction="max",
    )
    db.add(exp)
    await db.flush()
    db.add_all([
        ExperimentRun(experiment_id=exp.id, status="SUCCESS",
                      metrics={"accuracy": 0.85}, source_experiment_type="baseline"),
        ExperimentRun(experiment_id=exp.id, status="FAILED", source_experiment_type="baseline"),
        ExperimentRun(experiment_id=exp.id, status="RUNNING", source_experiment_type="baseline"),
    ])
    await db.flush()

    payload = await svc.get_modeling_task(db, modeling_task.id)
    assert len(payload["experiments"]) == 1
    assert payload["experiments"][0]["strategy_type"] == "baseline"
    stats = payload["run_stats"]
    assert stats == {"total": 3, "success": 1, "running": 1, "failed": 1}


async def test_update_toggles_status_and_sets_finished_at(db, modeling_task):
    updated = await svc.update_modeling_task(
        db, modeling_task.id, name="renamed", status="completed"
    )
    assert updated["name"] == "renamed"
    assert updated["status"] == "COMPLETED"
    assert updated["finished_at"] is not None


async def test_update_rejects_unknown_status(db, modeling_task):
    with pytest.raises(HTTPException):
        await svc.update_modeling_task(db, modeling_task.id, status="mystery")


async def test_update_blocks_objective_changes_after_finalization(db, modeling_task):
    modeling_task.config = {
        "_final_evaluation": {
            "state": "FINALIZED",
            "version": 1,
            "winner_run_id": "winner-1",
        }
    }
    await db.flush()

    with pytest.raises(HTTPException) as exc:
        await svc.update_modeling_task(
            db,
            modeling_task.id,
            objective_metric="f1",
            objective_direction="min",
        )

    assert exc.value.status_code == 409
    assert "优化目标" in exc.value.detail

    renamed = await svc.update_modeling_task(
        db, modeling_task.id, name="finalized-renamed"
    )
    assert renamed["name"] == "finalized-renamed"


async def test_delete_blocks_running(db, modeling_task):
    modeling_task.status = "RUNNING"
    await db.flush()
    with pytest.raises(HTTPException) as exc_info:
        await svc.delete_modeling_task(db, modeling_task.id)
    assert exc_info.value.status_code == 400


# ---------------------------------------------------------------------------
# Leaderboard + summary
# ---------------------------------------------------------------------------

async def test_leaderboard_ranks_across_experiments(db, modeling_task):
    # Two experiments, different strategies — leaderboard unifies them.
    exp_a = PlatformExperiment(
        modeling_task_id=modeling_task.id, name="baseline",
        strategy_type="baseline", objective_metric="accuracy", objective_direction="max",
    )
    exp_b = PlatformExperiment(
        modeling_task_id=modeling_task.id, name="grid",
        strategy_type="grid_search", objective_metric="accuracy", objective_direction="max",
    )
    db.add_all([exp_a, exp_b])
    await db.flush()

    db.add_all([
        ExperimentRun(experiment_id=exp_a.id, status="SUCCESS",
                      metrics={"accuracy": 0.99, "cv_avg_accuracy": 0.81},
                      source_experiment_type="baseline"),
        ExperimentRun(experiment_id=exp_b.id, status="SUCCESS",
                      metrics={"accuracy": 0.70, "selection_cv_mean_accuracy": 0.93},
                      source_experiment_type="grid_search", trial_no=2),
        ExperimentRun(experiment_id=exp_b.id, status="SUCCESS",
                      metrics={"accuracy": 0.98, "cv_avg_accuracy": 0.88},
                      source_experiment_type="grid_search", trial_no=1),
        ExperimentRun(experiment_id=exp_b.id, status="FAILED",
                      source_experiment_type="grid_search"),
    ])
    await db.flush()

    board = await svc.task_leaderboard(db, modeling_task.id, top_k=10)
    assert len(board) == 3  # failed excluded
    assert board[0]["objective_value"] == 0.93
    assert board[0]["selection_metric_key"] == "selection_cv_mean_accuracy"
    assert board[0]["selection_value"] == 0.93
    assert board[0]["final_test_value"] == 0.70
    assert board[0]["strategy_type"] == "grid_search"
    assert board[0]["trial_no"] == 2
    assert board[-1]["objective_value"] == 0.81


async def test_refresh_task_summary_updates_best_and_status(db, modeling_task):
    exp = PlatformExperiment(
        modeling_task_id=modeling_task.id, name="baseline",
        strategy_type="baseline", objective_metric="accuracy", objective_direction="max",
        status="COMPLETED",
    )
    db.add(exp)
    await db.flush()
    best_run = ExperimentRun(experiment_id=exp.id, status="SUCCESS",
                             metrics={"accuracy": 0.91}, source_experiment_type="baseline")
    db.add(best_run)
    await db.flush()

    await svc.refresh_task_summary(db, modeling_task.id)
    await db.refresh(modeling_task)

    assert modeling_task.best_run_id == best_run.id
    assert modeling_task.best_experiment_id == exp.id
    assert modeling_task.summary_snapshot["experiment_count"] == 1
    assert modeling_task.summary_snapshot["top_objective_value"] == 0.91
    assert modeling_task.status == "COMPLETED"


async def test_min_direction_inverts_leaderboard(db, dataset):
    result = await svc.create_modeling_task(
        db,
        name="rmse-task",
        dataset_id=dataset.id,
        target_column="price",
        task_type="regression",
        objective_metric="rmse",
        objective_direction="min",
    )
    task = await svc._get_task_or_404(db, result["id"])
    exp = PlatformExperiment(
        modeling_task_id=task.id, name="batch",
        strategy_type="baseline", objective_metric="rmse", objective_direction="min",
    )
    db.add(exp)
    await db.flush()
    db.add_all([
        ExperimentRun(experiment_id=exp.id, status="SUCCESS", metrics={"rmse": 3.2},
                      source_experiment_type="baseline"),
        ExperimentRun(experiment_id=exp.id, status="SUCCESS", metrics={"rmse": 1.1},
                      source_experiment_type="baseline"),
        ExperimentRun(experiment_id=exp.id, status="SUCCESS", metrics={"rmse": 2.0},
                      source_experiment_type="baseline"),
    ])
    await db.flush()

    board = await svc.task_leaderboard(db, task.id, top_k=5)
    assert [r["objective_value"] for r in board] == [1.1, 2.0, 3.2]


# ---------------------------------------------------------------------------
# Tuning spaces registry
# ---------------------------------------------------------------------------

def test_load_tuning_spaces_classification():
    spaces = svc.load_tuning_spaces("classification")
    assert "random_forest" in spaces
    rf = spaces["random_forest"]
    assert set(rf["grid_values"].keys()) >= {"n_estimators", "max_depth"}
    assert "distribution" in rf
    assert rf["distribution"]["n_estimators"]["type"] == "int"


def test_load_tuning_spaces_regression():
    spaces = svc.load_tuning_spaces("regression")
    assert "ridge" in spaces
    assert spaces["ridge"]["distribution"]["alpha"]["log"] is True


def test_load_tuning_spaces_unknown_raises():
    with pytest.raises(ValueError):
        svc.load_tuning_spaces("clustering")

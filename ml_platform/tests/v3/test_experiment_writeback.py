"""
V3 Stage 2 tests — Experiment / ExperimentRun write-back contract.

All service functions return serialized dicts, so tests use dict access.
JSON columns are serialized to TEXT in SQLite — the functions still work.
Fixtures provided by conftest.py (function-scoped in-memory SQLite).
"""

import pytest
from sqlalchemy import select

from app.models.database import ExperimentRun, PlatformTask, PlatformExperiment


# ---------------------------------------------------------------------------
# experiment_service.create_experiment
# ---------------------------------------------------------------------------

async def test_create_experiment(db):
    from app.services.experiment_service import create_experiment

    exp = await create_experiment(
        db,
        name="Test Experiment",
        description="unit test",
        objective_metric="accuracy",
        objective_direction="max",
        kind="single",
    )
    await db.commit()

    assert exp["id"] is not None
    assert exp["name"] == "Test Experiment"
    assert exp["status"] == "CREATED"
    assert exp["objective_metric"] == "accuracy"


async def test_list_experiments_returns_created(session_factory):
    from app.services.experiment_service import create_experiment, list_experiments

    async with session_factory() as db:
        await create_experiment(db, "Exp A", objective_metric="f1", kind="automl")
        await db.commit()

    async with session_factory() as db:
        listing = await list_experiments(db)

    assert listing["total"] >= 1
    names = [e["name"] for e in listing["items"]]
    assert "Exp A" in names


# ---------------------------------------------------------------------------
# create_run + update_run_metrics
# ---------------------------------------------------------------------------

async def test_create_run_links_to_experiment(session_factory):
    from app.services.experiment_service import create_experiment, create_run

    async with session_factory() as db:
        exp = await create_experiment(db, "Run Link Test", objective_metric="accuracy")
        await db.flush()

        run = await create_run(
            db,
            exp["id"],
            params={"model_type": "random_forest", "n_estimators": 100},
        )
        await db.commit()
        run_id, exp_id = run["id"], exp["id"]

    async with session_factory() as db:
        result = await db.execute(select(ExperimentRun).where(ExperimentRun.id == run_id))
        reloaded = result.scalar_one()

    assert reloaded.experiment_id == exp_id
    assert reloaded.status == "PENDING"
    assert reloaded.params["model_type"] == "random_forest"


async def test_leaderboard_max_direction(session_factory):
    from app.services.experiment_service import (
        create_experiment, create_run, update_run_metrics, get_leaderboard
    )

    async with session_factory() as db:
        exp = await create_experiment(
            db, "Leaderboard Max",
            objective_metric="accuracy", objective_direction="max", kind="automl",
        )
        await db.flush()

        run1 = await create_run(db, exp["id"], params={"model_type": "rf"})
        run2 = await create_run(db, exp["id"], params={"model_type": "xgb"})
        run3 = await create_run(db, exp["id"], params={"model_type": "lr"})
        await db.flush()

        await update_run_metrics(
            db, run1["id"], {"accuracy": 0.99, "cv_avg_accuracy": 0.85}, status="SUCCESS"
        )
        await update_run_metrics(
            db, run2["id"], {"accuracy": 0.50, "cv_avg_accuracy": 0.92}, status="SUCCESS"
        )
        await update_run_metrics(
            db, run3["id"], {"accuracy": 0.98, "cv_avg_accuracy": 0.78}, status="SUCCESS"
        )
        await db.commit()

        exp_id = exp["id"]
        r1_id, r2_id, r3_id = run1["id"], run2["id"], run3["id"]

    async with session_factory() as db:
        board = await get_leaderboard(db, exp_id)

    assert len(board) == 3
    assert board[0]["id"] == r2_id, "xgb (0.92) should be rank 1"
    assert board[0]["rank"] == 1
    assert board[1]["id"] == r1_id, "rf (0.85) should be rank 2"
    assert board[2]["id"] == r3_id, "lr (0.78) should be rank 3"


async def test_leaderboard_min_direction(session_factory):
    from app.services.experiment_service import (
        create_experiment, create_run, update_run_metrics, get_leaderboard
    )

    async with session_factory() as db:
        exp = await create_experiment(
            db, "Min Loss Test",
            objective_metric="loss", objective_direction="min",
        )
        await db.flush()

        run_a = await create_run(db, exp["id"], params={"model": "a"})
        run_b = await create_run(db, exp["id"], params={"model": "b"})
        await db.flush()

        await update_run_metrics(db, run_a["id"], {"loss": 0.45}, status="SUCCESS")
        await update_run_metrics(db, run_b["id"], {"loss": 0.22}, status="SUCCESS")
        await db.commit()

        exp_id = exp["id"]
        ra_id, rb_id = run_a["id"], run_b["id"]

    async with session_factory() as db:
        board = await get_leaderboard(db, exp_id)

    assert board[0]["id"] == rb_id, "run_b (0.22 loss) should be rank 1"
    assert board[0]["rank"] == 1


# ---------------------------------------------------------------------------
# PlatformTask ↔ ExperimentRun link
# ---------------------------------------------------------------------------

async def test_run_task_link_integrity(session_factory):
    """ExperimentRun.task_id must point to a valid PlatformTask row."""
    from app.services.experiment_service import create_experiment, create_run
    from app.scheduler.task_runner import register_domain_task

    async with session_factory() as db:
        exp = await create_experiment(db, "Link Integrity", objective_metric="accuracy")
        await db.flush()

        ptask = await register_domain_task(db, kind="train", payload_ref="train:dummy-123")
        await db.flush()

        run = await create_run(db, exp["id"], params={"model": "lgbm"})
        # Patch the task_id onto the run ORM object directly
        run_orm_result = await db.execute(
            select(ExperimentRun).where(ExperimentRun.id == run["id"])
        )
        run_orm = run_orm_result.scalar_one()
        run_orm.task_id = ptask.id
        await db.commit()

        run_id, ptask_id = run["id"], ptask.id

    async with session_factory() as fresh_db:
        result = await fresh_db.execute(select(ExperimentRun).where(ExperimentRun.id == run_id))
        reloaded_run = result.scalar_one()
        assert reloaded_run.task_id == ptask_id

        pt_result = await fresh_db.execute(select(PlatformTask).where(PlatformTask.id == ptask_id))
        reloaded_pt = pt_result.scalar_one()
        assert reloaded_pt.payload_ref == "train:dummy-123"

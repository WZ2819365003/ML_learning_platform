"""M2c — contracts of the real call chains, not just the service function.

The service-level tests in ``test_run_writeback.py`` prove the write-back
primitive behaves. These prove the two *callers* actually honour it:

* a lost claim must skip execution AND leave no half-written RUNNING state
* an executor failure and a book-keeping failure must not be conflated —
  a transient DB error while committing a *successful* run must propagate,
  never be re-reported as a failed trial
"""
from __future__ import annotations

from unittest.mock import patch

import pytest
from sqlalchemy import select

from app.models.database import (
    Dataset,
    ExperimentRun,
    ModelingTask,
    PlatformExperiment,
    PlatformTask,
)
from app.scheduler import executors
from app.services import run_writeback, tuning_service


@pytest.fixture(autouse=True)
def use_test_sessions(session_factory):
    with patch("app.models.database.async_session_factory", session_factory):
        yield


@pytest.fixture(autouse=True)
def quiet_tails(monkeypatch):
    """Silence the post-commit tails; this file is about the caller contract."""
    async def noop(*a, **k):
        return None

    monkeypatch.setattr("app.services.tuning_service._finalise_batch", noop)
    monkeypatch.setattr("app.services.tuning_service._mirror_logs_to_v3", noop)
    monkeypatch.setattr(run_writeback, "_propagate_dag", noop)


async def _seed(db, *, run_status="PENDING"):
    ds = Dataset(name="cc.csv", file_path="/tmp/cc.csv", file_size=1, row_count=5)
    db.add(ds)
    await db.flush()
    task = ModelingTask(
        name="cc", dataset_id=ds.id, target_column="y",
        task_type="classification", objective_metric="accuracy",
    )
    db.add(task)
    await db.flush()
    exp = PlatformExperiment(
        modeling_task_id=task.id, name="cc-batch", strategy_type="baseline",
        dataset_id=ds.id, objective_metric="accuracy", status="RUNNING",
    )
    db.add(exp)
    await db.flush()
    pt = PlatformTask(kind="cc_kind", status="QUEUED", payload_ref="cc_kind:dom-1")
    db.add(pt)
    await db.flush()
    run = ExperimentRun(
        experiment_id=exp.id, task_id=pt.id,
        params={"model_type": "logistic_regression"}, status=run_status,
    )
    db.add(run)
    await db.flush()
    await db.commit()
    return pt.id, run.id, exp.id


async def _states(session_factory, run_id, pt_id):
    async with session_factory() as s:
        run = (await s.execute(select(ExperimentRun).where(ExperimentRun.id == run_id))).scalar_one()
        pt = (await s.execute(select(PlatformTask).where(PlatformTask.id == pt_id))).scalar_one()
    return run.status, pt.status


async def test_lost_claim_skips_executor_and_leaves_task_terminal(db, session_factory):
    """Duplicate delivery of a finished trial must not re-run it, and must not
    strand the PlatformTask in RUNNING (the bug the claim ordering fixes)."""
    pt_id, run_id, exp_id = await _seed(db)
    await run_writeback.complete_platform_task(pt_id, status="SUCCESS", metrics={"accuracy": 0.9})

    ran = []

    async def executor(domain_id, platform_task_id):
        ran.append(platform_task_id)
        return {"metrics": {"accuracy": 0.1}}

    executors.register_executor("cc_kind", executor)

    result = await tuning_service._execute_single_trial(
        "dom-1", pt_id, run_id, exp_id, kind="cc_kind"
    )

    assert result["status"] == "SKIPPED"
    assert ran == [], "executor re-ran an already-finished trial"
    run_status, task_status = await _states(session_factory, run_id, pt_id)
    assert (run_status, task_status) == ("SUCCESS", "SUCCESS"), (
        "duplicate delivery dragged a terminal task back to RUNNING"
    )


async def test_successful_run_with_failing_writeback_propagates(db, session_factory):
    """A DB hiccup while committing a *success* must surface, not be recorded
    as a failed trial."""
    pt_id, run_id, exp_id = await _seed(db)

    async def executor(domain_id, platform_task_id):
        return {"metrics": {"accuracy": 0.95}}

    executors.register_executor("cc_kind", executor)

    async def boom(*a, **k):
        raise RuntimeError("commit exploded")

    with patch.object(run_writeback, "_commit_terminal_state", boom):
        with pytest.raises(RuntimeError, match="commit exploded"):
            await tuning_service._execute_single_trial(
                "dom-1", pt_id, run_id, exp_id, kind="cc_kind"
            )

    run_status, _ = await _states(session_factory, run_id, pt_id)
    assert run_status != "FAILED", "a successful trial was downgraded to FAILED"


async def test_executor_failure_is_recorded_as_failed(db, session_factory):
    pt_id, run_id, exp_id = await _seed(db)

    async def executor(domain_id, platform_task_id):
        raise ValueError("model blew up")

    executors.register_executor("cc_kind", executor)

    result = await tuning_service._execute_single_trial(
        "dom-1", pt_id, run_id, exp_id, kind="cc_kind"
    )
    assert result["status"] == "FAILED"
    run_status, task_status = await _states(session_factory, run_id, pt_id)
    assert (run_status, task_status) == ("FAILED", "FAILED")


async def test_non_dict_executor_result_is_a_trial_failure(db, session_factory):
    """A misbehaving executor must terminalise the run, not escape upward."""
    pt_id, run_id, exp_id = await _seed(db)

    async def executor(domain_id, platform_task_id):
        return ["not", "a", "dict"]

    executors.register_executor("cc_kind", executor)

    result = await tuning_service._execute_single_trial(
        "dom-1", pt_id, run_id, exp_id, kind="cc_kind"
    )
    assert result["status"] == "FAILED"
    run_status, task_status = await _states(session_factory, run_id, pt_id)
    assert (run_status, task_status) == ("FAILED", "FAILED")


async def test_claim_marks_both_records_running(db, session_factory):
    pt_id, run_id, _ = await _seed(db)
    assert await run_writeback.claim_for_execution(pt_id) is True
    run_status, task_status = await _states(session_factory, run_id, pt_id)
    assert (run_status, task_status) == ("RUNNING", "RUNNING")


async def test_bayesian_under_celery_is_rejected(monkeypatch):
    """Until M2d, routing bayesian to a worker would strand the experiment."""
    from app.scheduler.scheduler import CeleryScheduler

    monkeypatch.setattr(
        "app.scheduler.scheduler.get_scheduler", lambda kind: CeleryScheduler()
    )
    with pytest.raises(Exception) as exc:
        tuning_service._reject_bayesian_under_celery()
    assert "贝叶斯" in str(exc.value.detail)


async def test_startup_migration_adds_error_message(tmp_path):
    """Existing local SQLite DBs bootstrap via run_startup_migrations, not
    Alembic — the new column must be added there too."""
    from sqlalchemy import create_engine, inspect, text
    from app.models.database import Base

    db_file = tmp_path / "legacy.db"
    engine = create_engine(f"sqlite:///{db_file}")
    Base.metadata.create_all(engine)
    with engine.connect() as c:
        c.exec_driver_sql("ALTER TABLE experiment_runs DROP COLUMN error_message")
        c.commit()
    assert "error_message" not in {
        col["name"] for col in inspect(engine).get_columns("experiment_runs")
    }

    from app.core.migrations import _migrate_experiment_runs

    with engine.connect() as c:
        _migrate_experiment_runs(c)
        c.commit()

    assert "error_message" in {
        col["name"] for col in inspect(engine).get_columns("experiment_runs")
    }
    engine.dispose()

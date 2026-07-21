"""M2c — Celery-side contracts (`_execute_generic` + the task hooks).

The in-process caller is covered in ``test_run_writeback_callchain.py``. These
pin the worker path, where the failure modes are different: Celery decides
retries, and its hooks run *after* the task body returns.

Central invariant: a bookkeeping failure must never be laundered into "the
trial failed". Training that actually succeeded may not end up FAILED just
because the commit hiccuped.
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
from app.scheduler import celery_tasks, executors
from app.services import run_writeback
from app.services.run_writeback import WritebackError


@pytest.fixture(autouse=True)
def use_test_sessions(session_factory):
    with patch("app.models.database.async_session_factory", session_factory):
        yield


@pytest.fixture(autouse=True)
def quiet_tails(monkeypatch):
    async def noop(*a, **k):
        return None

    monkeypatch.setattr("app.services.tuning_service._finalise_batch", noop)
    monkeypatch.setattr("app.services.tuning_service._mirror_logs_to_v3", noop)
    monkeypatch.setattr(run_writeback, "_propagate_dag", noop)


async def _seed(db, kind="celery_kind"):
    ds = Dataset(name="ck.csv", file_path="/tmp/ck.csv", file_size=1, row_count=5)
    db.add(ds)
    await db.flush()
    mt = ModelingTask(
        name="ck", dataset_id=ds.id, target_column="y",
        task_type="classification", objective_metric="accuracy",
    )
    db.add(mt)
    await db.flush()
    exp = PlatformExperiment(
        modeling_task_id=mt.id, name="ck-batch", strategy_type="baseline",
        dataset_id=ds.id, objective_metric="accuracy", status="RUNNING",
    )
    db.add(exp)
    await db.flush()
    pt = PlatformTask(kind=kind, status="QUEUED", payload_ref=f"{kind}:dom-1")
    db.add(pt)
    await db.flush()
    run = ExperimentRun(
        experiment_id=exp.id, task_id=pt.id,
        params={"model_type": "logistic_regression"}, status="PENDING",
    )
    db.add(run)
    await db.flush()
    await db.commit()
    return pt.id, run.id


async def _states(session_factory, run_id, pt_id):
    async with session_factory() as s:
        run = (await s.execute(select(ExperimentRun).where(ExperimentRun.id == run_id))).scalar_one()
        pt = (await s.execute(select(PlatformTask).where(PlatformTask.id == pt_id))).scalar_one()
    return run.status, pt.status


async def test_generic_success_writes_both_records(db, session_factory):
    pt_id, run_id = await _seed(db)

    async def ok(domain_id, platform_task_id):
        return {"metrics": {"accuracy": 0.88}}

    executors.register_executor("celery_kind", ok)
    await celery_tasks._execute_generic(pt_id)

    assert await _states(session_factory, run_id, pt_id) == ("SUCCESS", "SUCCESS")


async def test_generic_respects_lost_claim(db, session_factory):
    """A duplicate delivery must not re-run a finished trial."""
    pt_id, run_id = await _seed(db)
    await run_writeback.complete_platform_task(pt_id, status="SUCCESS", metrics={"accuracy": 0.9})

    ran = []

    async def spy(domain_id, platform_task_id):
        ran.append(platform_task_id)
        return {"metrics": {"accuracy": 0.1}}

    executors.register_executor("celery_kind", spy)
    result = await celery_tasks._execute_generic(pt_id)

    assert result.get("skipped") is True
    assert ran == [], "executor re-ran an already-terminal trial"
    assert await _states(session_factory, run_id, pt_id) == ("SUCCESS", "SUCCESS")


async def test_generic_non_dict_result_terminalises(db, session_factory):
    pt_id, run_id = await _seed(db)

    async def bad(domain_id, platform_task_id):
        return "definitely not a dict"

    executors.register_executor("celery_kind", bad)
    with pytest.raises(TypeError):
        await celery_tasks._execute_generic(pt_id)

    # It must be recorded, not silently escape to Celery with the Run left RUNNING.
    assert await _states(session_factory, run_id, pt_id) == ("FAILED", "FAILED")


async def test_generic_tags_writeback_failure(db):
    """Success + failing commit must raise WritebackError, not a plain error —
    that tag is what stops on_failure from writing FAILED."""
    pt_id, _ = await _seed(db)

    async def ok(domain_id, platform_task_id):
        return {"metrics": {"accuracy": 0.9}}

    executors.register_executor("celery_kind", ok)

    async def boom(*a, **k):
        raise RuntimeError("commit exploded")

    with patch.object(run_writeback, "_commit_terminal_state", boom):
        with pytest.raises(WritebackError):
            await celery_tasks._execute_generic(pt_id)


# ``on_failure`` runs in Celery's *synchronous* worker process and uses
# ``_run_async`` to spin a fresh event loop. Invoking it from inside a running
# loop (as an async test does) raises "Cannot run the event loop while another
# loop is running" — a real limitation of the existing _run_async pattern,
# tracked as a known risk. These tests therefore assert the *decision* the hook
# makes (terminalise or not) by spying on the dispatch, not by executing it.

def _capture_on_failure(monkeypatch, exc, platform_task_id):
    """Run the hook with _run_async stubbed; report whether it terminalised."""
    dispatched: list[str] = []

    def fake_run_async(coro):
        coro.close()  # we only care that a completion was requested
        dispatched.append("called")
        return None

    monkeypatch.setattr(celery_tasks, "_run_async", fake_run_async)

    class _Task(celery_tasks.MLBaseTask):
        pass

    _Task().on_failure(exc, "celery-task-id", (platform_task_id,), {}, None)
    return bool(dispatched)


def test_on_failure_does_not_bury_a_successful_run(monkeypatch):
    """The whole point of WritebackError: a good run must not become FAILED."""
    terminalised = _capture_on_failure(
        monkeypatch, WritebackError("succeeded but write-back failed"), "pt-1"
    )
    assert terminalised is False, (
        "a bookkeeping failure was about to be recorded as a failed trial"
    )


def test_on_failure_still_terminalises_real_failures(monkeypatch):
    terminalised = _capture_on_failure(monkeypatch, RuntimeError("model exploded"), "pt-1")
    assert terminalised is True


async def test_mark_task_retry_cannot_overwrite_terminal(db, session_factory):
    """A late on_retry must not drag a committed SUCCESS back to RETRY."""
    pt_id, run_id = await _seed(db)
    await run_writeback.complete_platform_task(pt_id, status="SUCCESS", metrics={"accuracy": 0.9})

    await celery_tasks._mark_task_retry(pt_id, "late retry hook")

    _, task_status = await _states(session_factory, run_id, pt_id)
    assert task_status == "SUCCESS", "on_retry overwrote a terminal state"


async def test_park_for_retry_cannot_overwrite_terminal(db, session_factory):
    pt_id, run_id = await _seed(db)
    await run_writeback.complete_platform_task(pt_id, status="SUCCESS", metrics={"accuracy": 0.9})

    await run_writeback._park_for_retry(pt_id, "late park")

    assert await _states(session_factory, run_id, pt_id) == ("SUCCESS", "SUCCESS")


async def test_cancelled_task_is_terminal_for_claims(db, session_factory):
    """CANCELLED counts as terminal — a late delivery must not resurrect it."""
    pt_id, run_id = await _seed(db)
    async with session_factory() as s:
        run = (await s.execute(select(ExperimentRun).where(ExperimentRun.id == run_id))).scalar_one()
        run.status = "CANCELLED"
        await s.commit()

    assert await run_writeback.claim_for_execution(pt_id) is False

"""M2c — the single idempotent completion path shared by in-process and Celery.

These tests pin the properties a naive "move the code" refactor gets wrong. The
first version of this file recorded ``_finalise_batch`` calls but never asserted
the count, so a double-fire passed silently — every counting assertion below
exists because of that miss.

Not covered here (needs a real MySQL): whether ``FOR UPDATE`` genuinely
serialises two *simultaneous* completions. SQLite has no row locks, so these
tests prove the sequential contract only.
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
from app.services import run_writeback


@pytest.fixture(autouse=True)
def use_test_sessions(session_factory):
    """Point the service's lazily-resolved session factory at the test engine."""
    with patch("app.models.database.async_session_factory", session_factory):
        yield


@pytest.fixture
def spy(monkeypatch):
    """Count the post-commit tails so double-fires can't slip through."""
    calls = {"finalise": [], "mirror": [], "dag": []}

    async def fake_finalise(experiment_id, modeling_task_id):
        calls["finalise"].append((experiment_id, modeling_task_id))

    async def fake_mirror(*, domain_task_id, run_id):
        calls["mirror"].append((domain_task_id, run_id))

    async def fake_dag(platform_task_id):
        calls["dag"].append(platform_task_id)

    monkeypatch.setattr("app.services.tuning_service._finalise_batch", fake_finalise)
    monkeypatch.setattr("app.services.tuning_service._mirror_logs_to_v3", fake_mirror)
    monkeypatch.setattr(run_writeback, "_propagate_dag", fake_dag)
    return calls


async def _seed_trial(db, *, strategy_type: str = "baseline", run_status: str = "PENDING"):
    ds = Dataset(name="wb.csv", file_path="/tmp/wb.csv", file_size=1, row_count=10)
    db.add(ds)
    await db.flush()
    task = ModelingTask(
        name="wb-task", dataset_id=ds.id, target_column="label",
        task_type="classification", objective_metric="accuracy",
    )
    db.add(task)
    await db.flush()
    exp = PlatformExperiment(
        modeling_task_id=task.id, name="wb-batch", strategy_type=strategy_type,
        dataset_id=ds.id, objective_metric="accuracy", status="RUNNING",
    )
    db.add(exp)
    await db.flush()
    ptask = PlatformTask(kind="train", status="QUEUED", payload_ref="train:dom-1")
    db.add(ptask)
    await db.flush()
    run = ExperimentRun(
        experiment_id=exp.id, task_id=ptask.id,
        params={"model_type": "logistic_regression"}, status=run_status,
    )
    db.add(run)
    await db.flush()
    await db.commit()
    return ptask.id, run.id, exp.id


async def _fetch(session_factory, run_id, ptask_id):
    async with session_factory() as s:
        run = (await s.execute(select(ExperimentRun).where(ExperimentRun.id == run_id))).scalar_one()
        task = (await s.execute(select(PlatformTask).where(PlatformTask.id == ptask_id))).scalar_one()
    return run, task


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

async def test_success_commits_run_and_task_together(db, session_factory, spy):
    ptask_id, run_id, exp_id = await _seed_trial(db)

    outcome = await run_writeback.complete_platform_task(
        ptask_id, status="SUCCESS", metrics={"accuracy": 0.9}, domain_task_id="dom-1"
    )

    run, task = await _fetch(session_factory, run_id, ptask_id)
    assert (run.status, task.status) == ("SUCCESS", "SUCCESS")
    assert run.metrics.get("accuracy") == 0.9
    assert task.metrics_snapshot.get("accuracy") == 0.9
    # The outcome reports what was committed, not the caller's input.
    assert outcome.status == "SUCCESS" and outcome.run_id == run_id
    assert len(spy["finalise"]) == 1 and len(spy["dag"]) == 1 and len(spy["mirror"]) == 1


# ---------------------------------------------------------------------------
# Idempotency — the reason this module exists
# ---------------------------------------------------------------------------

async def test_duplicate_success_does_not_refire_tails(db, session_factory, spy):
    ptask_id, run_id, _ = await _seed_trial(db)

    await run_writeback.complete_platform_task(
        ptask_id, status="SUCCESS", metrics={"accuracy": 0.9}, domain_task_id="dom-1"
    )
    second = await run_writeback.complete_platform_task(
        ptask_id, status="SUCCESS", metrics={"accuracy": 0.1}, domain_task_id="dom-1"
    )

    run, _ = await _fetch(session_factory, run_id, ptask_id)
    assert run.metrics.get("accuracy") == 0.9, "second delivery overwrote the result"
    assert second.already_terminal is True
    # The killer assertions the first version of this file was missing:
    assert len(spy["finalise"]) == 1, "batch finalised twice → duplicate SHAP dispatch"
    assert len(spy["dag"]) == 1, "DAG propagated twice → dependents submitted twice"
    assert len(spy["mirror"]) == 1, "logs mirrored twice"


@pytest.mark.parametrize(
    "first,second",
    [("SUCCESS", "FAILED"), ("FAILED", "SUCCESS")],
)
async def test_opposite_outcome_cannot_split_run_and_task(
    db, session_factory, spy, first, second
):
    """A late contradicting delivery must not leave Run and Task disagreeing."""
    ptask_id, run_id, _ = await _seed_trial(db)

    await run_writeback.complete_platform_task(ptask_id, status=first, error="e1")
    await run_writeback.complete_platform_task(ptask_id, status=second, error="e2")

    run, task = await _fetch(session_factory, run_id, ptask_id)
    assert run.status == first, "first terminal state must win"
    assert task.status == run.status, (
        f"state split: run={run.status} task={task.status}"
    )


# ---------------------------------------------------------------------------
# Retry semantics
# ---------------------------------------------------------------------------

async def test_retryable_failure_writes_no_terminal_state(db, session_factory, spy):
    ptask_id, run_id, _ = await _seed_trial(db)

    await run_writeback.complete_platform_task(
        ptask_id, status="FAILED", error="transient boom", final_attempt=False
    )

    run, task = await _fetch(session_factory, run_id, ptask_id)
    assert (run.status, task.status) == ("RETRY", "RETRY")
    # The terminal column stays clean so a run that later succeeds carries no
    # leftover scare text.
    assert run.error_message is None
    assert spy["finalise"] == [] and spy["dag"] == []


async def test_retry_then_success_clears_transient_error(db, session_factory, spy):
    ptask_id, run_id, _ = await _seed_trial(db)

    await run_writeback.complete_platform_task(
        ptask_id, status="FAILED", error="transient boom", final_attempt=False
    )
    # Attempt 2 starts: the run must show RUNNING again, not a stale RETRY.
    assert await run_writeback.mark_run_started(ptask_id) is True
    run, _ = await _fetch(session_factory, run_id, ptask_id)
    assert run.status == "RUNNING"

    await run_writeback.complete_platform_task(ptask_id, status="SUCCESS", metrics={"accuracy": 1.0})

    run, task = await _fetch(session_factory, run_id, ptask_id)
    assert run.status == "SUCCESS"
    assert run.error_message is None
    assert task.error_message is None, "stale attempt error survived a success"


async def test_final_attempt_writes_terminal_failure(db, session_factory, spy):
    ptask_id, run_id, _ = await _seed_trial(db)
    await run_writeback.complete_platform_task(
        ptask_id, status="FAILED", error="fatal boom", final_attempt=True
    )
    run, task = await _fetch(session_factory, run_id, ptask_id)
    assert (run.status, task.status) == ("FAILED", "FAILED")
    assert "fatal boom" in (run.error_message or "")


# ---------------------------------------------------------------------------
# Failure handling — a swallowed write-back is the bug M2c exists to kill
# ---------------------------------------------------------------------------

async def test_terminal_write_failure_propagates(db, monkeypatch):
    """If the authoritative commit fails the caller must learn about it."""
    ptask_id, _, _ = await _seed_trial(db)

    async def boom(*a, **k):
        raise RuntimeError("db down")

    monkeypatch.setattr(run_writeback, "_commit_terminal_state", boom)
    with pytest.raises(RuntimeError, match="db down"):
        await run_writeback.complete_platform_task(ptask_id, status="SUCCESS", metrics={})


async def test_unknown_platform_task_raises(db):
    with pytest.raises(ValueError, match="not found"):
        await run_writeback.complete_platform_task("no-such-task", status="SUCCESS")


# ---------------------------------------------------------------------------
# Strategy-specific + non-V3 behaviour
# ---------------------------------------------------------------------------

async def test_bayesian_trial_is_not_finalised_per_trial(db, spy):
    """Bayesian creates runs incrementally — per-trial finalisation would close
    the experiment after trial 1 while the study is still running."""
    ptask_id, _, _ = await _seed_trial(db, strategy_type="bayesian_search")
    await run_writeback.complete_platform_task(ptask_id, status="SUCCESS", metrics={"accuracy": 0.8})
    assert spy["finalise"] == []
    assert len(spy["dag"]) == 1, "DAG must still propagate for bayesian trials"


async def test_non_v3_task_completes_without_run(db, session_factory, spy):
    ptask = PlatformTask(kind="explain", status="QUEUED", payload_ref="explain:dom-x")
    db.add(ptask)
    await db.flush()
    await db.commit()

    outcome = await run_writeback.complete_platform_task(ptask.id, status="SUCCESS", metrics={})
    assert outcome.run_id is None

    async with session_factory() as s:
        refreshed = (await s.execute(select(PlatformTask).where(PlatformTask.id == ptask.id))).scalar_one()
    assert refreshed.status == "SUCCESS"
    assert spy["finalise"] == [] and len(spy["dag"]) == 1


# ---------------------------------------------------------------------------
# Claim semantics
# ---------------------------------------------------------------------------

async def test_mark_run_started_claims_and_keeps_first_started_at(db, session_factory):
    ptask_id, run_id, _ = await _seed_trial(db)

    assert await run_writeback.mark_run_started(ptask_id) is True
    run, _ = await _fetch(session_factory, run_id, ptask_id)
    first_started = run.started_at
    assert run.status == "RUNNING" and first_started is not None

    assert await run_writeback.mark_run_started(ptask_id) is True
    run, _ = await _fetch(session_factory, run_id, ptask_id)
    assert run.started_at == first_started, "started_at must not be bumped"


async def test_mark_run_started_refuses_to_reopen_terminal_run(db, session_factory):
    """A duplicate delivery must not resurrect a finished run and re-execute it."""
    ptask_id, run_id, _ = await _seed_trial(db)
    await run_writeback.complete_platform_task(ptask_id, status="SUCCESS", metrics={"accuracy": 0.9})

    assert await run_writeback.mark_run_started(ptask_id) is False
    run, _ = await _fetch(session_factory, run_id, ptask_id)
    assert run.status == "SUCCESS"

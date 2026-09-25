"""M2c round 4 — the three holes Codex found in the previous pass.

1. A cancelled task could be resurrected: the real cancel path writes
   CANCELLED to the PlatformTask only, and the claim looked at the Run.
2. A WritebackError left the row non-terminal with nothing able to re-drive it,
   so "left for the reconciler" was a comment, not a mechanism.
3. Bundle pre-flight only covered three conditions, so a later batch could
   still 422 *after* an earlier one had committed and started training.
"""
from __future__ import annotations

from datetime import timedelta
from unittest.mock import patch

import pytest
from fastapi import HTTPException
from sqlalchemy import select

from app.models.database import (
    Dataset,
    ExperimentRun,
    ModelingTask,
    PlatformExperiment,
    PlatformTask,
)
from app.scheduler import scheduler as sched
from app.scheduler import task_runner
from app.services import run_writeback, tuning_service


@pytest.fixture(autouse=True)
def use_test_sessions(session_factory):
    with patch("app.models.database.async_session_factory", session_factory):
        yield


async def _seed(db, *, kind="rec_kind", task_status="QUEUED", run_status="PENDING"):
    ds = Dataset(name="r.csv", file_path="/tmp/r.csv", file_size=1, row_count=5)
    db.add(ds)
    await db.flush()
    mt = ModelingTask(
        name="r", dataset_id=ds.id, target_column="y",
        task_type="classification", objective_metric="accuracy",
    )
    db.add(mt)
    await db.flush()
    exp = PlatformExperiment(
        modeling_task_id=mt.id, name="r-batch", strategy_type="baseline",
        dataset_id=ds.id, objective_metric="accuracy", status="RUNNING",
    )
    db.add(exp)
    await db.flush()
    pt = PlatformTask(kind=kind, status=task_status, payload_ref=f"{kind}:dom-1")
    db.add(pt)
    await db.flush()
    run = ExperimentRun(
        experiment_id=exp.id, task_id=pt.id,
        params={"model_type": "logistic_regression"}, status=run_status,
    )
    db.add(run)
    await db.flush()
    await db.commit()
    return pt.id, run.id, mt.id


# ---------------------------------------------------------------------------
# 1. Cancellation must not be reversible by a late delivery
# ---------------------------------------------------------------------------

async def test_real_cancel_path_blocks_a_late_claim(db, session_factory):
    """The regression: cancel writes only the Task, so a Run-only terminality
    check let a duplicate delivery drag both rows back to RUNNING."""
    pt_id, run_id, _ = await _seed(db, task_status="QUEUED", run_status="PENDING")

    await task_runner.cancel_task(db, pt_id)
    await db.commit()

    assert await run_writeback.claim_for_execution(pt_id) is False

    async with session_factory() as s:
        task = (await s.execute(select(PlatformTask).where(PlatformTask.id == pt_id))).scalar_one()
        run = (await s.execute(select(ExperimentRun).where(ExperimentRun.id == run_id))).scalar_one()
    assert task.status == "CANCELLED"
    assert run.status == "CANCELLED", "cancel left the Run claimable"


async def test_claim_refuses_on_a_terminal_task_alone(db, session_factory):
    """Isolates the claim contract from the cancel fix.

    The test above passes even with the old Run-only check, because the cancel
    path now terminalises both rows. This one pins the claim itself: a Task
    that is terminal while its Run is not — a row cancelled before this fix
    shipped, or a cancel that raced a claim — must still be unclaimable.
    """
    pt_id, run_id, _ = await _seed(db, task_status="RUNNING", run_status="RUNNING")
    async with session_factory() as s:
        task = (await s.execute(select(PlatformTask).where(PlatformTask.id == pt_id))).scalar_one()
        task.status = "CANCELLED"
        await s.commit()

    assert await run_writeback.claim_for_execution(pt_id) is False, (
        "a terminal PlatformTask was re-claimed because only the Run was checked"
    )

    async with session_factory() as s:
        run = (await s.execute(select(ExperimentRun).where(ExperimentRun.id == run_id))).scalar_one()
        task = (await s.execute(select(PlatformTask).where(PlatformTask.id == pt_id))).scalar_one()
    assert task.status == "CANCELLED"
    assert run.status == "RUNNING", "a refused claim must not write anything"


async def test_cancel_refuses_when_the_completion_already_won(db, session_factory):
    """Cancel must not split the pair.

    The Task is still RUNNING but its Run already committed SUCCESS — the
    shape a completion/cancel race produces. Cancelling the Task anyway would
    leave ``Task=CANCELLED / Run=SUCCESS``, so it is refused instead.
    """
    pt_id, run_id, _ = await _seed(db, task_status="RUNNING", run_status="RUNNING")
    async with session_factory() as s:
        run = (await s.execute(select(ExperimentRun).where(ExperimentRun.id == run_id))).scalar_one()
        run.status = "SUCCESS"
        await s.commit()

    with pytest.raises(ValueError, match="already finished"):
        await task_runner.cancel_task(db, pt_id)

    async with session_factory() as s:
        run = (await s.execute(select(ExperimentRun).where(ExperimentRun.id == run_id))).scalar_one()
        task = (await s.execute(select(PlatformTask).where(PlatformTask.id == pt_id))).scalar_one()
    assert run.status == "SUCCESS"
    assert task.status != "CANCELLED", "cancel split the Task away from its Run"


async def test_cancel_closes_the_batch(db, session_factory, monkeypatch):
    """Cancelling the last outstanding trial produces no completion callback,
    so cancel itself must run finalisation — otherwise the batch hangs."""
    pt_id, run_id, mt_id = await _seed(db)
    async with session_factory() as s:
        run = (await s.execute(select(ExperimentRun).where(ExperimentRun.id == run_id))).scalar_one()
        exp_id = run.experiment_id

    finalised: list[tuple[str, str]] = []

    async def spy(experiment_id, modeling_task_id):
        finalised.append((experiment_id, modeling_task_id))

    propagated: list[str] = []

    async def spy_dag(task_id):
        propagated.append(task_id)

    monkeypatch.setattr(tuning_service, "_finalise_batch", spy)
    monkeypatch.setattr(run_writeback, "_propagate_dag", spy_dag)

    await task_runner.cancel_task(db, pt_id)

    assert finalised == [(exp_id, mt_id)], "cancel left the batch un-finalised"
    assert propagated == [pt_id], "cancel did not release downstream dependents"


async def test_cancelled_runs_let_the_batch_finalise(db, session_factory):
    """CANCELLED must count as done, or the batch hangs at done < total."""
    pt_id, run_id, mt_id = await _seed(db)
    async with session_factory() as s:
        run = (await s.execute(select(ExperimentRun).where(ExperimentRun.id == run_id))).scalar_one()
        run.status = "CANCELLED"
        exp_id = run.experiment_id
        await s.commit()

    # ``_finalise_batch`` binds async_session_factory at import time.
    with patch.object(tuning_service, "async_session_factory", session_factory), \
            patch.object(tuning_service, "_schedule_shap_for_top_runs", _noop):
        await tuning_service._finalise_batch(exp_id, mt_id)

    async with session_factory() as s:
        exp = (await s.execute(
            select(PlatformExperiment).where(PlatformExperiment.id == exp_id)
        )).scalar_one()
    assert exp.status in {"COMPLETED", "FAILED"}, "batch never settled with a cancelled trial"


async def _noop(*a, **k):
    return None


# ---------------------------------------------------------------------------
# 2. Stalled-task recovery — the WritebackError tail
# ---------------------------------------------------------------------------

async def test_recover_stalled_requeues_and_resubmits(db, session_factory, monkeypatch):
    """A task stuck RUNNING past the threshold gets re-armed and resubmitted."""
    pt_id, run_id, _ = await _seed(db, task_status="RUNNING", run_status="RUNNING")
    async with session_factory() as s:
        task = (await s.execute(select(PlatformTask).where(PlatformTask.id == pt_id))).scalar_one()
        task.started_at = sched._utcnow() - timedelta(hours=12)
        task.celery_task_id = "dead-worker-task"
        await s.commit()

    submitted: list[str] = []

    class _FakeCelery(sched.CeleryScheduler):
        async def submit(self, task_id):
            submitted.append(task_id)

    monkeypatch.setattr(sched, "get_scheduler", lambda kind: _FakeCelery())

    recovered = await sched.recover_stalled_tasks(older_than=timedelta(hours=6))

    assert recovered == [pt_id]
    assert submitted == [pt_id]
    async with session_factory() as s:
        task = (await s.execute(select(PlatformTask).where(PlatformTask.id == pt_id))).scalar_one()
        run = (await s.execute(select(ExperimentRun).where(ExperimentRun.id == run_id))).scalar_one()
    assert task.status == "QUEUED"
    assert task.celery_task_id is None, "stale broker id would block resubmission"
    assert task.attempt_token is None, "the dead attempt's token must be cleared"
    assert run.status == "PENDING"


async def test_recover_stalled_ignores_fresh_tasks(db, session_factory, monkeypatch):
    """A trial that is merely slow must not be duplicated."""
    pt_id, _, _ = await _seed(db, task_status="RUNNING", run_status="RUNNING")
    async with session_factory() as s:
        task = (await s.execute(select(PlatformTask).where(PlatformTask.id == pt_id))).scalar_one()
        task.started_at = sched._utcnow() - timedelta(minutes=5)
        await s.commit()

    monkeypatch.setattr(sched, "get_scheduler", lambda kind: sched.CeleryScheduler())
    assert await sched.recover_stalled_tasks(older_than=timedelta(hours=6)) == []


async def test_recover_stalled_skips_terminal_and_inprocess(db, session_factory, monkeypatch):
    """Terminal rows are never re-armed; in-process kinds are out of scope."""
    pt_id, _, _ = await _seed(db, task_status="RUNNING", run_status="RUNNING")
    async with session_factory() as s:
        task = (await s.execute(select(PlatformTask).where(PlatformTask.id == pt_id))).scalar_one()
        task.started_at = sched._utcnow() - timedelta(hours=12)
        task.status = "SUCCESS"
        await s.commit()

    monkeypatch.setattr(sched, "get_scheduler", lambda kind: sched.CeleryScheduler())
    assert await sched.recover_stalled_tasks(older_than=timedelta(hours=6)) == []

    # …and an in-process kind is skipped even when stalled.
    async with session_factory() as s:
        task = (await s.execute(select(PlatformTask).where(PlatformTask.id == pt_id))).scalar_one()
        task.status = "RUNNING"
        await s.commit()
    monkeypatch.setattr(sched, "get_scheduler", lambda kind: sched.InProcessScheduler())
    assert await sched.recover_stalled_tasks(older_than=timedelta(hours=6)) == []


# ---------------------------------------------------------------------------
# 3. Bundle pre-flight is all-or-nothing
# ---------------------------------------------------------------------------

async def _modeling_task(db):
    ds = Dataset(name="b.csv", file_path="/tmp/b.csv", file_size=1, row_count=50)
    db.add(ds)
    await db.flush()
    mt = ModelingTask(
        name="b", dataset_id=ds.id, target_column="y",
        task_type="classification", objective_metric="accuracy",
    )
    db.add(mt)
    await db.flush()
    await db.commit()
    return mt.id


@pytest.mark.parametrize(
    "bad_spec, expected",
    [
        ({"strategy_type": "grid_search", "selected_models": ["mlp_dl"]}, "baseline"),
        ({"strategy_type": "baseline", "selected_models": ["no_such_model"]}, "Unknown"),
        ({"strategy_type": "nonsense", "selected_models": ["logistic_regression"]}, "strategy_type"),
        (
            {
                "strategy_type": "baseline",
                "selected_models": ["logistic_regression"],
                "budget_config": {"test_size": "not-a-number"},
            },
            "numeric",
        ),
        (
            {
                "strategy_type": "baseline",
                "selected_models": ["logistic_regression"],
                "budget_config": {"cv_folds": 1},
            },
            "cv_folds",
        ),
    ],
)
async def test_bundle_rejects_before_launching_anything(db, bad_spec, expected):
    """The first batch is valid; the second is not. Nothing may start.

    Each of these used to be validated *inside* the per-batch path, i.e. only
    after batch #1 had committed and launched.
    """
    mt_id = await _modeling_task(db)
    launched: list[str] = []

    async def spy_launch(exp_id, modeling_task_id):
        launched.append(exp_id)

    with patch.object(tuning_service, "_launch_concurrent", spy_launch):
        with pytest.raises(HTTPException) as exc:
            await tuning_service.dispatch_experiment_bundle(
                db,
                modeling_task_id=mt_id,
                name="bundle",
                strategies=[
                    {"strategy_type": "baseline", "selected_models": ["logistic_regression"]},
                    bad_spec,
                ],
            )

    assert exc.value.status_code == 422
    assert expected in str(exc.value.detail)
    assert "strategy #2" in str(exc.value.detail), "error should name the offending item"
    assert launched == [], "a batch was launched despite the bundle being rejected"

    rows = (await db.execute(select(PlatformExperiment))).scalars().all()
    assert rows == [], "a partial experiment was persisted for a rejected bundle"


# ---------------------------------------------------------------------------
# Round 5 — budget coercion, error prefixes, and the recovery CAS
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "budget, expected",
    [
        ({"test_size": 0}, "test_size"),
        ({"cv_folds": 0}, "cv_folds"),
        ({"max_trials": 0}, "max_trials"),
    ],
)
async def test_explicit_zero_is_rejected_not_defaulted(db, budget, expected):
    """`or <default>` swallowed an explicit 0, so an invalid budget silently
    became the default instead of a 422."""
    mt_id = await _modeling_task(db)
    task = (await db.execute(select(ModelingTask).where(ModelingTask.id == mt_id))).scalar_one()

    with pytest.raises(HTTPException) as exc:
        tuning_service._preflight_batch(
            task,
            strategy_type="baseline",
            selected_models=["logistic_regression"],
            search_space={},
            budget_config=budget,
        )
    assert expected in str(exc.value.detail)


async def test_budget_is_normalised_for_downstream_consumers(db):
    """Regression: preflight coerced max_trials locally but left the raw
    string in budget_config, which the background bayesian study reads —
    the API reported success and the experiment then died on a TypeError."""
    mt_id = await _modeling_task(db)
    task = (await db.execute(select(ModelingTask).where(ModelingTask.id == mt_id))).scalar_one()

    pf = tuning_service._preflight_batch(
        task,
        strategy_type="baseline",
        selected_models=["logistic_regression"],
        search_space={},
        budget_config={"max_trials": "5", "test_size": "0.3", "cv_folds": "4"},
    )
    assert pf.budget_config["max_trials"] == 5
    assert isinstance(pf.budget_config["max_trials"], int)
    assert pf.budget_config["test_size"] == 0.3
    assert pf.budget_config["cv_folds"] == 4


async def test_bundle_prefix_covers_helper_raised_errors(db):
    """`_validate_search_space` raises its own HTTPException; without
    re-wrapping, exactly the errors a bundle caller needs to locate arrived
    with no "strategy #N"."""
    mt_id = await _modeling_task(db)
    with pytest.raises(HTTPException) as exc:
        await tuning_service.dispatch_experiment_bundle(
            db,
            modeling_task_id=mt_id,
            name="bundle",
            strategies=[
                {"strategy_type": "baseline", "selected_models": ["logistic_regression"]},
                {
                    "strategy_type": "grid_search",
                    "selected_models": ["logistic_regression"],
                    "search_space": {"logistic_regression": {"C": "not-a-list"}},
                },
            ],
        )
    assert "strategy #2" in str(exc.value.detail)


async def test_recover_skips_a_task_reclaimed_after_the_scan(db, session_factory, monkeypatch):
    """CAS must key on the observed attempt, not merely on status.

    ``recover_stalled_tasks`` opens one session to scan and another to write.
    Wrapping the factory so the *second* open first lands a fresh claim
    reproduces the race deterministically — no fire-and-forget ordering.

    Without the identity predicate the UPDATE matches on status alone and
    resets a task that is legitimately running again.
    """
    pt_id, _, _ = await _seed(db, task_status="RUNNING", run_status="RUNNING")
    async with session_factory() as s:
        task = (await s.execute(select(PlatformTask).where(PlatformTask.id == pt_id))).scalar_one()
        task.started_at = sched._utcnow() - timedelta(hours=12)
        task.attempt_token = "old-attempt-token"
        task.celery_task_id = "old-attempt"
        await s.commit()

    submitted: list[str] = []

    class _FakeCelery(sched.CeleryScheduler):
        async def submit(self, task_id):
            submitted.append(task_id)

    monkeypatch.setattr(sched, "get_scheduler", lambda kind: _FakeCelery())

    async def _reclaim():
        """A fresh worker claims the task — via the REAL claim path.

        Hand-editing the columns would only prove "CAS works when the identity
        changed". What actually needs proving is that a genuine claim *mints* a
        new identity, which is why this calls claim_for_execution itself.
        """
        assert await run_writeback.claim_for_execution(pt_id) is True

    calls = {"n": 0}

    class _RacingFactory:
        def __call__(self):
            calls["n"] += 1
            return _RacingSession(calls["n"])

    class _RacingSession:
        def __init__(self, n):
            self._n = n
            self._cm = None

        async def __aenter__(self):
            if self._n == 2:  # the write session — race in just before it
                await _reclaim()
            self._cm = session_factory()
            return await self._cm.__aenter__()

        async def __aexit__(self, *exc):
            return await self._cm.__aexit__(*exc)

    with patch("app.models.database.async_session_factory", _RacingFactory()):
        recovered = await sched.recover_stalled_tasks(older_than=timedelta(hours=6))

    async with session_factory() as s:
        task = (await s.execute(select(PlatformTask).where(PlatformTask.id == pt_id))).scalar_one()

    assert recovered == [], "recovery re-queued a task that had just been re-claimed"
    assert submitted == [], "recovery resubmitted a task that is already running"
    assert task.status == "RUNNING"
    assert task.attempt_token is not None, "the fresh attempt was clobbered"
    assert task.attempt_token != "old-attempt-token"


async def test_recovery_sweep_runs_both_repairs_and_survives_failure(monkeypatch):
    """The sweep is where ``on_failure`` delegates correctness, so it must
    actually call both repairs — and a transient failure in one must not kill
    the loop for the rest of the process lifetime."""
    import asyncio

    from app import main as app_main

    calls: list[str] = []

    async def failing_reconcile():
        calls.append("reconcile")
        raise RuntimeError("transient DB outage")

    async def recover(*, older_than):
        calls.append("recover")
        return []

    monkeypatch.setattr(app_main, "get_settings", lambda: _SweepSettings())
    monkeypatch.setattr(app_main, "_SWEEP_MIN_INTERVAL_SECONDS", 0)

    task = asyncio.create_task(
        app_main._recovery_sweep_loop(failing_reconcile, recover)
    )
    for _ in range(200):
        await asyncio.sleep(0)
        if calls.count("recover") >= 2:
            break
    task.cancel()
    with __import__("contextlib").suppress(asyncio.CancelledError):
        await task

    assert "reconcile" in calls
    assert calls.count("recover") >= 2, (
        f"loop stopped after a failing repair: {calls}"
    )


class _SweepSettings:
    """Minimum interval so the test does not sleep for real."""
    recovery_sweep_interval_seconds = 0
    stalled_task_timeout_seconds = 3600


# ---------------------------------------------------------------------------
# Round 6 — CAS gates, attempt identity, bayesian cancellation
# ---------------------------------------------------------------------------

async def test_cancel_cas_catches_a_stale_read(db, session_factory):
    """Reproduces the SQLite race: no row locks, so the decision read can be
    stale by the time the write happens.

    Loading the Run into the session's identity map and then terminalising it
    from another session gives ``cancel_task`` exactly what a concurrent
    completion would give it — a cached object still claiming RUNNING. Only the
    conditional UPDATE's rowcount can catch that; the early status check cannot.
    """
    pt_id, run_id, _ = await _seed(db, task_status="RUNNING", run_status="RUNNING")

    # Populate the identity map with the pre-race state.
    stale = (await db.execute(select(ExperimentRun).where(ExperimentRun.id == run_id))).scalar_one()
    assert stale.status == "RUNNING"

    async with session_factory() as other:
        run = (await other.execute(select(ExperimentRun).where(ExperimentRun.id == run_id))).scalar_one()
        run.status = "SUCCESS"
        await other.commit()

    with pytest.raises(ValueError, match="already finished|finished before"):
        await task_runner.cancel_task(db, pt_id)

    async with session_factory() as s:
        run = (await s.execute(select(ExperimentRun).where(ExperimentRun.id == run_id))).scalar_one()
        task = (await s.execute(select(PlatformTask).where(PlatformTask.id == pt_id))).scalar_one()
    assert run.status == "SUCCESS"
    assert task.status != "CANCELLED", "cancel split the pair despite the CAS gate"


async def test_completion_cas_refuses_a_cancelled_run(db, session_factory):
    """The mirror direction: cancel won, so the completion must be a no-op and
    report the cancellation rather than overwriting it with SUCCESS."""
    pt_id, run_id, _ = await _seed(db, task_status="RUNNING", run_status="RUNNING")
    async with session_factory() as s:
        run = (await s.execute(select(ExperimentRun).where(ExperimentRun.id == run_id))).scalar_one()
        run.status = "CANCELLED"
        await s.commit()

    outcome = await run_writeback.complete_platform_task(
        pt_id, status="SUCCESS", metrics={"accuracy": 0.99}
    )

    assert outcome.already_terminal is True
    assert outcome.status == "CANCELLED"
    async with session_factory() as s:
        run = (await s.execute(select(ExperimentRun).where(ExperimentRun.id == run_id))).scalar_one()
    assert run.status == "CANCELLED", "a completion overwrote a cancellation"


async def test_claim_mints_a_new_attempt_token_every_time(db, session_factory):
    """The token is the attempt identity recovery keys on, so a *second* claim
    must not reuse the first one's value."""
    pt_id, _, _ = await _seed(db)

    assert await run_writeback.claim_for_execution(pt_id) is True
    async with session_factory() as s:
        first = (await s.execute(select(PlatformTask).where(PlatformTask.id == pt_id))).scalar_one().attempt_token
    assert first

    assert await run_writeback.claim_for_execution(pt_id) is True
    async with session_factory() as s:
        second = (await s.execute(select(PlatformTask).where(PlatformTask.id == pt_id))).scalar_one().attempt_token

    assert second and second != first, "a re-claim reused the previous attempt token"


async def test_cancelling_a_bayesian_trial_does_not_close_the_study(db, session_factory, monkeypatch):
    """bayesian_search creates runs incrementally, so finalising on one
    cancelled trial would mark the experiment FAILED while the study keeps
    producing trials. Cancellation must apply the same exclusion the
    completion path already applies."""
    pt_id, run_id, _ = await _seed(db)
    async with session_factory() as s:
        run = (await s.execute(select(ExperimentRun).where(ExperimentRun.id == run_id))).scalar_one()
        exp = (await s.execute(
            select(PlatformExperiment).where(PlatformExperiment.id == run.experiment_id)
        )).scalar_one()
        exp.strategy_type = "bayesian_search"
        await s.commit()

    finalised: list[str] = []

    async def spy(experiment_id, modeling_task_id):
        finalised.append(experiment_id)

    async def spy_dag(task_id):
        return None

    monkeypatch.setattr(tuning_service, "_finalise_batch", spy)
    monkeypatch.setattr(run_writeback, "_propagate_dag", spy_dag)

    await task_runner.cancel_task(db, pt_id)

    assert finalised == [], "cancelling one bayesian trial closed the whole study"


# ---------------------------------------------------------------------------
# TD-5 — bayesian must actually reach the TPE modelling stage
# ---------------------------------------------------------------------------

def test_default_budget_leaves_room_for_tpe():
    """The regression this guards: Optuna's default n_startup_trials is 10 and
    this platform's default n_trials_per_model is also 10, so a default
    「贝叶斯搜索」 spent its entire budget in the random startup phase and never
    modelled anything — random search wearing a Bayesian label."""
    startup = tuning_service._tpe_startup_trials(10)
    assert startup < 10, "default budget still spends every trial on random startup"
    assert 10 - startup >= 5, "too few trials left for TPE to be worth the label"


@pytest.mark.parametrize(
    "n_trials, expected",
    [
        (1, 3),      # floor: never model on nothing
        (5, 3),
        (10, 3),
        (15, 5),
        (30, 10),    # ceiling: Optuna's own default once affordable
        (100, 10),
    ],
)
def test_startup_trials_scale_with_budget(n_trials, expected):
    assert tuning_service._tpe_startup_trials(n_trials) == expected


def test_explicit_startup_trials_wins():
    assert tuning_service._tpe_startup_trials(10, {"n_startup_trials": 7}) == 7
    # …but never zero: TPE needs at least one observation.
    assert tuning_service._tpe_startup_trials(10, {"n_startup_trials": 0}) == 1


async def test_startup_migration_covers_every_alembic_column(tmp_path):
    """The non-production bootstrap must not fall behind Alembic.

    `datasets.content_sha256` shipped as Alembic 0002 but had no counterpart
    here, so a dev database created through create_all + startup migrations
    permanently lacked it and every dataset query failed. This pins the
    specific gap and the general rule.
    """
    from sqlalchemy import create_engine, inspect

    from app.core.migrations import (
        _migrate_datasets_content_sha256,
        _migrate_platform_tasks_attempt_token,
    )
    from app.models.database import Base

    db_file = tmp_path / "legacy.db"
    engine = create_engine(f"sqlite:///{db_file}")
    Base.metadata.create_all(engine)

    # Simulate a database created before those columns existed.
    with engine.connect() as c:
        c.exec_driver_sql("ALTER TABLE datasets DROP COLUMN content_sha256")
        c.exec_driver_sql("ALTER TABLE platform_tasks DROP COLUMN attempt_token")
        c.commit()

    with engine.connect() as c:
        _migrate_datasets_content_sha256(c)
        _migrate_platform_tasks_attempt_token(c)
        c.commit()

    datasets = {col["name"] for col in inspect(engine).get_columns("datasets")}
    tasks = {col["name"] for col in inspect(engine).get_columns("platform_tasks")}
    assert "content_sha256" in datasets
    assert "attempt_token" in tasks
    engine.dispose()

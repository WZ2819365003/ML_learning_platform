from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from sqlalchemy import select

from app.models.database import PlatformTask


def test_queue_contract_accepts_declared_targets_and_rejects_missing_queue():
    from app.scheduler.queues import DECLARED_QUEUES, KIND_TO_QUEUE, assert_queue_contract

    assert_queue_contract(DECLARED_QUEUES, KIND_TO_QUEUE)

    broken_routes = {**KIND_TO_QUEUE, "broken_kind": "missing_queue"}
    try:
        assert_queue_contract(DECLARED_QUEUES, broken_routes)
    except RuntimeError as exc:
        assert "broken_kind" in str(exc)
        assert "missing_queue" in str(exc)
    else:  # pragma: no cover - assertion helper
        raise AssertionError("queue contract accepted a route to an undeclared queue")


def test_get_scheduler_routes_per_kind_and_keeps_global_celery_compatibility(monkeypatch):
    import app.scheduler.scheduler as scheduler_module

    scheduler_module.set_scheduler(None)
    monkeypatch.setattr(
        "app.config.get_settings",
        lambda: SimpleNamespace(scheduler_mode="inprocess", celery_kinds="explain"),
    )
    assert isinstance(scheduler_module.get_scheduler("explain"), scheduler_module.CeleryScheduler)
    assert isinstance(scheduler_module.get_scheduler("train"), scheduler_module.InProcessScheduler)

    scheduler_module.set_scheduler(None)
    monkeypatch.setattr(
        "app.config.get_settings",
        lambda: SimpleNamespace(scheduler_mode="inprocess", celery_kinds=""),
    )
    assert isinstance(scheduler_module.get_scheduler("explain"), scheduler_module.InProcessScheduler)
    assert isinstance(scheduler_module.get_scheduler("train"), scheduler_module.InProcessScheduler)

    scheduler_module.set_scheduler(None)
    monkeypatch.setattr(
        "app.config.get_settings",
        lambda: SimpleNamespace(scheduler_mode="celery", celery_kinds=""),
    )
    assert isinstance(scheduler_module.get_scheduler("anything"), scheduler_module.CeleryScheduler)
    scheduler_module.set_scheduler(None)


async def test_celery_submit_routes_and_persists_returned_task_id(db, session_factory):
    from app.scheduler.scheduler import CeleryScheduler

    task = PlatformTask(kind="explain", status="QUEUED", payload_ref="explain:run-1")
    db.add(task)
    await db.commit()
    task_id = task.id

    async_result = SimpleNamespace(id="celery-result-123")
    with (
        patch("app.models.database.async_session_factory", session_factory),
        patch(
            "app.scheduler.celery_tasks.run_platform_task_generic.apply_async",
            return_value=async_result,
        ) as apply_async,
    ):
        await CeleryScheduler().submit(task_id)

    apply_async.assert_called_once_with(args=[task_id], queue="explain")
    async with session_factory() as verify_db:
        persisted = await verify_db.get(PlatformTask, task_id)
        assert persisted.celery_task_id == "celery-result-123"


async def test_reconciler_resubmits_only_stale_celery_tasks(db, session_factory, monkeypatch):
    from app.scheduler.scheduler import CeleryScheduler, reconcile_queued_tasks

    now = datetime.now(timezone.utc)
    stale = PlatformTask(
        kind="explain",
        status="QUEUED",
        payload_ref="explain:stale",
        queued_at=now - timedelta(minutes=3),
    )
    fresh = PlatformTask(
        kind="explain",
        status="QUEUED",
        payload_ref="explain:fresh",
        queued_at=now - timedelta(seconds=30),
    )
    inprocess = PlatformTask(
        kind="train",
        status="QUEUED",
        payload_ref="train:legacy",
        queued_at=now - timedelta(minutes=3),
    )
    db.add_all([stale, fresh, inprocess])
    await db.commit()

    celery_scheduler = CeleryScheduler()
    celery_scheduler.submit = AsyncMock()

    def fake_get_scheduler(kind: str):
        return celery_scheduler if kind == "explain" else SimpleNamespace()

    monkeypatch.setattr("app.scheduler.scheduler.get_scheduler", fake_get_scheduler)
    with patch("app.models.database.async_session_factory", session_factory):
        submitted = await reconcile_queued_tasks(older_than=timedelta(minutes=2), now=now)

    assert submitted == [stale.id]
    celery_scheduler.submit.assert_awaited_once_with(stale.id)


async def test_inprocess_scheduler_preserves_v3_trial_writeback_wrapper(
    db, session_factory, monkeypatch
):
    from app.models.database import ExperimentRun, ModelingTask, PlatformExperiment
    from app.scheduler.scheduler import InProcessScheduler
    from app.services import tuning_service

    modeling_task = ModelingTask(
        name="scheduler-contract",
        task_type="classification",
        objective_metric="accuracy",
        status="RUNNING",
    )
    db.add(modeling_task)
    await db.flush()
    experiment = PlatformExperiment(
        modeling_task_id=modeling_task.id,
        name="v3-batch",
        kind="baseline",
        strategy_type="baseline",
        objective_metric="accuracy",
        objective_direction="max",
        status="RUNNING",
        config={"submitted_from": "v3_workbench"},
    )
    db.add(experiment)
    await db.flush()
    platform_task = PlatformTask(
        kind="train", status="QUEUED", payload_ref="train:domain-1"
    )
    db.add(platform_task)
    await db.flush()
    run = ExperimentRun(
        experiment_id=experiment.id,
        task_id=platform_task.id,
        status="PENDING",
    )
    db.add(run)
    await db.commit()

    called = []

    async def fake_execute(domain_id, platform_id, run_id, experiment_id, *, kind):
        called.append((domain_id, platform_id, run_id, experiment_id, kind))
        return {"status": "SUCCESS", "metrics": {}}

    monkeypatch.setattr(tuning_service, "_execute_single_trial", fake_execute)
    monkeypatch.setattr(tuning_service, "async_session_factory", session_factory)
    with patch("app.models.database.async_session_factory", session_factory):
        handle = await InProcessScheduler().submit(platform_task.id)
        assert isinstance(handle, asyncio.Task)
        await handle

    assert called == [
        ("domain-1", platform_task.id, run.id, experiment.id, "train")
    ]


def test_generic_celery_task_uses_exponential_backoff():
    from app.scheduler.celery_tasks import run_platform_task_generic

    assert run_platform_task_generic.retry_backoff is True


def test_executor_module_loader_has_one_explicit_import_list():
    from app.scheduler import executors

    with patch("app.scheduler.executors.import_module") as import_module:
        executors.load_executor_modules()

    # The list is explicit on purpose: an executor that is never imported is
    # never registered, and the failure surfaces far away as "no executor for
    # kind=X" at dispatch time. Adding a kind means adding it here.
    assert [call.args[0] for call in import_module.call_args_list] == [
        "app.services.training_service",
        "app.services.dl_service",
        "app.services.explain_service",
        "app.services.batch_prediction_service",
    ]

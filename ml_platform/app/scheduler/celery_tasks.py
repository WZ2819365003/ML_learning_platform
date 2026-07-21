"""
Celery task definitions for the ML Platform V3 scheduler.

Each task:
1. Looks up the PlatformTask record in MySQL
2. Updates status to RUNNING
3. Dispatches to the appropriate domain service
4. Writes back metrics/status on completion
5. Handles retry on failure
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone

from celery import Task as CeleryTask
from celery.exceptions import SoftTimeLimitExceeded

from app.scheduler.celery_app import celery_app

logger = logging.getLogger(__name__)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _run_async(coro):
    """Run an async coroutine from a sync Celery task."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


# ---------------------------------------------------------------------------
# Base task with retry hooks
# ---------------------------------------------------------------------------

class MLBaseTask(CeleryTask):
    abstract = True

    def on_failure(self, exc, task_id, args, kwargs, einfo):
        platform_task_id = kwargs.get("platform_task_id") or (args[0] if args else None)
        if platform_task_id:
            _run_async(_mark_task_failed(platform_task_id, str(exc)))
            from app.scheduler.scheduler import CeleryScheduler

            _run_async(CeleryScheduler().on_task_done(platform_task_id))
        logger.error("Task %s failed: %s", task_id, exc)

    def on_retry(self, exc, task_id, args, kwargs, einfo):
        platform_task_id = kwargs.get("platform_task_id") or (args[0] if args else None)
        if platform_task_id:
            _run_async(_mark_task_retry(platform_task_id, str(exc)))
        logger.warning("Task %s retrying: %s", task_id, exc)


# ---------------------------------------------------------------------------
# DB helpers (async, run inside _run_async)
# ---------------------------------------------------------------------------

async def _mark_task_running(platform_task_id: str, celery_id: str) -> None:
    from app.models.database import PlatformTask, async_session_factory
    from sqlalchemy import select
    async with async_session_factory() as db:
        result = await db.execute(
            select(PlatformTask).where(PlatformTask.id == platform_task_id)
        )
        task = result.scalar_one_or_none()
        if task:
            task.status = "RUNNING"
            task.celery_task_id = celery_id
            task.started_at = _utcnow()
            await db.commit()


async def _mark_task_success(platform_task_id: str, metrics: dict | None = None) -> None:
    from app.models.database import PlatformTask, async_session_factory
    from sqlalchemy import select
    async with async_session_factory() as db:
        result = await db.execute(
            select(PlatformTask).where(PlatformTask.id == platform_task_id)
        )
        task = result.scalar_one_or_none()
        if task:
            task.status = "SUCCESS"
            task.finished_at = _utcnow()
            task.progress = 1.0
            if metrics:
                task.metrics_snapshot = metrics
            await db.commit()


async def _mark_task_failed(platform_task_id: str, error: str) -> None:
    from app.models.database import PlatformTask, async_session_factory
    from sqlalchemy import select
    async with async_session_factory() as db:
        result = await db.execute(
            select(PlatformTask).where(PlatformTask.id == platform_task_id)
        )
        task = result.scalar_one_or_none()
        if task:
            task.status = "FAILED"
            task.finished_at = _utcnow()
            task.error_message = error[:2000]
            await db.commit()


async def _mark_task_retry(platform_task_id: str, error: str) -> None:
    from app.models.database import PlatformTask, async_session_factory
    from sqlalchemy import select
    async with async_session_factory() as db:
        result = await db.execute(
            select(PlatformTask).where(PlatformTask.id == platform_task_id)
        )
        task = result.scalar_one_or_none()
        if task:
            task.status = "RETRY"
            task.retry_count = (task.retry_count or 0) + 1
            task.error_message = error[:2000]
            await db.commit()


# ---------------------------------------------------------------------------
# Task: run_train_task
# Wraps the existing training_service logic via PlatformTask.payload_ref
# ---------------------------------------------------------------------------

@celery_app.task(
    bind=True,
    base=MLBaseTask,
    name="app.scheduler.celery_tasks.run_train_task",
    max_retries=3,
    default_retry_delay=30,
)
def run_train_task(self, platform_task_id: str) -> dict:
    """Execute a training run referenced by a PlatformTask record."""
    try:
        _run_async(_mark_task_running(platform_task_id, self.request.id))
        result = _run_async(_execute_train(platform_task_id))
        _run_async(_mark_task_success(platform_task_id, result.get("metrics")))
        return result
    except SoftTimeLimitExceeded:
        _run_async(_mark_task_failed(platform_task_id, "Task exceeded time limit"))
        raise
    except Exception as exc:
        _run_async(_mark_task_retry(platform_task_id, str(exc)))
        raise self.retry(exc=exc)


async def _execute_train(platform_task_id: str) -> dict:
    """Resolve payload_ref and call the appropriate service."""
    from app.models.database import PlatformTask, async_session_factory
    from sqlalchemy import select
    async with async_session_factory() as db:
        result = await db.execute(
            select(PlatformTask).where(PlatformTask.id == platform_task_id)
        )
        ptask = result.scalar_one_or_none()
        if ptask is None:
            raise ValueError(f"PlatformTask {platform_task_id} not found")

        payload_ref = ptask.payload_ref or ""

    # payload_ref format: "train:<training_task_id>"
    kind, _, ref_id = payload_ref.partition(":")
    if kind == "train" and ref_id:
        from app.services.training_service import _run_training_sync_by_id
        return await _run_training_sync_by_id(ref_id, platform_task_id)
    elif kind == "dl_train" and ref_id:
        from app.services.dl_service import _run_dl_training_by_id
        return await _run_dl_training_by_id(ref_id, platform_task_id)
    else:
        raise ValueError(f"Unknown payload_ref: {payload_ref!r}")


# ---------------------------------------------------------------------------
# Task: run_explain_task
# Runs SHAP explanation for a completed ExperimentRun
# ---------------------------------------------------------------------------

@celery_app.task(
    bind=True,
    base=MLBaseTask,
    name="app.scheduler.celery_tasks.run_explain_task",
    max_retries=2,
    default_retry_delay=60,
)
def run_explain_task(self, platform_task_id: str) -> dict:
    """Run SHAP explanation for a model, referenced via PlatformTask.payload_ref."""
    try:
        _run_async(_mark_task_running(platform_task_id, self.request.id))
        result = _run_async(_execute_explain(platform_task_id))
        _run_async(_mark_task_success(platform_task_id, result.get("metrics")))
        return result
    except SoftTimeLimitExceeded:
        _run_async(_mark_task_failed(platform_task_id, "Explain task exceeded time limit"))
        raise
    except Exception as exc:
        _run_async(_mark_task_retry(platform_task_id, str(exc)))
        raise self.retry(exc=exc)


async def _execute_explain(platform_task_id: str) -> dict:
    from app.models.database import PlatformTask, async_session_factory
    from sqlalchemy import select
    async with async_session_factory() as db:
        result = await db.execute(
            select(PlatformTask).where(PlatformTask.id == platform_task_id)
        )
        ptask = result.scalar_one_or_none()
        if ptask is None:
            raise ValueError(f"PlatformTask {platform_task_id} not found")
        payload_ref = ptask.payload_ref or ""

    # payload_ref: "explain:<run_id>"
    _, _, run_id = payload_ref.partition(":")
    if not run_id:
        raise ValueError(f"Invalid explain payload_ref: {payload_ref!r}")

    from app.services.explain_service import run_shap_explanation
    return await run_shap_explanation(run_id, platform_task_id)


# ---------------------------------------------------------------------------
# Task: run_platform_task_generic (Phase 5 — CeleryScheduler entry point)
#
# Instead of per-kind Celery tasks, the V3 scheduler routes every submission
# through this one task: it looks up the PlatformTask, then delegates to the
# executor registry (``app.scheduler.executors``).  That way a new ``kind``
# only needs to ``register_executor(...)`` at service-import time — no Celery
# boilerplate.  Status writeback is handled inside ``run_with_status`` so the
# Celery retry envelope only needs to swallow/re-raise.
# ---------------------------------------------------------------------------

@celery_app.task(
    bind=True,
    base=MLBaseTask,
    name="app.scheduler.celery_tasks.run_platform_task_generic",
    max_retries=3,
    default_retry_delay=30,
    retry_backoff=True,
)
def run_platform_task_generic(self, platform_task_id: str) -> dict:
    """Generic PlatformTask runner — dispatches via the executor registry."""
    try:
        # Record the Celery task id so the UI's "worker/celery task id" column
        # has something to show (status transitions beyond RUNNING are owned
        # by run_with_status).
        _run_async(_mark_celery_id(platform_task_id, self.request.id))
        result = _run_async(_execute_generic(platform_task_id))
        return result
    except SoftTimeLimitExceeded:
        _run_async(_mark_task_failed(platform_task_id, "Task exceeded time limit"))
        raise
    except Exception as exc:
        # run_with_status already wrote FAILED; bubble up so Celery retries.
        countdown = 30 * (2 ** int(getattr(self.request, "retries", 0)))
        raise self.retry(exc=exc, countdown=countdown)


async def _mark_celery_id(platform_task_id: str, celery_id: str) -> None:
    """Only stamp the celery_task_id — status/timestamps are owned elsewhere."""
    from app.models.database import PlatformTask, async_session_factory
    from sqlalchemy import select
    async with async_session_factory() as db:
        result = await db.execute(
            select(PlatformTask).where(PlatformTask.id == platform_task_id)
        )
        task = result.scalar_one_or_none()
        if task:
            task.celery_task_id = celery_id
            await db.commit()


async def _execute_generic(platform_task_id: str) -> dict:
    """Resolve kind + payload_ref and invoke the executor via run_with_status."""
    from app.models.database import PlatformTask, async_session_factory
    from sqlalchemy import select
    from app.scheduler.executors import has_executor, load_executor_modules, run_with_status

    # Force-import services so their register_executor calls have fired —
    # the Celery worker process may not have run app.main lifespan.
    load_executor_modules()

    async with async_session_factory() as db:
        result = await db.execute(
            select(PlatformTask).where(PlatformTask.id == platform_task_id)
        )
        ptask = result.scalar_one_or_none()
        if ptask is None:
            raise ValueError(f"PlatformTask {platform_task_id} not found")
        kind = ptask.kind or ""
        payload_ref = ptask.payload_ref or ""

    _, _, domain_id = payload_ref.partition(":")
    if not kind or not domain_id:
        raise ValueError(
            f"Invalid PlatformTask {platform_task_id}: kind={kind!r}, "
            f"payload_ref={payload_ref!r}"
        )

    if not has_executor(kind):
        # Fall back to legacy per-kind dispatch so services that haven't
        # migrated (e.g. ts_forecast in edge cases) still work under Celery.
        from app.scheduler.task_runner import dispatch_platform_task
        await dispatch_platform_task(
            platform_task_id=platform_task_id,
            kind=kind,
            payload_ref=payload_ref,
            priority=5,
        )
        return {"metrics": {}}

    # run_with_status owns RUNNING → SUCCESS/FAILED transitions.
    result = await run_with_status(kind, domain_id, platform_task_id)
    from app.scheduler.scheduler import CeleryScheduler

    await CeleryScheduler().on_task_done(platform_task_id)
    return result

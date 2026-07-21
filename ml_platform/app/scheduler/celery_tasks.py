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


async def _complete(platform_task_id: str, status: str, *, error: str | None = None):
    """Terminalise via the single M2c write-back entry (safe to call twice)."""
    from app.services.run_writeback import complete_platform_task

    return await complete_platform_task(
        platform_task_id, status=status, error=error, final_attempt=True
    )


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
        """Last-resort terminaliser once Celery gives up.

        Routed through the unified completion entry (idempotent) so the
        ExperimentRun, batch finalisation and DAG propagation all happen
        exactly once — the old path wrote only the PlatformTask and then
        propagated the DAG a second time.
        """
        from app.services.run_writeback import WritebackError

        platform_task_id = kwargs.get("platform_task_id") or (args[0] if args else None)
        if isinstance(exc, WritebackError):
            # The work itself succeeded; only recording it failed. Writing
            # FAILED here would erase a real result. The row stays non-terminal
            # so ``scheduler.recover_stalled_tasks`` can re-arm and re-execute
            # it — the result was never persisted, so re-running is the only
            # honest recovery. Note that only the *terminal write-back* is
            # idempotent: the executor is not attempt-fenced, so the re-run
            # rewrites the same artifacts. See recover_stalled_tasks for the
            # full statement of what that costs.
            logger.error(
                "Task %s: work completed but write-back never landed (%s). "
                "Left non-terminal for recover_stalled_tasks to re-drive.",
                task_id,
                exc,
            )
            return
        if platform_task_id:
            _run_async(_complete(platform_task_id, "FAILED", error=str(exc)))
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


async def _mark_task_retry(platform_task_id: str, error: str) -> None:
    """Record a retry attempt without ever clobbering a terminal state.

    Conditional UPDATE rather than read-then-write: under concurrency a late
    on_retry could otherwise observe RUNNING, lose the race to a delivery that
    commits SUCCESS, and then overwrite it with RETRY.
    """
    from app.models.database import PlatformTask, async_session_factory
    from app.services.run_writeback import _TERMINAL_STATES
    from sqlalchemy import update
    async with async_session_factory() as db:
        await db.execute(
            update(PlatformTask)
            .where(
                PlatformTask.id == platform_task_id,
                PlatformTask.status.notin_(list(_TERMINAL_STATES)),
            )
            .values(
                status="RETRY",
                retry_count=PlatformTask.retry_count + 1,
                error_message=error[:2000],
            )
        )
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
        _run_async(_complete(platform_task_id, "FAILED", error="Task exceeded time limit"))
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
        _run_async(_complete(platform_task_id, "FAILED", error="Explain task exceeded time limit"))
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
        # M2c: only the last attempt may write a terminal FAILED — earlier
        # attempts park the run in RETRY so the status doesn't flap.
        retries = int(getattr(self.request, "retries", 0) or 0)
        max_retries = int(getattr(self, "max_retries", 0) or 0)
        result = _run_async(
            _execute_generic(platform_task_id, final_attempt=retries >= max_retries)
        )
        return result
    except SoftTimeLimitExceeded:
        # A hard timeout is terminal — route it through the unified entry so the
        # ExperimentRun reaches a terminal state too (not just the PlatformTask).
        _run_async(
            _complete(platform_task_id, "FAILED", error="Task exceeded time limit")
        )
        raise
    except Exception as exc:
        # _execute_generic already recorded RETRY/FAILED via the unified entry.
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


async def _execute_generic(platform_task_id: str, final_attempt: bool = True) -> dict:
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

    # M2c: a PlatformTask that backs a V3 ExperimentRun needs the full
    # write-back tail (normalise metrics → Run → mirror logs → batch
    # finalisation → DAG), not just the PlatformTask status that
    # run_with_status writes. Both are funnelled through
    # run_writeback.complete_platform_task so worker-side completion is
    # byte-identical to the in-process path.
    from app.scheduler.executors import get_executor
    from app.services.run_writeback import (
        WritebackError,
        claim_for_execution,
        complete_platform_task,
    )

    # Claim first (writes RUNNING atomically on Task+Run, refuses terminal).
    # Re-executing an already-finished trial would burn a worker slot and could
    # clobber a committed result, so a lost claim exits immediately.
    if not await claim_for_execution(platform_task_id):
        logger.info(
            "PlatformTask %s already terminal; skipping duplicate execution",
            platform_task_id,
        )
        return {"metrics": {}, "skipped": True}

    try:
        result = await get_executor(kind)(domain_id, platform_task_id)
        if not isinstance(result, dict):
            raise TypeError(
                f"executor for kind={kind!r} returned {type(result).__name__}, expected dict"
            )
        metrics = result.get("metrics") or {}
        evaluation_mode = result.get("evaluation_mode", "standard")
    except Exception as exc:  # noqa: BLE001 — genuine execution failure
        # Terminal-vs-retryable is decided by the caller (the Celery task body
        # knows the remaining retry budget) and passed via _FINAL_ATTEMPT.
        await complete_platform_task(
            platform_task_id,
            status="FAILED",
            error=str(exc),
            domain_task_id=domain_id,
            final_attempt=final_attempt,
        )
        raise

    # Execution succeeded. A failure from here is a *bookkeeping* failure: it is
    # tagged so Celery's failure hook won't convert a good run into FAILED.
    try:
        await complete_platform_task(
            platform_task_id,
            status="SUCCESS",
            metrics=metrics,
            evaluation_mode=evaluation_mode,
            domain_task_id=domain_id,
            final_attempt=True,
        )
    except Exception as exc:  # noqa: BLE001
        raise WritebackError(
            f"executor for {platform_task_id} succeeded but write-back failed: {exc}"
        ) from exc
    return result

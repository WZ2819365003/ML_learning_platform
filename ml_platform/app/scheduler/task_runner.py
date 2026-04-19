"""
Task Runner — high-level API for submitting, retrying and cancelling PlatformTasks.

Usage from any service:
    from app.scheduler.task_runner import submit_task

    platform_task_id = await submit_task(
        db=db,
        kind="train",
        payload_ref="train:<training_task_id>",
        priority=5,
    )
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.models.database import PlatformTask

logger = logging.getLogger(__name__)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _split_payload_ref(payload_ref: str | None) -> tuple[str | None, str | None]:
    if not payload_ref:
        return None, None
    payload_kind, _, domain_id = payload_ref.partition(":")
    return payload_kind or None, domain_id or None


# Map task kind → Celery task name
_KIND_TO_CELERY: dict[str, str] = {
    "train":    "app.scheduler.celery_tasks.run_train_task",
    "dl_train": "app.scheduler.celery_tasks.run_train_task",
    "explain":  "app.scheduler.celery_tasks.run_explain_task",
}


async def register_domain_task(
    db: AsyncSession,
    kind: str,
    payload_ref: str,
    priority: int = 5,
) -> "PlatformTask":
    """
    Register an asyncio-executed domain task as a PlatformTask (no Celery dispatch).

    Use this when the actual work is driven by asyncio.create_task() inside a
    service (training_service, dl_service, timeseries_service).  The returned
    PlatformTask is flushed but NOT committed — caller must commit.
    """
    task = PlatformTask(
        kind=kind,
        status="QUEUED",
        priority=priority,
        payload_ref=payload_ref,
        queued_at=_utcnow(),
    )
    db.add(task)
    await db.flush()
    await db.refresh(task)
    return task


async def update_platform_task_status(
    platform_task_id: str,
    status: str,
    metrics: dict | None = None,
    error: str | None = None,
    progress: float | None = None,
) -> None:
    """
    Update a PlatformTask's status from an asyncio coroutine (write-back contract).

    Opens its own session so it can be called from inside _execute_* coroutines
    that own a separate session scope.
    """
    from sqlalchemy import select
    from app.models.database import async_session_factory

    try:
        async with async_session_factory() as db:
            result = await db.execute(
                select(PlatformTask).where(PlatformTask.id == platform_task_id)
            )
            task = result.scalar_one_or_none()
            if task is None:
                logger.warning("update_platform_task_status: id=%r not found", platform_task_id)
                return
            task.status = status
            if status == "RUNNING" and task.started_at is None:
                task.started_at = _utcnow()
            if status in ("SUCCESS", "FAILED", "CANCELLED"):
                task.finished_at = _utcnow()
            if status == "SUCCESS":
                task.progress = 1.0
            if progress is not None:
                task.progress = progress
            if metrics:
                task.metrics_snapshot = metrics
            if error:
                task.error_message = str(error)[:2000]
            await db.commit()
    except Exception as exc:
        logger.error("Failed to update PlatformTask %r → %s: %s", platform_task_id, status, exc)


async def submit_task(
    db: AsyncSession,
    kind: str,
    payload_ref: str,
    priority: int = 5,
    max_retries: int = 3,
    extra: dict[str, Any] | None = None,
) -> PlatformTask:
    """
    Create a PlatformTask record and dispatch it to Celery.

    Returns the newly created PlatformTask (already flushed, not committed —
    caller must commit via the FastAPI get_db() dependency).
    """
    task = PlatformTask(
        kind=kind,
        status="QUEUED",
        priority=priority,
        payload_ref=payload_ref,
        max_retries=max_retries,
        queued_at=_utcnow(),
        metrics_snapshot=extra,
    )
    db.add(task)
    await db.flush()
    await db.refresh(task)

    celery_task_name = _KIND_TO_CELERY.get(kind)
    if celery_task_name:
        _dispatch_to_celery(celery_task_name, task.id, priority)
    else:
        logger.warning("No Celery task registered for kind=%r; task %s stays QUEUED", kind, task.id)

    return task


def _dispatch_to_celery(task_name: str, platform_task_id: str, priority: int) -> None:
    """Fire-and-forget Celery dispatch. Errors are logged but not re-raised."""
    try:
        from app.scheduler.celery_app import celery_app
        async_result = celery_app.send_task(
            task_name,
            args=[platform_task_id],
            priority=max(0, min(9, 10 - priority)),  # Celery: 9=highest, 0=lowest
        )
        logger.info("Dispatched %s → Celery task %s (priority=%d)", platform_task_id, task_name, priority)
        return async_result.id
    except Exception as exc:
        logger.error("Failed to dispatch task %s to Celery: %s", platform_task_id, exc)
        # Task stays QUEUED — will be picked up on next worker reconnect
        return None


async def _run_train_task_inprocess(domain_task_id: str, platform_task_id: str) -> None:
    from app.services.training_service import _run_training_sync_by_id

    try:
        await update_platform_task_status(platform_task_id, "RUNNING", progress=0.0)
        result = await _run_training_sync_by_id(domain_task_id, platform_task_id)
        await update_platform_task_status(
            platform_task_id,
            "SUCCESS",
            metrics=result.get("metrics", {}),
        )
    except Exception as exc:  # noqa: BLE001
        await update_platform_task_status(platform_task_id, "FAILED", error=str(exc))


async def _run_dl_task_inprocess(domain_task_id: str, platform_task_id: str) -> None:
    from app.services.dl_service import _run_dl_training_by_id

    try:
        await update_platform_task_status(platform_task_id, "RUNNING", progress=0.0)
        result = await _run_dl_training_by_id(domain_task_id, platform_task_id)
        await update_platform_task_status(
            platform_task_id,
            "SUCCESS",
            metrics=result.get("metrics", {}),
        )
    except Exception as exc:  # noqa: BLE001
        await update_platform_task_status(platform_task_id, "FAILED", error=str(exc))


async def _run_explain_task_inprocess(run_id: str, platform_task_id: str) -> None:
    from app.services.explain_service import run_shap_explanation

    try:
        await update_platform_task_status(platform_task_id, "RUNNING", progress=0.0)
        result = await run_shap_explanation(run_id, platform_task_id)
        await update_platform_task_status(
            platform_task_id,
            "SUCCESS",
            metrics={"feature_count": len(result.get("feature_importances", {}))},
        )
    except Exception as exc:  # noqa: BLE001
        await update_platform_task_status(platform_task_id, "FAILED", error=str(exc))


async def dispatch_platform_task(
    platform_task_id: str,
    kind: str,
    payload_ref: str | None,
    priority: int,
) -> None:
    """Launch a queued PlatformTask using the current in-process execution model."""
    payload_kind, domain_id = _split_payload_ref(payload_ref)
    if not payload_kind or not domain_id:
        raise ValueError(f"Task {platform_task_id!r} has invalid payload_ref {payload_ref!r}")

    if kind == "train" and payload_kind == "train":
        from sqlalchemy import select
        from app.models.database import ExperimentRun, async_session_factory

        async with async_session_factory() as db:
            result = await db.execute(
                select(ExperimentRun).where(ExperimentRun.task_id == platform_task_id)
            )
            run = result.scalar_one_or_none()

        if run is not None:
            from app.services.automl_service import _run_automl_candidate

            asyncio.create_task(
                _run_automl_candidate(
                    domain_task_id=domain_id,
                    platform_task_id=platform_task_id,
                    run_id=run.id,
                    experiment_id=run.experiment_id,
                )
            )
            return

        asyncio.create_task(_run_train_task_inprocess(domain_id, platform_task_id))
        return

    if kind == "dl_train" and payload_kind == "dl_train":
        asyncio.create_task(_run_dl_task_inprocess(domain_id, platform_task_id))
        return

    if kind == "predict" and payload_kind == "ts_forecast":
        from app.services.timeseries_service import _run_forecast_bg

        asyncio.create_task(_run_forecast_bg(domain_id, platform_task_id=platform_task_id))
        return

    if kind == "explain" and payload_kind == "explain":
        asyncio.create_task(_run_explain_task_inprocess(domain_id, platform_task_id))
        return

    celery_task_name = _KIND_TO_CELERY.get(kind)
    if celery_task_name:
        _dispatch_to_celery(celery_task_name, platform_task_id, priority)
        return

    raise ValueError(f"No dispatcher available for task kind={kind!r}, payload_ref={payload_ref!r}")


async def retry_task(db: AsyncSession, platform_task_id: str) -> PlatformTask:
    """Re-queue a FAILED task. Raises ValueError if task not found or not failed."""
    from sqlalchemy import select
    result = await db.execute(
        select(PlatformTask).where(PlatformTask.id == platform_task_id)
    )
    task = result.scalar_one_or_none()
    if task is None:
        raise ValueError(f"PlatformTask {platform_task_id!r} not found")
    if task.status not in ("FAILED", "RETRY"):
        raise ValueError(f"Task {platform_task_id!r} is {task.status!r}, cannot retry")
    if task.retry_count >= task.max_retries:
        raise ValueError(f"Task {platform_task_id!r} exhausted retries ({task.retry_count}/{task.max_retries})")

    task.status = "QUEUED"
    task.queued_at = _utcnow()
    task.started_at = None
    task.finished_at = None
    task.progress = 0.0
    task.metrics_snapshot = None
    task.error_message = None
    task.celery_task_id = None
    task.retry_count = (task.retry_count or 0) + 1
    await db.flush()
    return task


async def cancel_task(db: AsyncSession, platform_task_id: str) -> PlatformTask:
    """Cancel a PENDING or QUEUED task. Running tasks are marked for soft-cancel."""
    from sqlalchemy import select
    result = await db.execute(
        select(PlatformTask).where(PlatformTask.id == platform_task_id)
    )
    task = result.scalar_one_or_none()
    if task is None:
        raise ValueError(f"PlatformTask {platform_task_id!r} not found")
    if task.status in ("SUCCESS", "FAILED", "CANCELLED"):
        raise ValueError(f"Task {platform_task_id!r} is already {task.status!r}")

    if task.celery_task_id:
        try:
            from app.scheduler.celery_app import celery_app
            celery_app.control.revoke(task.celery_task_id, terminate=False)
        except Exception as exc:
            logger.warning("Could not revoke Celery task %s: %s", task.celery_task_id, exc)

    task.status = "CANCELLED"
    task.finished_at = _utcnow()
    await db.flush()
    return task

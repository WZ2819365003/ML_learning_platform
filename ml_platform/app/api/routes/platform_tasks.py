"""
V3 Unified Platform Task API — /api/platform/tasks

Provides CRUD + status polling + retry/cancel for PlatformTask records.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.database import PlatformTask, get_db
from app.scheduler.task_runner import cancel_task, retry_task

router = APIRouter(prefix="/platform/tasks", tags=["Platform Tasks V3"])


# ---------------------------------------------------------------------------
# Serializer
# ---------------------------------------------------------------------------

def _serialize(task: PlatformTask) -> dict[str, Any]:
    return {
        "id": task.id,
        "kind": task.kind,
        "status": task.status,
        "priority": task.priority,
        "celery_task_id": task.celery_task_id,
        "worker_id": task.worker_id,
        "retry_count": task.retry_count,
        "max_retries": task.max_retries,
        "payload_ref": task.payload_ref,
        "progress": task.progress,
        "logs_uri": task.logs_uri,
        "metrics_uri": task.metrics_uri,
        "artifacts_uri": task.artifacts_uri,
        "metrics_snapshot": task.metrics_snapshot,
        "error_message": task.error_message,
        "queued_at": task.queued_at.isoformat() if task.queued_at else None,
        "started_at": task.started_at.isoformat() if task.started_at else None,
        "finished_at": task.finished_at.isoformat() if task.finished_at else None,
        "created_at": task.created_at.isoformat() if task.created_at else None,
        "updated_at": task.updated_at.isoformat() if task.updated_at else None,
    }


# ---------------------------------------------------------------------------
# List
# ---------------------------------------------------------------------------

@router.get("/", summary="List platform tasks")
async def list_platform_tasks(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=200),
    kind: str | None = Query(None),
    status: str | None = Query(None),
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    stmt = select(PlatformTask)
    count_stmt = select(func.count(PlatformTask.id))

    if kind:
        stmt = stmt.where(PlatformTask.kind == kind)
        count_stmt = count_stmt.where(PlatformTask.kind == kind)
    if status:
        stmt = stmt.where(PlatformTask.status == status.upper())
        count_stmt = count_stmt.where(PlatformTask.status == status.upper())

    total = (await db.execute(count_stmt)).scalar_one()
    rows = await db.execute(
        stmt.order_by(PlatformTask.created_at.desc())
        .offset((page - 1) * page_size)
        .limit(page_size)
    )
    tasks = rows.scalars().all()
    return {
        "items": [_serialize(t) for t in tasks],
        "total": total,
        "page": page,
        "page_size": page_size,
    }


# ---------------------------------------------------------------------------
# Get
# ---------------------------------------------------------------------------

@router.get("/{task_id}", summary="Get a platform task")
async def get_platform_task(
    task_id: str,
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    result = await db.execute(select(PlatformTask).where(PlatformTask.id == task_id))
    task = result.scalar_one_or_none()
    if task is None:
        raise HTTPException(status_code=404, detail=f"PlatformTask {task_id!r} not found")
    return _serialize(task)


# ---------------------------------------------------------------------------
# Retry
# ---------------------------------------------------------------------------

@router.post("/{task_id}/retry", summary="Retry a failed task")
async def retry_platform_task(
    task_id: str,
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    task = await retry_task(db, task_id)
    return _serialize(task)


# ---------------------------------------------------------------------------
# Cancel
# ---------------------------------------------------------------------------

@router.post("/{task_id}/cancel", summary="Cancel a pending/queued task")
async def cancel_platform_task(
    task_id: str,
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    task = await cancel_task(db, task_id)
    return _serialize(task)


# ---------------------------------------------------------------------------
# Delete
# ---------------------------------------------------------------------------

@router.delete("/{task_id}", summary="Delete a finished task record")
async def delete_platform_task(
    task_id: str,
    db: AsyncSession = Depends(get_db),
) -> dict[str, str]:
    result = await db.execute(select(PlatformTask).where(PlatformTask.id == task_id))
    task = result.scalar_one_or_none()
    if task is None:
        raise HTTPException(status_code=404, detail=f"PlatformTask {task_id!r} not found")
    if task.status in ("QUEUED", "RUNNING"):
        raise HTTPException(status_code=400, detail="Cannot delete an active task; cancel it first")
    await db.delete(task)
    return {"message": "deleted"}

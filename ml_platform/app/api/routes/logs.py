"""Log routes -- retrieve, download, and query metrics for training tasks."""

from fastapi import APIRouter, Depends, Query
from fastapi.responses import FileResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.auth import current_username_from_authorization, owner_scope_username
from app.core.ownership import ensure_task_owner
from app.models.database import get_db
from app.models.schemas import LogResponse, MetricsResponse
from app.services.log_service import (
    download_task_logs,
    get_task_logs,
    get_task_metrics,
)

router = APIRouter(prefix="/logs", tags=["Logs"])


async def owned_task_id(
    task_id: str,
    db: AsyncSession = Depends(get_db),
    username: str = Depends(current_username_from_authorization),
) -> str:
    await ensure_task_owner(db, task_id, owner_scope_username(username))
    return task_id


@router.get("/{task_id}", response_model=LogResponse)
async def get_logs(
    task_id: str = Depends(owned_task_id),
    level: str | None = Query(None, description="Filter by log level (INFO, WARN, ERROR)"),
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(100, ge=1, le=500, description="Entries per page"),
):
    """Return paginated log entries for a training task."""
    return await get_task_logs(
        task_id=task_id, level=level, page=page, page_size=page_size
    )


@router.get("/{task_id}/download", response_class=FileResponse)
async def download_logs(
    task_id: str = Depends(owned_task_id),
    format: str = Query("txt", pattern="^(txt|json)$", description="Export format"),
):
    """Download the full log file or metrics JSON for a training task."""
    return await download_task_logs(task_id=task_id, fmt=format)


@router.get("/{task_id}/metrics", response_model=MetricsResponse)
async def get_metrics(task_id: str = Depends(owned_task_id)):
    """Return per-step / per-fold metrics for a training task."""
    return await get_task_metrics(task_id=task_id)

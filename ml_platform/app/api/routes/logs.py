"""Log routes -- retrieve, download, and query metrics for training tasks."""

from fastapi import APIRouter, Query
from fastapi.responses import FileResponse

from app.models.schemas import LogResponse, MetricsResponse
from app.services.log_service import (
    download_task_logs,
    get_task_logs,
    get_task_metrics,
)

router = APIRouter(prefix="/logs", tags=["Logs"])


@router.get("/{task_id}", response_model=LogResponse)
async def get_logs(
    task_id: str,
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
    task_id: str,
    format: str = Query("txt", pattern="^(txt|json)$", description="Export format"),
):
    """Download the full log file or metrics JSON for a training task."""
    return await download_task_logs(task_id=task_id, fmt=format)


@router.get("/{task_id}/metrics", response_model=MetricsResponse)
async def get_metrics(task_id: str):
    """Return per-step / per-fold metrics for a training task."""
    return await get_task_metrics(task_id=task_id)

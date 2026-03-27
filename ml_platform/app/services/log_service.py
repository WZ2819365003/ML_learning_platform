"""Log service — query and export training logs."""

from fastapi import HTTPException
from fastapi.responses import FileResponse

from app.core.logger import TrainingLogger


def _get_logger(task_id: str) -> TrainingLogger:
    """Create a TrainingLogger instance for reading (no model_type needed)."""
    tl = TrainingLogger.__new__(TrainingLogger)
    from app.config import get_settings

    settings = get_settings()
    tl.task_id = task_id
    tl.model_type = ""
    tl.log_dir = settings.storage_logs
    tl.log_file = tl.log_dir / f"{task_id}.log"
    tl.metrics_file = tl.log_dir / f"{task_id}_metrics.json"
    tl._metrics_data = {"task_id": task_id, "model_type": "", "steps": []}
    return tl


async def get_task_logs(
    task_id: str,
    level: str | None = None,
    page: int = 1,
    page_size: int = 100,
) -> dict:
    """Return paginated log entries for a training task."""
    tl = _get_logger(task_id)
    if not tl.log_file.exists():
        raise HTTPException(
            status_code=404, detail=f"No logs found for task {task_id}"
        )
    return tl.get_log_content(
        level_filter=level, page=page, page_size=page_size
    )


async def download_task_logs(
    task_id: str, fmt: str = "txt"
) -> FileResponse:
    """Return a downloadable log or metrics file for a training task."""
    tl = _get_logger(task_id)
    file_path = tl.export(fmt=fmt)
    if not file_path.exists():
        raise HTTPException(
            status_code=404, detail=f"No log file for task {task_id}"
        )

    media_type = "application/json" if fmt == "json" else "text/plain"
    filename = (
        f"{task_id}_metrics.json" if fmt == "json" else f"{task_id}_logs.txt"
    )
    return FileResponse(
        path=str(file_path), media_type=media_type, filename=filename
    )


async def get_task_metrics(task_id: str) -> dict:
    """Return metrics data for a training task."""
    tl = _get_logger(task_id)
    if not tl.metrics_file.exists():
        raise HTTPException(
            status_code=404, detail=f"No metrics found for task {task_id}"
        )
    return tl.get_metrics()

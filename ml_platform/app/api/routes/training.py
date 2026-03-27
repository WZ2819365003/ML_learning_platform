"""Training routes -- start, stop, status, and list training tasks."""

from typing import Optional

from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.trainer import list_available_models
from app.models.database import get_db
from app.models.schemas import (
    TrainingListResponse,
    TrainingRequest,
    TrainingTaskResponse,
)
from app.services.training_service import (
    get_training_status,
    list_training_tasks,
    start_training,
    stop_training,
)

router = APIRouter(prefix="/training", tags=["Training"])


@router.post("/start", response_model=TrainingTaskResponse, status_code=201)
async def start_training_route(
    request: TrainingRequest,
    db: AsyncSession = Depends(get_db),
):
    """Submit a new training job."""
    task = await start_training(request.model_dump(), db)
    return task


@router.get("/{task_id}/status", response_model=TrainingTaskResponse)
async def get_training_status_route(
    task_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Return the current status of a training task."""
    return await get_training_status(task_id, db)


@router.post("/{task_id}/stop", response_model=TrainingTaskResponse)
async def stop_training_route(
    task_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Request cancellation of a running training task."""
    return await stop_training(task_id, db)


@router.get("/list", response_model=TrainingListResponse)
async def list_training_tasks_route(
    page: int = Query(default=1, ge=1, description="Page number"),
    page_size: int = Query(default=20, ge=1, le=100, description="Items per page"),
    status: Optional[str] = Query(default=None, description="Filter by status (PENDING, RUNNING, SUCCESS, FAILED)"),
    db: AsyncSession = Depends(get_db),
):
    """List all training tasks with optional filtering and pagination."""
    return await list_training_tasks(db, page=page, page_size=page_size, status_filter=status)


@router.get("/models", response_model=list[str])
async def get_available_models():
    """Return a list of available model types for training."""
    return list_available_models()

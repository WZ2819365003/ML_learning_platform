"""Deep-learning training routes — REST + WebSocket."""

from typing import Optional

from fastapi import APIRouter, Depends, Query, WebSocket, WebSocketDisconnect
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.dl_registry import (
    DL_CATEGORY_REGISTRY,
    DL_MODEL_REGISTRY,
    DL_OPTIMIZER_PARAMS,
    DL_TRAIN_PARAMS,
)
from app.core.logger import event_bus
from app.models.database import get_db
from app.models.schemas import (
    DLModelsResponse,
    DLTaskListResponse,
    DLTaskResponse,
    DLTrainingRequest,
)
from app.services.dl_service import (
    get_dl_status,
    list_dl_tasks,
    start_dl_training,
    stop_dl_training,
)

router = APIRouter(prefix="/dl", tags=["Deep Learning"])


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

@router.get("/models", response_model=DLModelsResponse)
async def get_dl_models():
    """Return the full DL model registry (categories, models, optimizer/train params)."""
    return {
        "categories":       DL_CATEGORY_REGISTRY,
        "models":           DL_MODEL_REGISTRY,
        "optimizer_params": DL_OPTIMIZER_PARAMS,
        "train_params":     DL_TRAIN_PARAMS,
    }


# ---------------------------------------------------------------------------
# Task management
# ---------------------------------------------------------------------------

@router.post("/train", response_model=DLTaskResponse, status_code=201)
async def start_dl_training_route(
    request: DLTrainingRequest,
    db: AsyncSession = Depends(get_db),
):
    """Submit a new deep-learning training job."""
    task = await start_dl_training(request.model_dump(), db)
    return task


@router.get("/list", response_model=DLTaskListResponse)
async def list_dl_tasks_route(
    page:      int           = Query(default=1, ge=1),
    page_size: int           = Query(default=20, ge=1, le=100),
    status:    Optional[str] = Query(default=None),
    db: AsyncSession = Depends(get_db),
):
    """Return a paginated list of DL training tasks."""
    return await list_dl_tasks(db, page=page, page_size=page_size, status_filter=status)


@router.get("/{task_id}/status", response_model=DLTaskResponse)
async def get_dl_status_route(
    task_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Return the current status and metrics of a DL training task."""
    return await get_dl_status(task_id, db)


@router.post("/{task_id}/stop", response_model=DLTaskResponse)
async def stop_dl_training_route(
    task_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Request cancellation of a running DL training task."""
    return await stop_dl_training(task_id, db)


# ---------------------------------------------------------------------------
# WebSocket — real-time epoch streaming
# ---------------------------------------------------------------------------

@router.websocket("/ws/{task_id}")
async def dl_ws(websocket: WebSocket, task_id: str):
    """Stream per-epoch training metrics for a DL task.

    Message format:
      {"type": "epoch", "epoch": N, "total": M, "train_loss": …, "val_loss": …, …}
      {"type": "done",  "status": "SUCCESS"|"FAILED", "metrics": {…}}
    """
    await websocket.accept()
    queue = event_bus.subscribe(f"dl:{task_id}")
    try:
        while True:
            msg = await queue.get()
            await websocket.send_json(msg)
            if msg.get("type") == "done":
                break
    except WebSocketDisconnect:
        pass
    except Exception:
        pass
    finally:
        event_bus.unsubscribe(f"dl:{task_id}", queue)

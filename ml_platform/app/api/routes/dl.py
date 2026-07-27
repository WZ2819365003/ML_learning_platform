"""Deep-learning training routes — REST + WebSocket."""

from typing import Optional

from fastapi import APIRouter, Depends, Query, WebSocket, WebSocketDisconnect
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.auth import current_username_from_authorization, owner_scope_username
from app.core.dl_registry import (
    DL_CATEGORY_REGISTRY,
    DL_MODEL_REGISTRY,
    DL_OPTIMIZER_PARAMS,
    DL_TRAIN_PARAMS,
)
from app.core.logger import event_bus
from app.models.database import get_db
from app.models.schemas import (
    DLDeploymentListResponse,
    DLDeploymentResponse,
    DLDeployRequest,
    DLEpochListResponse,
    DLModelsResponse,
    DLPredictionRequest,
    DLPredictionResponse,
    DLTaskListResponse,
    DLTaskResponse,
    DLTrainingRequest,
    ModelMetaUpdateRequest,
    TaskRenameRequest,
)
from app.services.dl_service import (
    create_dl_deployment,
    delete_dl_deployment,
    delete_dl_task,
    get_dl_status,
    list_dl_logs,
    list_dl_deployments,
    list_dl_epochs,
    list_dl_tasks,
    list_dl_trained_models,
    predict_dl_deployment,
    predict_dl_task_direct,
    rename_dl_task,
    start_dl_training,
    stop_dl_training,
    toggle_dl_deployment_status,
    update_dl_task_meta,
)
from app.services.log_service import get_task_logs

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
    username: str = Depends(current_username_from_authorization),
):
    """Submit a new deep-learning training job."""
    task = await start_dl_training(
        request.model_dump(),
        db,
        owner_username=owner_scope_username(username),
    )
    return task


@router.get("/list", response_model=DLTaskListResponse)
async def list_dl_tasks_route(
    page:      int           = Query(default=1, ge=1),
    page_size: int           = Query(default=20, ge=1, le=100),
    status:    Optional[str] = Query(default=None),
    db: AsyncSession = Depends(get_db),
    username: str = Depends(current_username_from_authorization),
):
    """Return a paginated list of DL training tasks."""
    return await list_dl_tasks(
        db,
        page=page,
        page_size=page_size,
        status_filter=status,
        owner_username=owner_scope_username(username),
    )


@router.get("/trained-models", response_model=DLTaskListResponse)
async def list_dl_trained_models_route(
    page:      int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
    username: str = Depends(current_username_from_authorization),
):
    """Return paginated list of successfully trained DL models."""
    return await list_dl_trained_models(
        db,
        page=page,
        page_size=page_size,
        owner_username=owner_scope_username(username),
    )


@router.get("/{task_id}/status", response_model=DLTaskResponse)
async def get_dl_status_route(
    task_id: str,
    db: AsyncSession = Depends(get_db),
    username: str = Depends(current_username_from_authorization),
):
    """Return the current status and metrics of a DL training task."""
    return await get_dl_status(
        task_id,
        db,
        owner_username=owner_scope_username(username),
    )


@router.post("/{task_id}/stop", response_model=DLTaskResponse)
async def stop_dl_training_route(
    task_id: str,
    db: AsyncSession = Depends(get_db),
    username: str = Depends(current_username_from_authorization),
):
    """Request cancellation of a running DL training task."""
    return await stop_dl_training(
        task_id,
        db,
        owner_username=owner_scope_username(username),
    )


@router.patch("/{task_id}/name", response_model=DLTaskResponse)
async def rename_dl_task_route(
    task_id: str,
    body: TaskRenameRequest,
    db: AsyncSession = Depends(get_db),
    username: str = Depends(current_username_from_authorization),
):
    """Rename a DL training task."""
    return await rename_dl_task(
        task_id,
        body.name,
        db,
        owner_username=owner_scope_username(username),
    )


@router.patch("/{task_id}/meta", response_model=DLTaskResponse)
async def update_dl_task_meta_route(
    task_id: str,
    body: ModelMetaUpdateRequest,
    db: AsyncSession = Depends(get_db),
    username: str = Depends(current_username_from_authorization),
):
    """Update notes and/or tags of a DL task."""
    return await update_dl_task_meta(
        task_id,
        body.notes,
        body.tags,
        db,
        owner_username=owner_scope_username(username),
    )


@router.get("/{task_id}/logs")
async def get_dl_task_logs_route(
    task_id: str,
    level: Optional[str] = Query(default=None),
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=200, ge=1, le=1000),
    db: AsyncSession = Depends(get_db),
    username: str = Depends(current_username_from_authorization),
):
    """Return paginated training log entries for a DL task."""
    payload = await list_dl_logs(
        task_id,
        db,
        level=level,
        page=page,
        page_size=page_size,
        owner_username=owner_scope_username(username),
    )
    if payload["total"] > 0:
        return payload
    return await get_task_logs(task_id, level=level, page=page, page_size=page_size)


@router.get("/{task_id}/epochs", response_model=DLEpochListResponse)
async def list_dl_epochs_route(
    task_id: str,
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=1000, ge=1, le=2000),
    db: AsyncSession = Depends(get_db),
    username: str = Depends(current_username_from_authorization),
):
    """Return paginated epoch records for a DL task (for page-refresh restore)."""
    return await list_dl_epochs(
        task_id,
        db,
        page=page,
        page_size=page_size,
        owner_username=owner_scope_username(username),
    )


@router.post("/{task_id}/predict")
async def predict_dl_task_route(
    task_id: str,
    body: DLPredictionRequest,
    db: AsyncSession = Depends(get_db),
    username: str = Depends(current_username_from_authorization),
):
    """Direct inference against a trained DL model (no deployment required)."""
    return await predict_dl_task_direct(
        task_id,
        body.rows,
        db,
        owner_username=owner_scope_username(username),
    )


@router.delete("/{task_id}", status_code=204)
async def delete_dl_task_route(
    task_id: str,
    db: AsyncSession = Depends(get_db),
    username: str = Depends(current_username_from_authorization),
):
    """Delete a DL training task (not allowed while RUNNING)."""
    await delete_dl_task(
        task_id,
        db,
        owner_username=owner_scope_username(username),
    )


# ---------------------------------------------------------------------------
# DL Deployments
# ---------------------------------------------------------------------------

@router.post("/deployments/{dl_task_id}", response_model=DLDeploymentResponse, status_code=201)
async def create_dl_deployment_route(
    dl_task_id: str,
    body: DLDeployRequest,
    db: AsyncSession = Depends(get_db),
    username: str = Depends(current_username_from_authorization),
):
    """Deploy a successfully trained DL model."""
    return await create_dl_deployment(
        dl_task_id,
        body.name,
        body.description,
        db,
        owner_username=owner_scope_username(username),
    )


@router.get("/deployments", response_model=DLDeploymentListResponse)
async def list_dl_deployments_route(
    db: AsyncSession = Depends(get_db),
    username: str = Depends(current_username_from_authorization),
):
    """List all DL model deployments."""
    return await list_dl_deployments(
        db,
        owner_username=owner_scope_username(username),
    )


@router.patch("/deployments/{dep_id}/status", response_model=DLDeploymentResponse)
async def toggle_dl_deployment_route(
    dep_id: str,
    status: str = Query(...),
    db: AsyncSession = Depends(get_db),
    username: str = Depends(current_username_from_authorization),
):
    """Pause or resume a DL deployment."""
    return await toggle_dl_deployment_status(
        dep_id,
        status,
        db,
        owner_username=owner_scope_username(username),
    )


@router.delete("/deployments/{dep_id}", status_code=204)
async def delete_dl_deployment_route(
    dep_id: str,
    db: AsyncSession = Depends(get_db),
    username: str = Depends(current_username_from_authorization),
):
    """Delete a DL deployment."""
    await delete_dl_deployment(
        dep_id,
        db,
        owner_username=owner_scope_username(username),
    )


@router.post("/deployments/{dep_id}/predict", response_model=DLPredictionResponse)
async def predict_dl_deployment_route(
    dep_id: str,
    body: DLPredictionRequest,
    db: AsyncSession = Depends(get_db),
    username: str = Depends(current_username_from_authorization),
):
    """Run inference via a DL deployment."""
    return await predict_dl_deployment(
        dep_id,
        body.rows,
        db,
        owner_username=owner_scope_username(username),
    )


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
    from app.api.websocket import ws_task_authorized

    if not await ws_task_authorized(websocket, task_id):
        return
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

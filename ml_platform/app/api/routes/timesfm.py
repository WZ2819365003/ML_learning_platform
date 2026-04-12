from __future__ import annotations

import asyncio
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.database import get_db
from app.models.schemas import ModelMetaUpdateRequest
from app.services.chronos_service import get_status, is_available, preload_model_async
from app.services.timeseries_service import (
    create_legacy_ts_task,
    create_ts_deployment,
    create_ts_task,
    delete_ts_deployment,
    delete_ts_task,
    get_ts_deployment,
    get_ts_task,
    list_ts_deployments,
    list_ts_tasks,
    run_ts_deployment_prediction,
    update_ts_task_meta,
    update_ts_deployment_status,
)

router = APIRouter(prefix="/timesfm", tags=["Time Series Compatibility"])
ts_router = APIRouter(prefix="/ts", tags=["Time Series"])


class ForecastStartRequest(BaseModel):
    dataset_id: str
    value_column: str
    time_column: str | None = None
    horizon: int = Field(default=24, ge=1, le=512)
    frequency: str = Field(default="high", pattern="^(high|medium|low)$")
    model_name: str = Field(default="amazon/chronos-t5-small")


class TimeSeriesTaskCreateRequest(BaseModel):
    dataset_id: str
    deployment_id: str | None = None   # optional — auto-creates default deployment if omitted
    value_column: str
    time_column: str | None = None
    horizon: int = Field(default=24, ge=1, le=512)
    frequency: str = Field(default="high", pattern="^(high|medium|low)$")


class TimeSeriesDeploymentCreateRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    description: str | None = None
    backend_label: str = Field(default="amazon/chronos-t5-small")
    config: dict[str, Any] | None = None


class TimeSeriesDeploymentPredictRequest(BaseModel):
    dataset_id: str
    value_column: str
    time_column: str | None = None
    horizon: int = Field(default=24, ge=1, le=512)
    frequency: str = Field(default="high", pattern="^(high|medium|low)$")


@router.get("/model/status")
async def model_status() -> dict[str, Any]:
    return get_status()


@router.post("/model/preload")
async def preload_model(model_name: str = Query(default="amazon/chronos-t5-small")) -> dict[str, str]:
    if not is_available():
        raise HTTPException(status_code=503, detail="chronos-forecasting not installed")
    asyncio.create_task(preload_model_async(model_name))
    return {"message": f"Preloading '{model_name}' in background"}


@ts_router.get("/model/status")
async def ts_model_status() -> dict[str, Any]:
    return get_status()


@ts_router.post("/model/preload")
async def ts_preload_model(
    model_name: str = Query(default="amazon/chronos-t5-small"),
) -> dict[str, str]:
    if not is_available():
        raise HTTPException(status_code=503, detail="chronos-forecasting not installed")
    asyncio.create_task(preload_model_async(model_name))
    return {"message": f"Preloading '{model_name}' in background"}


@router.post("/start", status_code=201)
async def start_forecast(
    body: ForecastStartRequest,
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    if not is_available():
        raise HTTPException(
            status_code=503,
            detail="chronos-forecasting is not installed. Run: pip install chronos-forecasting torch",
        )
    return await create_legacy_ts_task(
        db=db,
        dataset_id=body.dataset_id,
        value_column=body.value_column,
        time_column=body.time_column,
        horizon=body.horizon,
        frequency=body.frequency,
        model_name=body.model_name,
    )


@router.get("/list")
async def list_forecasts(
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
    status: str | None = Query(default=None, description="Filter by status"),
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    return await list_ts_tasks(db=db, page=page, page_size=page_size, status=status)


@router.get("/{task_id}")
async def get_forecast(task_id: str, db: AsyncSession = Depends(get_db)) -> dict[str, Any]:
    return await get_ts_task(db=db, task_id=task_id)


@router.delete("/{task_id}")
async def delete_forecast(task_id: str, db: AsyncSession = Depends(get_db)) -> dict[str, Any]:
    await delete_ts_task(db=db, task_id=task_id)
    return {"message": "Forecast task deleted", "id": task_id}


@ts_router.post("/tasks", status_code=201)
async def create_task(
    body: TimeSeriesTaskCreateRequest,
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    if not is_available():
        raise HTTPException(
            status_code=503,
            detail="chronos-forecasting is not installed. Run: pip install chronos-forecasting torch",
        )
    if body.deployment_id:
        # Use the specified deployment
        return await create_ts_task(
            db=db,
            dataset_id=body.dataset_id,
            deployment_id=body.deployment_id,
            value_column=body.value_column,
            time_column=body.time_column,
            horizon=body.horizon,
            frequency=body.frequency,
        )
    else:
        # No deployment selected — auto-create / reuse a default deployment
        return await create_legacy_ts_task(
            db=db,
            dataset_id=body.dataset_id,
            value_column=body.value_column,
            time_column=body.time_column,
            horizon=body.horizon,
            frequency=body.frequency,
            model_name="amazon/chronos-t5-small",
        )


@ts_router.get("/tasks")
async def list_tasks(
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
    status: str | None = Query(default=None),
    dataset_id: str | None = Query(default=None),
    deployment_id: str | None = Query(default=None),
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    return await list_ts_tasks(
        db=db,
        page=page,
        page_size=page_size,
        status=status,
        dataset_id=dataset_id,
        deployment_id=deployment_id,
    )


@ts_router.get("/tasks/{task_id}")
async def get_task(task_id: str, db: AsyncSession = Depends(get_db)) -> dict[str, Any]:
    return await get_ts_task(db=db, task_id=task_id)


@ts_router.delete("/tasks/{task_id}")
async def delete_task(task_id: str, db: AsyncSession = Depends(get_db)) -> dict[str, Any]:
    await delete_ts_task(db=db, task_id=task_id)
    return {"message": "Time-series task deleted", "id": task_id}


@ts_router.patch("/tasks/{task_id}/meta")
async def update_task_meta(
    task_id: str,
    body: ModelMetaUpdateRequest,
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    return await update_ts_task_meta(
        db=db,
        task_id=task_id,
        notes=body.notes,
        tags=body.tags,
    )


@ts_router.post("/deployments", status_code=201)
async def create_deployment(
    body: TimeSeriesDeploymentCreateRequest,
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    return await create_ts_deployment(
        db=db,
        name=body.name,
        description=body.description,
        backend_label=body.backend_label,
        config=body.config,
    )


@ts_router.get("/deployments")
async def list_deployments(
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    return await list_ts_deployments(db=db, page=page, page_size=page_size)


@ts_router.get("/deployments/{deployment_id}")
async def get_deployment(
    deployment_id: str,
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    return await get_ts_deployment(db=db, deployment_id=deployment_id)


@ts_router.patch("/deployments/{deployment_id}/status")
async def update_deployment_status(
    deployment_id: str,
    status: str = Query(..., pattern="^(active|paused)$"),
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    return await update_ts_deployment_status(db=db, deployment_id=deployment_id, status=status)


@ts_router.delete("/deployments/{deployment_id}")
async def remove_deployment(
    deployment_id: str,
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    await delete_ts_deployment(db=db, deployment_id=deployment_id)
    return {"message": "Time-series deployment deleted", "id": deployment_id}


@ts_router.post("/deployments/{deployment_id}/predict")
async def predict_with_deployment(
    deployment_id: str,
    body: TimeSeriesDeploymentPredictRequest,
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    if not is_available():
        raise HTTPException(
            status_code=503,
            detail="chronos-forecasting is not installed. Run: pip install chronos-forecasting torch",
        )
    return await run_ts_deployment_prediction(
        db=db,
        deployment_id=deployment_id,
        dataset_id=body.dataset_id,
        value_column=body.value_column,
        time_column=body.time_column,
        horizon=body.horizon,
        frequency=body.frequency,
    )

"""Time-series forecasting routes (powered by Amazon Chronos).

Follows the same async-task pattern as ML/DL training:
  POST /timesfm/start       → create task + launch background job
  GET  /timesfm/list        → paginated task history
  GET  /timesfm/{id}        → task detail + result payload
  DELETE /timesfm/{id}      → delete record
  GET  /timesfm/model/status → Chronos availability / load status
  POST /timesfm/model/preload → trigger model download (background)
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import Response
from pydantic import BaseModel, Field
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.database import Dataset, TimeSeriesForecastTask, async_session_factory, get_db
from app.services.chronos_service import get_status, is_available, preload_model_async, run_forecast_async

router = APIRouter(prefix="/timesfm", tags=["Time Series Forecast"])


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class ForecastStartRequest(BaseModel):
    dataset_id: str
    value_column: str
    time_column: str | None = None
    horizon: int = Field(default=24, ge=1, le=512)
    frequency: str = Field(default="high", pattern="^(high|medium|low)$")
    model_name: str = Field(default="amazon/chronos-t5-small")


# ---------------------------------------------------------------------------
# Background worker
# ---------------------------------------------------------------------------

async def _run_forecast_bg(task_id: str) -> None:
    """Background coroutine — updates DB with result when done."""
    async with async_session_factory() as db:
        result = await db.execute(
            select(TimeSeriesForecastTask).where(TimeSeriesForecastTask.id == task_id)
        )
        task = result.scalar_one_or_none()
        if task is None:
            return

        task.status = "RUNNING"
        task.started_at = datetime.now(timezone.utc)
        await db.commit()

        try:
            # Resolve dataset file path
            ds_result = await db.execute(
                select(Dataset).where(Dataset.id == task.dataset_id)
            )
            dataset = ds_result.scalar_one_or_none()
            if dataset is None:
                raise ValueError(f"Dataset '{task.dataset_id}' not found.")

            forecast_result = await run_forecast_async(
                file_path=dataset.file_path,
                value_column=task.value_column,
                time_column=task.time_column,
                horizon=task.horizon,
                frequency=task.frequency,
                model_name=task.model_name,
            )

            task.status = "SUCCESS"
            task.result = forecast_result
            task.finished_at = datetime.now(timezone.utc)

        except Exception as exc:  # noqa: BLE001
            task.status = "FAILED"
            task.error_message = str(exc)
            task.finished_at = datetime.now(timezone.utc)

        await db.commit()


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _task_to_dict(task: TimeSeriesForecastTask, include_result: bool = False) -> dict[str, Any]:
    d: dict[str, Any] = {
        "id": task.id,
        "dataset_id": task.dataset_id,
        "dataset_name": task.dataset_name,
        "value_column": task.value_column,
        "time_column": task.time_column,
        "horizon": task.horizon,
        "frequency": task.frequency,
        "model_name": task.model_name,
        "status": task.status,
        "error_message": task.error_message,
        "created_at": task.created_at.isoformat() if task.created_at else None,
        "started_at": task.started_at.isoformat() if task.started_at else None,
        "finished_at": task.finished_at.isoformat() if task.finished_at else None,
    }
    if include_result:
        d["result"] = task.result
    return d


# ---------------------------------------------------------------------------
# Routes — literal paths BEFORE parametric paths
# ---------------------------------------------------------------------------

@router.get("/model/status")
async def model_status() -> dict[str, Any]:
    """Return Chronos model availability and load state."""
    return get_status()


@router.post("/model/preload")
async def preload_model(model_name: str = Query(default="amazon/chronos-t5-small")) -> dict:
    """Trigger model download / warm-up in the background thread."""
    if not is_available():
        raise HTTPException(status_code=503, detail="chronos-forecasting not installed")
    asyncio.create_task(preload_model_async(model_name))
    return {"message": f"Preloading '{model_name}' in background…"}


@router.post("/start", status_code=201)
async def start_forecast(
    body: ForecastStartRequest,
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    """Create a forecast task and launch it as a background job."""
    if not is_available():
        raise HTTPException(
            status_code=503,
            detail=(
                "chronos-forecasting is not installed. "
                "Run: pip install chronos-forecasting torch"
            ),
        )

    # Resolve dataset
    ds_result = await db.execute(select(Dataset).where(Dataset.id == body.dataset_id))
    dataset = ds_result.scalar_one_or_none()
    if dataset is None:
        raise HTTPException(status_code=404, detail=f"Dataset '{body.dataset_id}' not found.")

    task = TimeSeriesForecastTask(
        dataset_id=body.dataset_id,
        dataset_name=dataset.name,
        value_column=body.value_column,
        time_column=body.time_column,
        horizon=body.horizon,
        frequency=body.frequency,
        model_name=body.model_name,
        status="PENDING",
    )
    db.add(task)
    await db.flush()
    await db.refresh(task)

    task_id = task.id
    asyncio.create_task(_run_forecast_bg(task_id))

    return _task_to_dict(task)


@router.get("/list")
async def list_forecasts(
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
    status: str | None = Query(default=None, description="Filter by status: PENDING, RUNNING, SUCCESS, FAILED"),
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    """Return a paginated list of forecast tasks (newest first)."""
    base_q = select(TimeSeriesForecastTask)
    count_q = select(func.count(TimeSeriesForecastTask.id))
    if status:
        base_q = base_q.where(TimeSeriesForecastTask.status == status.upper())
        count_q = count_q.where(TimeSeriesForecastTask.status == status.upper())

    total_result = await db.execute(count_q)
    total = total_result.scalar_one()

    offset = (page - 1) * page_size
    rows_result = await db.execute(
        base_q.order_by(TimeSeriesForecastTask.created_at.desc())
        .offset(offset)
        .limit(page_size)
    )
    tasks = rows_result.scalars().all()

    return {
        "items": [_task_to_dict(t) for t in tasks],
        "total": total,
        "page": page,
        "page_size": page_size,
    }


# ---------------------------------------------------------------------------
# Parametric routes — AFTER all literal routes above
# ---------------------------------------------------------------------------

@router.get("/{task_id}")
async def get_forecast(
    task_id: str,
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    """Return full task info including result payload."""
    result = await db.execute(
        select(TimeSeriesForecastTask).where(TimeSeriesForecastTask.id == task_id)
    )
    task = result.scalar_one_or_none()
    if task is None:
        raise HTTPException(status_code=404, detail="Forecast task not found")
    return _task_to_dict(task, include_result=True)


@router.delete("/{task_id}")
async def delete_forecast(
    task_id: str,
    db: AsyncSession = Depends(get_db),
) -> Response:
    """Delete a forecast task record."""
    result = await db.execute(
        select(TimeSeriesForecastTask).where(TimeSeriesForecastTask.id == task_id)
    )
    task = result.scalar_one_or_none()
    if task is None:
        raise HTTPException(status_code=404, detail="Forecast task not found")
    if task.status == "RUNNING":
        raise HTTPException(status_code=400, detail="Cannot delete a running forecast task")
    await db.delete(task)
    await db.flush()
    return Response(status_code=204)

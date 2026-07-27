from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any

from fastapi import HTTPException
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.database import (
    Dataset,
    TimeSeriesDeployment,
    TimeSeriesForecastTask,
    async_session_factory,
)
from app.services.chronos_service import run_forecast_async

_active_forecast_jobs: set[str] = set()


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _build_predict_url(deployment_id: str) -> str:
    return f"http://127.0.0.1:8000/api/ts/deployments/{deployment_id}/predict"


async def _get_dataset_or_404(
    db: AsyncSession,
    dataset_id: str,
    owner_username: str | None = None,
) -> Dataset:
    stmt = select(Dataset).where(Dataset.id == dataset_id)
    if owner_username:
        stmt = stmt.where(Dataset.owner_username == owner_username)
    result = await db.execute(stmt)
    dataset = result.scalar_one_or_none()
    if dataset is None:
        raise HTTPException(status_code=404, detail=f"Dataset '{dataset_id}' not found")
    return dataset


async def _get_deployment_or_404(
    db: AsyncSession,
    deployment_id: str,
    owner_username: str | None = None,
) -> TimeSeriesDeployment:
    stmt = select(TimeSeriesDeployment).where(TimeSeriesDeployment.id == deployment_id)
    if owner_username:
        stmt = stmt.where(TimeSeriesDeployment.owner_username == owner_username)
    result = await db.execute(stmt)
    deployment = result.scalar_one_or_none()
    if deployment is None:
        raise HTTPException(status_code=404, detail="Time-series deployment not found")
    return deployment


def _task_name(dataset_name: str | None, value_column: str, horizon: int) -> str:
    base = dataset_name or "dataset"
    return f"{base} / {value_column} / {horizon}步"


def _serialize_deployment(deployment: TimeSeriesDeployment) -> dict[str, Any]:
    return {
        "deployment_id": deployment.id,
        "name": deployment.name,
        "description": deployment.description,
        "backend_label": deployment.backend_label,
        "status": deployment.status,
        "request_count": deployment.request_count,
        "config": deployment.config or {},
        "created_at": deployment.created_at.isoformat() if deployment.created_at else None,
        "updated_at": deployment.updated_at.isoformat() if deployment.updated_at else None,
        "predict_url": _build_predict_url(deployment.id),
    }


def _serialize_task(
    task: TimeSeriesForecastTask,
    deployment: TimeSeriesDeployment | None = None,
    include_result: bool = False,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "id": task.id,
        "name": _task_name(task.dataset_name, task.value_column, task.horizon),
        "dataset_id": task.dataset_id,
        "dataset_name": task.dataset_name,
        "value_column": task.value_column,
        "time_column": task.time_column,
        "horizon": task.horizon,
        "frequency": task.frequency,
        "status": task.status,
        "error_message": task.error_message,
        "created_at": task.created_at.isoformat() if task.created_at else None,
        "started_at": task.started_at.isoformat() if task.started_at else None,
        "finished_at": task.finished_at.isoformat() if task.finished_at else None,
        "deployment_id": deployment.id if deployment else None,
        "deployment_name": deployment.name if deployment else None,
        "backend_label": deployment.backend_label if deployment else None,
        "predict_url": _build_predict_url(deployment.id) if deployment else None,
        "model_name": deployment.backend_label if deployment else task.model_name,
        "legacy_model_name": task.model_name,
        "notes": task.notes,
        "tags": task.tags or [],
    }
    if include_result:
        payload["result"] = task.result
    return payload


async def list_ts_deployments(
    db: AsyncSession,
    page: int = 1,
    page_size: int = 20,
    owner_username: str | None = None,
) -> dict[str, Any]:
    count_stmt = select(func.count(TimeSeriesDeployment.id))
    stmt = select(TimeSeriesDeployment)
    if owner_username:
        count_stmt = count_stmt.where(TimeSeriesDeployment.owner_username == owner_username)
        stmt = stmt.where(TimeSeriesDeployment.owner_username == owner_username)
    count_result = await db.execute(count_stmt)
    total = count_result.scalar_one()

    rows = await db.execute(
        stmt
        .order_by(TimeSeriesDeployment.created_at.desc())
        .offset((page - 1) * page_size)
        .limit(page_size)
    )
    deployments = rows.scalars().all()
    return {
        "items": [_serialize_deployment(item) for item in deployments],
        "total": total,
        "page": page,
        "page_size": page_size,
    }


async def create_ts_deployment(
    db: AsyncSession,
    name: str,
    description: str | None = None,
    backend_label: str = "amazon/chronos-t5-small",
    config: dict[str, Any] | None = None,
    owner_username: str | None = None,
) -> dict[str, Any]:
    deployment = TimeSeriesDeployment(
        owner_username=owner_username,
        name=name,
        description=description,
        backend_label=backend_label,
        status="active",
        request_count=0,
        config=config,
    )
    db.add(deployment)
    await db.flush()
    await db.refresh(deployment)
    return _serialize_deployment(deployment)


async def ensure_default_ts_deployment(
    db: AsyncSession,
    backend_label: str = "amazon/chronos-t5-small",
    owner_username: str | None = None,
) -> TimeSeriesDeployment:
    stmt = select(TimeSeriesDeployment).where(
        TimeSeriesDeployment.name == "TimesFM 默认部署",
        TimeSeriesDeployment.backend_label == backend_label,
    )
    if owner_username:
        stmt = stmt.where(TimeSeriesDeployment.owner_username == owner_username)
    result = await db.execute(stmt.limit(1))
    deployment = result.scalar_one_or_none()
    if deployment is not None:
        return deployment

    deployment = TimeSeriesDeployment(
        owner_username=owner_username,
        name="TimesFM 默认部署",
        description="兼容旧版 /api/timesfm 接口自动创建",
        backend_label=backend_label,
        status="active",
        request_count=0,
    )
    db.add(deployment)
    await db.flush()
    await db.refresh(deployment)
    return deployment


async def get_ts_deployment(
    db: AsyncSession,
    deployment_id: str,
    owner_username: str | None = None,
) -> dict[str, Any]:
    deployment = await _get_deployment_or_404(
        db,
        deployment_id,
        owner_username=owner_username,
    )
    return _serialize_deployment(deployment)


async def update_ts_deployment_status(
    db: AsyncSession,
    deployment_id: str,
    status: str,
    owner_username: str | None = None,
) -> dict[str, Any]:
    deployment = await _get_deployment_or_404(
        db,
        deployment_id,
        owner_username=owner_username,
    )
    deployment.status = status
    await db.flush()
    return _serialize_deployment(deployment)


async def delete_ts_deployment(
    db: AsyncSession,
    deployment_id: str,
    owner_username: str | None = None,
) -> None:
    deployment = await _get_deployment_or_404(
        db,
        deployment_id,
        owner_username=owner_username,
    )
    await db.delete(deployment)
    await db.flush()


async def run_ts_deployment_prediction(
    db: AsyncSession,
    deployment_id: str,
    dataset_id: str,
    value_column: str,
    time_column: str | None,
    horizon: int,
    frequency: str,
    owner_username: str | None = None,
) -> dict[str, Any]:
    deployment = await _get_deployment_or_404(
        db,
        deployment_id,
        owner_username=owner_username,
    )
    if deployment.status != "active":
        raise HTTPException(status_code=400, detail="Deployment is paused")

    dataset = await _get_dataset_or_404(
        db,
        dataset_id,
        owner_username=owner_username,
    )
    result = await run_forecast_async(
        file_path=dataset.file_path,
        value_column=value_column,
        time_column=time_column,
        horizon=horizon,
        frequency=frequency,
        model_name=deployment.backend_label,
    )
    deployment.request_count = (deployment.request_count or 0) + 1
    await db.flush()
    return {
        "deployment_id": deployment.id,
        "deployment_name": deployment.name,
        "predict_url": _build_predict_url(deployment.id),
        "result": result,
    }


async def _resolve_task_deployment(
    db: AsyncSession,
    task: TimeSeriesForecastTask,
) -> TimeSeriesDeployment | None:
    if not task.model_name:
        return None
    result = await db.execute(
        select(TimeSeriesDeployment).where(TimeSeriesDeployment.id == task.model_name)
    )
    return result.scalar_one_or_none()


async def _run_forecast_bg(task_id: str, platform_task_id: str | None = None) -> None:
    """Run a time-series forecast in the background and write-back to PlatformTask."""
    from app.scheduler.task_runner import update_platform_task_status

    async with async_session_factory() as db:
        result = await db.execute(
            select(TimeSeriesForecastTask).where(TimeSeriesForecastTask.id == task_id)
        )
        task = result.scalar_one_or_none()
        if task is None:
            return

        task.status = "RUNNING"
        task.started_at = _utcnow()
        await db.commit()

    if platform_task_id:
        await update_platform_task_status(platform_task_id, "RUNNING")

    async with async_session_factory() as db:
        result = await db.execute(
            select(TimeSeriesForecastTask).where(TimeSeriesForecastTask.id == task_id)
        )
        task = result.scalar_one_or_none()
        if task is None:
            return

        try:
            dataset = await _get_dataset_or_404(db, task.dataset_id)
            deployment = await _resolve_task_deployment(db, task)
            backend_label = deployment.backend_label if deployment else task.model_name

            forecast_result = await run_forecast_async(
                file_path=dataset.file_path,
                value_column=task.value_column,
                time_column=task.time_column,
                horizon=task.horizon,
                frequency=task.frequency,
                model_name=backend_label,
            )

            if deployment is not None:
                deployment.request_count = (deployment.request_count or 0) + 1

            task.result = forecast_result
            task.status = "SUCCESS"
            task.finished_at = _utcnow()
            await db.commit()

            if platform_task_id:
                await update_platform_task_status(
                    platform_task_id, "SUCCESS",
                    metrics={"horizon": task.horizon, "model": backend_label},
                )

        except Exception as exc:  # noqa: BLE001
            task.status = "FAILED"
            task.error_message = str(exc)
            task.finished_at = _utcnow()
            await db.commit()

            if platform_task_id:
                await update_platform_task_status(platform_task_id, "FAILED", error=str(exc))


def _schedule_forecast(task_id: str, platform_task_id: str | None = None) -> None:
    if task_id in _active_forecast_jobs:
        return

    _active_forecast_jobs.add(task_id)

    async def _runner() -> None:
        try:
            await _run_forecast_bg(task_id, platform_task_id=platform_task_id)
        finally:
            _active_forecast_jobs.discard(task_id)

    asyncio.create_task(_runner())


async def create_ts_task(
    db: AsyncSession,
    dataset_id: str,
    deployment_id: str,
    value_column: str,
    time_column: str | None,
    horizon: int,
    frequency: str,
    owner_username: str | None = None,
) -> dict[str, Any]:
    deployment = await _get_deployment_or_404(
        db,
        deployment_id,
        owner_username=owner_username,
    )
    if deployment.status != "active":
        raise HTTPException(status_code=400, detail="Deployment is paused")
    dataset = await _get_dataset_or_404(
        db,
        dataset_id,
        owner_username=owner_username,
    )

    task = TimeSeriesForecastTask(
        owner_username=owner_username or dataset.owner_username,
        dataset_id=dataset.id,
        dataset_name=dataset.name,
        value_column=value_column,
        time_column=time_column,
        horizon=horizon,
        frequency=frequency,
        model_name=deployment.id,
        status="PENDING",
    )
    db.add(task)
    await db.flush()
    await db.refresh(task)

    # ── Write-back contract: register in unified platform task table ──────────
    from app.scheduler.task_runner import register_domain_task
    platform_task = await register_domain_task(
        db=db,
        kind="predict",
        payload_ref=f"ts_forecast:{task.id}",
    )
    await db.commit()

    _schedule_forecast(task.id, platform_task_id=platform_task.id)
    return _serialize_task(task, deployment=deployment)


async def create_legacy_ts_task(
    db: AsyncSession,
    dataset_id: str,
    value_column: str,
    time_column: str | None,
    horizon: int,
    frequency: str,
    model_name: str,
    owner_username: str | None = None,
) -> dict[str, Any]:
    deployment = await ensure_default_ts_deployment(
        db,
        backend_label=model_name,
        owner_username=owner_username,
    )
    return await create_ts_task(
        db=db,
        dataset_id=dataset_id,
        deployment_id=deployment.id,
        value_column=value_column,
        time_column=time_column,
        horizon=horizon,
        frequency=frequency,
        owner_username=owner_username,
    )


async def list_ts_tasks(
    db: AsyncSession,
    page: int = 1,
    page_size: int = 20,
    status: str | None = None,
    dataset_id: str | None = None,
    deployment_id: str | None = None,
    owner_username: str | None = None,
) -> dict[str, Any]:
    stmt = select(TimeSeriesForecastTask)
    count_stmt = select(func.count(TimeSeriesForecastTask.id))

    if status:
        stmt = stmt.where(TimeSeriesForecastTask.status == status.upper())
        count_stmt = count_stmt.where(TimeSeriesForecastTask.status == status.upper())
    if dataset_id:
        stmt = stmt.where(TimeSeriesForecastTask.dataset_id == dataset_id)
        count_stmt = count_stmt.where(TimeSeriesForecastTask.dataset_id == dataset_id)
    if deployment_id:
        stmt = stmt.where(TimeSeriesForecastTask.model_name == deployment_id)
        count_stmt = count_stmt.where(TimeSeriesForecastTask.model_name == deployment_id)
    if owner_username:
        stmt = stmt.where(TimeSeriesForecastTask.owner_username == owner_username)
        count_stmt = count_stmt.where(TimeSeriesForecastTask.owner_username == owner_username)

    total = (await db.execute(count_stmt)).scalar_one()

    # Late-row-lookup: sort on lightweight id+created_at first (index-only),
    # then fetch full rows by PK to avoid loading large JSON result columns
    # into MySQL's sort buffer.
    id_filters = stmt.whereclause  # reuse existing WHERE conditions
    id_query = (
        select(TimeSeriesForecastTask.id, TimeSeriesForecastTask.created_at)
        .where(id_filters if id_filters is not None else True)
        .order_by(TimeSeriesForecastTask.created_at.desc())
        .offset((page - 1) * page_size)
        .limit(page_size)
    )
    id_rows = await db.execute(id_query)
    page_ids = [row[0] for row in id_rows.fetchall()]

    if not page_ids:
        tasks = []
    else:
        full_rows = await db.execute(
            select(TimeSeriesForecastTask).where(TimeSeriesForecastTask.id.in_(page_ids))
        )
        # Preserve the original DESC order
        id_order = {pid: idx for idx, pid in enumerate(page_ids)}
        tasks = sorted(full_rows.scalars().all(), key=lambda t: id_order.get(t.id, 0))

    deployment_ids = {task.model_name for task in tasks if task.model_name}
    deployment_map: dict[str, TimeSeriesDeployment] = {}
    if deployment_ids:
        deployment_rows = await db.execute(
            select(TimeSeriesDeployment).where(TimeSeriesDeployment.id.in_(deployment_ids))
        )
        deployment_map = {item.id: item for item in deployment_rows.scalars().all()}

    return {
        "items": [
            _serialize_task(task, deployment=deployment_map.get(task.model_name))
            for task in tasks
        ],
        "total": total,
        "page": page,
        "page_size": page_size,
    }


async def get_ts_task(
    db: AsyncSession,
    task_id: str,
    owner_username: str | None = None,
) -> dict[str, Any]:
    stmt = select(TimeSeriesForecastTask).where(TimeSeriesForecastTask.id == task_id)
    if owner_username:
        stmt = stmt.where(TimeSeriesForecastTask.owner_username == owner_username)
    result = await db.execute(stmt)
    task = result.scalar_one_or_none()
    if task is None:
        raise HTTPException(status_code=404, detail="Forecast task not found")
    deployment = await _resolve_task_deployment(db, task)
    return _serialize_task(task, deployment=deployment, include_result=True)


async def delete_ts_task(
    db: AsyncSession,
    task_id: str,
    owner_username: str | None = None,
) -> None:
    stmt = select(TimeSeriesForecastTask).where(TimeSeriesForecastTask.id == task_id)
    if owner_username:
        stmt = stmt.where(TimeSeriesForecastTask.owner_username == owner_username)
    result = await db.execute(stmt)
    task = result.scalar_one_or_none()
    if task is None:
        raise HTTPException(status_code=404, detail="Forecast task not found")
    if task.status == "RUNNING":
        raise HTTPException(status_code=400, detail="Cannot delete a running forecast task")
    await db.delete(task)
    await db.flush()


async def update_ts_task_meta(
    db: AsyncSession,
    task_id: str,
    notes: str | None,
    tags: list[str] | None,
    owner_username: str | None = None,
) -> dict[str, Any]:
    stmt = select(TimeSeriesForecastTask).where(TimeSeriesForecastTask.id == task_id)
    if owner_username:
        stmt = stmt.where(TimeSeriesForecastTask.owner_username == owner_username)
    result = await db.execute(stmt)
    task = result.scalar_one_or_none()
    if task is None:
        raise HTTPException(status_code=404, detail="Forecast task not found")

    if notes is not None:
        task.notes = notes
    if tags is not None:
        task.tags = [tag for tag in tags if tag]

    await db.flush()
    deployment = await _resolve_task_deployment(db, task)
    return _serialize_task(task, deployment=deployment, include_result=True)


async def resume_unfinished_ts_tasks() -> list[str]:
    from app.models.database import PlatformTask

    async with async_session_factory() as db:
        result = await db.execute(
            select(TimeSeriesForecastTask.id).where(
                TimeSeriesForecastTask.finished_at.is_(None),
                TimeSeriesForecastTask.status.in_(["PENDING", "RUNNING"]),
            )
        )
        task_ids = result.scalars().all()

    async with async_session_factory() as db:
        for task_id in task_ids:
            platform_result = await db.execute(
                select(PlatformTask.id).where(
                    PlatformTask.kind == "predict",
                    PlatformTask.payload_ref == f"ts_forecast:{task_id}",
                )
            )
            platform_task_id = platform_result.scalar_one_or_none()
            _schedule_forecast(task_id, platform_task_id=platform_task_id)

    return task_ids

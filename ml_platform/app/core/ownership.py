"""Ownership checks for read-side endpoints that accept mixed task identifiers."""

from __future__ import annotations

from fastapi import HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.database import (
    DLTrainingTask,
    ExperimentRun,
    ModelingTask,
    PlatformExperiment,
    PlatformTask,
    TimeSeriesForecastTask,
    TrainingTask,
)


async def ensure_task_owner(
    db: AsyncSession,
    task_id: str,
    owner_username: str | None,
) -> None:
    """Validate owner for legacy task ids, V3 run ids, and PlatformTask ids."""
    if not owner_username:
        return

    if await _owned_domain_task_exists(db, task_id, owner_username):
        return
    if await _owned_experiment_run_exists(db, task_id, owner_username):
        return

    platform_task = (
        await db.execute(select(PlatformTask).where(PlatformTask.id == task_id))
    ).scalar_one_or_none()
    if platform_task is not None:
        if await _owned_experiment_run_exists(db, platform_task.id, owner_username, by_platform_task=True):
            return
        payload_ref = platform_task.payload_ref or ""
        if ":" in payload_ref:
            kind, _, domain_id = payload_ref.partition(":")
            if kind in {"train", "dl_train", "ts_forecast"} and domain_id:
                if await _owned_domain_task_exists(db, domain_id, owner_username):
                    return

    raise HTTPException(status_code=404, detail="Task not found")


async def _owned_domain_task_exists(
    db: AsyncSession,
    task_id: str,
    owner_username: str,
) -> bool:
    checks = (
        select(TrainingTask.id).where(
            TrainingTask.id == task_id,
            TrainingTask.owner_username == owner_username,
        ),
        select(DLTrainingTask.id).where(
            DLTrainingTask.id == task_id,
            DLTrainingTask.owner_username == owner_username,
        ),
        select(ModelingTask.id).where(
            ModelingTask.id == task_id,
            ModelingTask.owner_username == owner_username,
        ),
        select(TimeSeriesForecastTask.id).where(
            TimeSeriesForecastTask.id == task_id,
            TimeSeriesForecastTask.owner_username == owner_username,
        ),
    )
    for stmt in checks:
        if (await db.execute(stmt)).scalar_one_or_none() is not None:
            return True
    return False


async def _owned_experiment_run_exists(
    db: AsyncSession,
    run_or_platform_task_id: str,
    owner_username: str,
    *,
    by_platform_task: bool = False,
) -> bool:
    stmt = (
        select(ExperimentRun.id)
        .join(PlatformExperiment, PlatformExperiment.id == ExperimentRun.experiment_id)
        .join(ModelingTask, ModelingTask.id == PlatformExperiment.modeling_task_id)
        .where(ModelingTask.owner_username == owner_username)
    )
    if by_platform_task:
        stmt = stmt.where(ExperimentRun.task_id == run_or_platform_task_id)
    else:
        stmt = stmt.where(ExperimentRun.id == run_or_platform_task_id)
    return (await db.execute(stmt.limit(1))).scalar_one_or_none() is not None

"""
V3 Modeling Task API — /api/v3/tasks

Top-level workbench entity: a modeling task owns experiments (baseline /
grid_search / bayesian_search) which in turn own runs.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.database import get_db
from app.services import modeling_task_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v3/tasks", tags=["V3 Modeling Tasks"])


# ---------------------------------------------------------------------------
# Request schemas
# ---------------------------------------------------------------------------

class CreateModelingTaskRequest(BaseModel):
    name: str
    dataset_id: str | None = None
    dataset_version_id: str | None = None
    target_column: str | None = None
    task_type: str = "classification"         # classification | regression
    objective_metric: str = "accuracy"
    objective_direction: str = "max"          # max | min
    description: str | None = None
    config: dict | None = None


class UpdateModelingTaskRequest(BaseModel):
    name: str | None = None
    description: str | None = None
    status: str | None = None
    objective_metric: str | None = None
    objective_direction: str | None = None


class CreateExperimentBatchRequest(BaseModel):
    """
    Create a new experiment (策略批次) under a modeling task.

    - baseline:       single run per `selected_models` entry, no search_space
    - grid_search:    cartesian product over `search_space`; runs per combo
    - bayesian_search: Optuna TPE sampling from `search_space` distributions
    """
    name: str
    strategy_type: str = Field(description="baseline | grid_search | bayesian_search")
    selected_models: list[str] = Field(..., min_length=1)
    search_space: dict | None = None
    budget_config: dict | None = None
    description: str | None = None


# ---------------------------------------------------------------------------
# Tuning spaces (must be registered BEFORE /{task_id} routes)
# ---------------------------------------------------------------------------

@router.get("/tuning-spaces/{task_type}", summary="List tuning templates for a task_type")
async def list_tuning_spaces(task_type: str) -> dict[str, Any]:
    try:
        spaces = modeling_task_service.load_tuning_spaces(task_type)
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {"task_type": task_type, "models": spaces}


# ---------------------------------------------------------------------------
# Task endpoints
# ---------------------------------------------------------------------------

@router.get("/", summary="List modeling tasks")
async def list_modeling_tasks(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    status: str | None = Query(None),
    dataset_id: str | None = Query(None),
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    return await modeling_task_service.list_modeling_tasks(
        db, page=page, page_size=page_size, status=status, dataset_id=dataset_id
    )


@router.post("/", summary="Create a modeling task", status_code=201)
async def create_modeling_task(
    body: CreateModelingTaskRequest,
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    return await modeling_task_service.create_modeling_task(
        db,
        name=body.name,
        dataset_id=body.dataset_id,
        dataset_version_id=body.dataset_version_id,
        target_column=body.target_column,
        task_type=body.task_type,
        objective_metric=body.objective_metric,
        objective_direction=body.objective_direction,
        description=body.description,
        config=body.config,
    )


@router.get("/{task_id}", summary="Get a modeling task with experiments + run stats")
async def get_modeling_task(
    task_id: str,
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    return await modeling_task_service.get_modeling_task(db, task_id)


@router.patch("/{task_id}", summary="Update modeling task")
async def update_modeling_task(
    task_id: str,
    body: UpdateModelingTaskRequest,
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    return await modeling_task_service.update_modeling_task(
        db,
        task_id,
        name=body.name,
        description=body.description,
        status=body.status,
        objective_metric=body.objective_metric,
        objective_direction=body.objective_direction,
    )


@router.delete("/{task_id}", summary="Delete modeling task (cascades experiments)")
async def delete_modeling_task(
    task_id: str,
    db: AsyncSession = Depends(get_db),
) -> dict[str, str]:
    await modeling_task_service.delete_modeling_task(db, task_id)
    return {"message": "deleted"}


# ---------------------------------------------------------------------------
# Aggregated leaderboard across all experiments under this task
# ---------------------------------------------------------------------------

@router.get("/{task_id}/leaderboard", summary="Best runs across all experiments")
async def task_leaderboard(
    task_id: str,
    top_k: int = Query(20, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
) -> list[dict[str, Any]]:
    return await modeling_task_service.task_leaderboard(db, task_id, top_k=top_k)


# ---------------------------------------------------------------------------
# Experiment batch dispatch  (actual tuning engines ship in Commit C)
# ---------------------------------------------------------------------------

@router.post("/{task_id}/experiments", summary="Launch a new experiment batch", status_code=201)
async def create_experiment_batch(
    task_id: str,
    body: CreateExperimentBatchRequest,
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    """
    V3 experiment dispatch.

    Commit B ships the contract + validation; Commit C wires the real
    tuning engines (grid_search / bayesian_search).  Baseline is live today.
    """
    from app.services import tuning_service  # imported lazily; may not exist until Commit C

    strategy = body.strategy_type
    if strategy not in ("baseline", "grid_search", "bayesian_search"):
        raise HTTPException(
            status_code=422,
            detail="strategy_type must be baseline|grid_search|bayesian_search",
        )
    if strategy in ("grid_search", "bayesian_search") and not body.search_space:
        raise HTTPException(
            status_code=422,
            detail=f"{strategy} requires a search_space",
        )

    try:
        return await tuning_service.dispatch_experiment_batch(
            db,
            modeling_task_id=task_id,
            name=body.name,
            strategy_type=strategy,
            selected_models=body.selected_models,
            search_space=body.search_space or {},
            budget_config=body.budget_config or {},
            description=body.description,
        )
    except ImportError as exc:
        # Commit C not yet applied — return 501 rather than crashing.
        logger.warning("tuning_service not available: %s", exc)
        raise HTTPException(
            status_code=501,
            detail="Tuning engines not yet enabled on this server (Commit C pending).",
        ) from exc



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
from app.services.progress_tree_service import get_progress_tree

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
    training_plan_id: str | None = Field(
        default=None,
        description="Optional TrainingPlan to bind. A frozen snapshot is captured at "
                    "create-time so subsequent plan edits don't rewrite history.",
    )


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
    eval_metrics: list[str] | None = None
    description: str | None = None
    model_family: str | None = Field(
        default=None, description="ml | dl | mixed — inferred from selected_models if omitted"
    )
    dl_config: dict | None = Field(
        default=None,
        description="Per-DL-model presets { model_id: { arch, opt, train } }; missing entries backfilled from registry defaults",
    )


class ExperimentStrategyRequest(BaseModel):
    strategy_type: str = Field(description="baseline | grid_search | bayesian_search")
    selected_models: list[str] = Field(..., min_length=1)
    search_space: dict | None = None
    budget_config: dict | None = None
    eval_metrics: list[str] | None = None
    name: str | None = None
    description: str | None = None


class CreateExperimentBundleRequest(BaseModel):
    """Launch several strategy batches under one modeling task."""
    name: str
    strategies: list[ExperimentStrategyRequest] = Field(..., min_length=1)
    description: str | None = None


class DeployRunRequest(BaseModel):
    """Deploy the model trained by a specific run (workflow 部署 step)."""
    name: str = Field(..., min_length=1, max_length=255)
    description: str | None = None
    max_batch_size: int = Field(default=100, ge=1, le=10000)


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
        training_plan_id=body.training_plan_id,
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
# Progress tree — per-model / per-epoch aggregate for ModelingTaskDetail
# ---------------------------------------------------------------------------

@router.get(
    "/{task_id}/progress-tree",
    summary="Per-experiment / per-run progress aggregate (ML + DL in one shape)",
)
async def task_progress_tree(
    task_id: str,
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    """
    Powers the ProgressTree widget on ModelingTaskDetail.

    Shape::

        {
          modeling_task: {id, name, status, progress_aggregated, ...},
          experiments: [{
              id, name, strategy_type, status,
              progress_aggregated, run_count,
              runs: [{id, model_type, family, status, progress, current_step, …}]
          }],
          has_active_runs: bool,   # true → UI should keep polling
        }

    ``family`` is ``"ml" | "dl" | "unknown"`` and lets the frontend pick
    ML vs DL icons.  ``current_step`` is a short human string like
    ``"epoch 12/50"`` (DL) or ``"训练中 (60%)"`` (ML).
    """
    return await get_progress_tree(db, task_id)


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


@router.get(
    "/{task_id}/strategy-comparison",
    summary="Compare baseline / grid_search / bayesian_search for this task",
)
async def task_strategy_comparison(
    task_id: str,
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    """Per-strategy five-number summary + best run + raw points.

    Powers the 策略对比 tab on ModelingTaskDetail: the UI renders three
    cards (baseline / grid / bayesian best value), a box plot from the
    per-strategy stats, and a ranking table driven by raw_points.
    """
    return await modeling_task_service.strategy_comparison(db, task_id)


@router.get("/{task_id}/runs", summary="All runs with scheduler progress")
async def task_runs(
    task_id: str,
    status: str | None = Query(None),
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    return await modeling_task_service.task_runs(db, task_id, status=status)


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
    try:
        return await tuning_service.dispatch_experiment_batch(
            db,
            modeling_task_id=task_id,
            name=body.name,
            strategy_type=strategy,
            selected_models=body.selected_models,
            search_space=body.search_space or {},
            budget_config=body.budget_config or {},
            eval_metrics=body.eval_metrics,
            description=body.description,
            model_family=body.model_family,
            dl_config=body.dl_config,
        )
    except ImportError as exc:
        # Usually a transient optional dependency (optuna / sklearn) missing —
        # surface the real cause so operators can fix their install instead of
        # a generic 501 that hides the traceback.
        logger.exception("tuning_service import failed during dispatch")
        raise HTTPException(
            status_code=501,
            detail=f"Tuning engine unavailable: {exc}",
        ) from exc


@router.post("/{task_id}/experiments/bulk", summary="Launch multiple experiment batches", status_code=201)
async def create_experiment_bundle(
    task_id: str,
    body: CreateExperimentBundleRequest,
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    """Submit baseline/grid/bayesian batches together from the V3 workbench."""
    from app.services import tuning_service

    strategies = [item.model_dump() for item in body.strategies]
    return await tuning_service.dispatch_experiment_bundle(
        db,
        modeling_task_id=task_id,
        name=body.name,
        strategies=strategies,
        description=body.description,
    )


@router.post("/{task_id}/runs/{run_id}/deploy", summary="Deploy the model trained by a run", status_code=201)
async def deploy_run_route(
    task_id: str,
    run_id: str,
    body: DeployRunRequest,
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    """Bridge a V3 run to a live deployment (workflow 部署上线 step).

    Resolves the run's underlying domain model (ML TrainingTask / DL
    DLTrainingTask) and reuses the existing deployment services. Only
    SUCCESS runs are deployable.
    """
    return await modeling_task_service.deploy_run(
        db,
        task_id,
        run_id,
        name=body.name,
        description=body.description,
        max_batch_size=body.max_batch_size,
    )

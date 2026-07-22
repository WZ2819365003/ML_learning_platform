"""
V3 Platform Experiments API — /api/platform/experiments

Manages PlatformExperiment + ExperimentRun lifecycle including AutoML dispatch.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.database import get_db
from app.services import experiment_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/platform/experiments", tags=["Platform Experiments V3"])


# ---------------------------------------------------------------------------
# Request schemas
# ---------------------------------------------------------------------------

class CreateExperimentRequest(BaseModel):
    name: str
    description: str | None = None
    dataset_id: str | None = None
    objective_metric: str = "accuracy"
    objective_direction: str = "max"  # max | min
    kind: str = "single"              # single | automl | grid_search
    config: dict | None = None


class CreateRunRequest(BaseModel):
    params: dict | None = None
    parent_run_id: str | None = None
    notes: str | None = None


class AutoMLRequest(BaseModel):
    candidates: list[dict] = Field(
        description=(
            "List of candidate configs, each a dict with keys: "
            "model_type, hyperparameters, dataset_id, target_column, ..."
        )
    )


class AutoMLRegistryRequest(BaseModel):
    """Use the built-in YAML candidate registry for AutoML search."""
    dataset_id: str
    target_column: str
    task_type: str = "classification"   # classification | regression
    eval_metrics: list[str] | None = None
    test_size: float = 0.2
    max_candidates: int | None = Field(None, ge=1, le=20, description="Limit candidates (default: all)")


# ---------------------------------------------------------------------------
# Experiment endpoints
# ---------------------------------------------------------------------------

@router.get("/", summary="List experiments")
async def list_experiments(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    status: str | None = Query(None),
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    return await experiment_service.list_experiments(db, page=page, page_size=page_size, status=status)


@router.post("/", summary="Create an experiment", status_code=201)
async def create_experiment(
    body: CreateExperimentRequest,
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    return await experiment_service.create_experiment(
        db,
        name=body.name,
        dataset_id=body.dataset_id,
        description=body.description,
        objective_metric=body.objective_metric,
        objective_direction=body.objective_direction,
        kind=body.kind,
        config=body.config,
    )


@router.get("/{experiment_id}", summary="Get experiment")
async def get_experiment(
    experiment_id: str,
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    return await experiment_service.get_experiment(db, experiment_id)


@router.delete("/{experiment_id}", summary="Delete experiment")
async def delete_experiment(
    experiment_id: str,
    db: AsyncSession = Depends(get_db),
) -> dict[str, str]:
    await experiment_service.delete_experiment(db, experiment_id)
    return {"message": "deleted"}


# ---------------------------------------------------------------------------
# Runs under an experiment
# ---------------------------------------------------------------------------

@router.get("/{experiment_id}/runs", summary="List runs for an experiment")
async def list_runs(
    experiment_id: str,
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=200),
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    return await experiment_service.list_runs(db, experiment_id, page=page, page_size=page_size)


@router.post("/{experiment_id}/runs", summary="Create a run", status_code=201)
async def create_run(
    experiment_id: str,
    body: CreateRunRequest,
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    return await experiment_service.create_run(
        db,
        experiment_id=experiment_id,
        params=body.params,
        parent_run_id=body.parent_run_id,
        notes=body.notes,
    )


@router.get("/{experiment_id}/runs/{run_id}", summary="Get a single run")
async def get_run(
    experiment_id: str,
    run_id: str,
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    run = await experiment_service.get_run(db, run_id)
    if run["experiment_id"] != experiment_id:
        raise HTTPException(status_code=404, detail="Run not found in this experiment")
    return run


# ---------------------------------------------------------------------------
# Leaderboard
# ---------------------------------------------------------------------------

@router.get("/{experiment_id}/leaderboard", summary="Get ranked leaderboard")
async def get_leaderboard(
    experiment_id: str,
    db: AsyncSession = Depends(get_db),
) -> list[dict[str, Any]]:
    return await experiment_service.get_leaderboard(db, experiment_id)


# ---------------------------------------------------------------------------
# AutoML: submit N parallel runs
# ---------------------------------------------------------------------------

@router.get("/automl/candidates", summary="List available AutoML candidate models")
async def list_automl_candidates(
    task_type: str = "classification",
) -> dict[str, Any]:
    """Return the candidate list from the registry YAML (for frontend display)."""
    from app.services.automl_service import load_candidates, warm_start_capable
    try:
        candidates = load_candidates(task_type)
        return {
            "task_type": task_type,
            "candidates": candidates,
            "warm_start_capable": warm_start_capable(),
        }
    except (FileNotFoundError, ValueError) as e:
        raise HTTPException(status_code=404, detail=str(e))


# ---------------------------------------------------------------------------
# SHAP explanation for a run
# ---------------------------------------------------------------------------

@router.post("/{experiment_id}/runs/{run_id}/explain", summary="Trigger SHAP explanation")
async def trigger_explain(
    experiment_id: str,
    run_id: str,
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    """
    Trigger a SHAP explanation for a successful ExperimentRun.

    Uses asyncio.create_task (no Celery worker required). The result is stored
    inline in ExperimentRun.metrics["shap_importances"] and optionally in MinIO.
    Poll GET .../explain until it returns 200.
    """
    import asyncio

    from app.scheduler.task_runner import register_domain_task, update_platform_task_status
    from app.services.explain_service import run_shap_explanation

    run = await experiment_service.get_run(db, run_id)
    if run["experiment_id"] != experiment_id:
        raise HTTPException(status_code=404, detail="Run not found in this experiment")
    if run["status"] != "SUCCESS":
        raise HTTPException(status_code=400, detail="Can only explain a successful run")

    # Check idempotency — don't re-run if importances already stored
    if (run.get("metrics") or {}).get("shap_importances"):
        return {"platform_task_id": None, "status": "SUCCESS", "already_computed": True}

    ptask = await register_domain_task(
        db=db,
        kind="explain",
        payload_ref=f"explain:{run_id}",
        priority=7,
    )
    await db.commit()

    async def _explain_wrapper(platform_task_id: str) -> None:
        try:
            await update_platform_task_status(platform_task_id, "RUNNING")
            result = await run_shap_explanation(run_id, platform_task_id)
            await update_platform_task_status(
                platform_task_id, "SUCCESS",
                metrics={"feature_count": len(result.get("feature_importances", {}))},
            )
        except Exception as exc:
            logger.error("SHAP explain failed for run %r: %s", run_id, exc)
            await update_platform_task_status(platform_task_id, "FAILED", error=str(exc))

    asyncio.create_task(_explain_wrapper(ptask.id))
    return {"platform_task_id": ptask.id, "status": "QUEUED"}


@router.get("/{experiment_id}/runs/{run_id}/explain", summary="Get SHAP explanation result")
async def get_explain_result(
    experiment_id: str,
    run_id: str,
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any] | None:
    from app.services.explain_service import get_shap_result
    run = await experiment_service.get_run(db, run_id)
    if run["experiment_id"] != experiment_id:
        raise HTTPException(status_code=404, detail="Run not found in this experiment")
    result = await get_shap_result(run_id)
    if result is None:
        raise HTTPException(status_code=404, detail="SHAP explanation not yet available")
    return result

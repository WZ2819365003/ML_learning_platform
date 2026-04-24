"""
V3 Cross-Task Run List — /api/v3/runs

A flat listing of every ExperimentRun across all ModelingTasks. Powers the
"Run 诊断中心" nav page (and any future bulk-select workflows) so the user
doesn't have to drill into each modeling task to find a specific run.

This router lives on its own prefix (not nested under /v3/tasks) to avoid
colliding with the parameterised `/v3/tasks/{task_id}` routes.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.database import get_db
from app.services import modeling_task_service

router = APIRouter(prefix="/v3/runs", tags=["V3 Run List"])


@router.get("/", summary="List all ExperimentRun rows across modeling tasks")
async def list_all_runs(
    status: str | None = Query(None, description="SUCCESS | FAILED | RUNNING | PENDING"),
    strategy_type: str | None = Query(None, description="baseline | grid_search | bayesian_search"),
    task_type: str | None = Query(None, description="classification | regression"),
    limit: int = Query(500, ge=1, le=2000),
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    """Flat cross-task run listing.

    Rows shape::

        {
          run_id, experiment_id, experiment_name, strategy_type,
          task_id, task_name, task_type,
          objective_metric, objective_direction, objective_value,
          trial_no, rank, status, model_type,
          created_at, finished_at,
        }
    """
    return await modeling_task_service.list_all_runs(
        db,
        status=status,
        strategy_type=strategy_type,
        task_type=task_type,
        limit=limit,
    )

"""
TrainingPlan service — CRUD for reusable training-plan templates.

A TrainingPlan captures (strategy_type, selected_models, search_space,
budget_config, eval_metrics) + human metadata; it is dataset-agnostic.  When a
user launches an experiment batch on a ModelingTask they can optionally pick a
plan and the batch-config form gets prepopulated.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from fastapi import HTTPException
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.database import TrainingPlan

logger = logging.getLogger(__name__)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# Serialiser
# ---------------------------------------------------------------------------

def _serialise(plan: TrainingPlan) -> dict[str, Any]:
    return {
        "id": plan.id,
        "name": plan.name,
        "description": plan.description,
        "task_type": plan.task_type,
        "strategy_type": plan.strategy_type,
        "selected_models": plan.selected_models or [],
        "search_space": plan.search_space or {},
        "budget_config": plan.budget_config or {},
        "eval_metrics": plan.eval_metrics or [],
        "default_objective_metric": plan.default_objective_metric,
        "default_objective_direction": plan.default_objective_direction,
        "use_count": plan.use_count,
        "last_used_at": plan.last_used_at.isoformat() if plan.last_used_at else None,
        "created_at": plan.created_at.isoformat() if plan.created_at else None,
        "updated_at": plan.updated_at.isoformat() if plan.updated_at else None,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_VALID_STRATEGIES = {"baseline", "grid_search", "bayesian_search"}
_VALID_TASK_TYPES = {"classification", "regression"}


def _validate_shape(data: dict[str, Any]) -> None:
    if data.get("task_type") and data["task_type"] not in _VALID_TASK_TYPES:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid task_type {data['task_type']!r}. Expected one of {sorted(_VALID_TASK_TYPES)}",
        )
    if data.get("strategy_type") and data["strategy_type"] not in _VALID_STRATEGIES:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid strategy_type {data['strategy_type']!r}. Expected one of {sorted(_VALID_STRATEGIES)}",
        )
    if data.get("selected_models") is not None and not isinstance(data["selected_models"], list):
        raise HTTPException(status_code=422, detail="selected_models must be a list")
    if data.get("eval_metrics") is not None and not isinstance(data["eval_metrics"], list):
        raise HTTPException(status_code=422, detail="eval_metrics must be a list")


async def _get_or_404(db: AsyncSession, plan_id: str) -> TrainingPlan:
    result = await db.execute(select(TrainingPlan).where(TrainingPlan.id == plan_id))
    plan = result.scalar_one_or_none()
    if plan is None:
        raise HTTPException(status_code=404, detail=f"TrainingPlan {plan_id!r} not found")
    return plan


# ---------------------------------------------------------------------------
# CRUD
# ---------------------------------------------------------------------------

async def list_plans(
    db: AsyncSession,
    task_type: str | None = None,
    page: int = 1,
    page_size: int = 50,
) -> dict[str, Any]:
    stmt = select(TrainingPlan)
    count_stmt = select(func.count(TrainingPlan.id))
    if task_type:
        stmt = stmt.where(TrainingPlan.task_type == task_type)
        count_stmt = count_stmt.where(TrainingPlan.task_type == task_type)

    total = (await db.execute(count_stmt)).scalar_one()
    # MySQL doesn't support NULLS LAST — emulate with an extra IS NULL ordering
    # key (0 for non-null, 1 for null, so non-null rows sort first).
    rows = await db.execute(
        stmt.order_by(
            TrainingPlan.last_used_at.is_(None).asc(),
            TrainingPlan.last_used_at.desc(),
            TrainingPlan.created_at.desc(),
        )
        .offset((page - 1) * page_size)
        .limit(page_size)
    )
    return {
        "items": [_serialise(p) for p in rows.scalars().all()],
        "total": total,
        "page": page,
        "page_size": page_size,
    }


async def create_plan(db: AsyncSession, data: dict[str, Any]) -> dict[str, Any]:
    _validate_shape(data)
    if not data.get("name"):
        raise HTTPException(status_code=422, detail="name is required")
    if not data.get("selected_models"):
        raise HTTPException(status_code=422, detail="selected_models cannot be empty")

    plan = TrainingPlan(
        name=data["name"],
        description=data.get("description"),
        task_type=data.get("task_type", "classification"),
        strategy_type=data.get("strategy_type", "baseline"),
        selected_models=data["selected_models"],
        search_space=data.get("search_space") or None,
        budget_config=data.get("budget_config") or None,
        eval_metrics=data.get("eval_metrics") or None,
        default_objective_metric=data.get("default_objective_metric"),
        default_objective_direction=data.get("default_objective_direction"),
    )
    db.add(plan)
    await db.flush()
    await db.refresh(plan)
    logger.info("Created TrainingPlan %s (%s / %s)", plan.id, plan.task_type, plan.strategy_type)
    return _serialise(plan)


async def get_plan(db: AsyncSession, plan_id: str) -> dict[str, Any]:
    return _serialise(await _get_or_404(db, plan_id))


async def update_plan(
    db: AsyncSession, plan_id: str, data: dict[str, Any]
) -> dict[str, Any]:
    _validate_shape(data)
    plan = await _get_or_404(db, plan_id)
    # Only update fields actually present in the payload
    for key in (
        "name", "description", "task_type", "strategy_type",
        "selected_models", "search_space", "budget_config", "eval_metrics",
        "default_objective_metric", "default_objective_direction",
    ):
        if key in data:
            setattr(plan, key, data[key])
    await db.flush()
    await db.refresh(plan)
    return _serialise(plan)


async def delete_plan(db: AsyncSession, plan_id: str) -> None:
    plan = await _get_or_404(db, plan_id)
    await db.delete(plan)
    await db.flush()


async def mark_used(db: AsyncSession, plan_id: str) -> dict[str, Any]:
    """Bump use_count + last_used_at (called when a batch is dispatched with this plan)."""
    plan = await _get_or_404(db, plan_id)
    plan.use_count = (plan.use_count or 0) + 1
    plan.last_used_at = _utcnow()
    await db.flush()
    await db.refresh(plan)
    return _serialise(plan)

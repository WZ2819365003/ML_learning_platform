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

from app.core.model_registry import resolve_model_family
from app.core.dl_registry import build_default_dl_config
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
        "model_family": plan.model_family or "ml",
        "selected_models": plan.selected_models or [],
        "search_space": plan.search_space or {},
        "dl_config": plan.dl_config or {},
        "budget_config": plan.budget_config or {},
        "eval_metrics": plan.eval_metrics or [],
        "default_objective_metric": plan.default_objective_metric,
        "default_objective_direction": plan.default_objective_direction,
        "use_count": plan.use_count,
        "version": plan.version or 1,
        "last_used_at": plan.last_used_at.isoformat() if plan.last_used_at else None,
        "created_at": plan.created_at.isoformat() if plan.created_at else None,
        "updated_at": plan.updated_at.isoformat() if plan.updated_at else None,
    }


# ---------------------------------------------------------------------------
# Snapshot helper (V3 Phase 2)
# ---------------------------------------------------------------------------
# ModelingTask.training_plan_snapshot stores a frozen copy of the plan at the
# moment of binding so subsequent edits/deletes to the plan don't rewrite
# history.  ``capture_snapshot`` produces the canonical structure — callers
# should never build this dict inline.

def capture_snapshot(plan: TrainingPlan) -> dict[str, Any]:
    """
    Freeze a TrainingPlan into the snapshot payload stored on ModelingTask.

    Structure::

        {
            "plan_id":      "<id>",
            "plan_name":    "<name>",
            "plan_version": <int>,
            "captured_at":  "<iso8601>",
            "payload":      { full _serialise() output },
            "dag_shape":    { "num_models": N, "strategy": "baseline"|... },
        }

    The ``dag_shape`` section gives a cheap hint for Phase 5 dispatchers that
    want to decide up-front how many child PlatformTasks a batch will fan out
    to without re-walking the whole payload.
    """
    payload = _serialise(plan)
    selected = plan.selected_models or []
    return {
        "plan_id": plan.id,
        "plan_name": plan.name,
        "plan_version": plan.version or 1,
        "captured_at": _utcnow().isoformat(),
        "payload": payload,
        "dag_shape": {
            "num_models": len(selected),
            "strategy": plan.strategy_type,
            "model_family": plan.model_family or "ml",
        },
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_VALID_STRATEGIES = {"baseline", "grid_search", "bayesian_search"}
_VALID_TASK_TYPES = {"classification", "regression"}
_VALID_FAMILIES = {"ml", "dl", "mixed"}


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
    if data.get("model_family") and data["model_family"] not in _VALID_FAMILIES:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid model_family {data['model_family']!r}. Expected one of {sorted(_VALID_FAMILIES)}",
        )
    if data.get("selected_models") is not None and not isinstance(data["selected_models"], list):
        raise HTTPException(status_code=422, detail="selected_models must be a list")
    if data.get("eval_metrics") is not None and not isinstance(data["eval_metrics"], list):
        raise HTTPException(status_code=422, detail="eval_metrics must be a list")
    if data.get("dl_config") is not None and not isinstance(data["dl_config"], dict):
        raise HTTPException(status_code=422, detail="dl_config must be a dict")


def _validate_family_vs_models(
    family: str, selected_models: list[str]
) -> tuple[list[str], list[str]]:
    """
    Ensure every selected_model belongs to the expected registry and return
    (ml_tokens, dl_tokens) splits for downstream dispatch.  Raises 422 if any
    token is unknown or inconsistent with the declared family.
    """
    ml_tokens: list[str] = []
    dl_tokens: list[str] = []
    for token in selected_models:
        resolved = resolve_model_family(token)
        if resolved is None:
            raise HTTPException(
                status_code=422,
                detail=f"Unknown model {token!r} (not in ML or DL registry)",
            )
        if family == "ml" and resolved != "ml":
            raise HTTPException(
                status_code=422,
                detail=f"Model {token!r} is a DL model but plan.model_family='ml'",
            )
        if family == "dl" and resolved != "dl":
            raise HTTPException(
                status_code=422,
                detail=f"Model {token!r} is an ML model but plan.model_family='dl'",
            )
        if resolved == "ml":
            ml_tokens.append(token)
        else:
            dl_tokens.append(token)
    return ml_tokens, dl_tokens


def _fill_dl_config_defaults(
    dl_tokens: list[str], dl_config: dict | None
) -> dict:
    """Ensure every DL token has an entry in dl_config, backfilling from registry defaults."""
    out = dict(dl_config or {})
    for token in dl_tokens:
        if token not in out:
            out[token] = build_default_dl_config(token)
    return out


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

    family = data.get("model_family", "ml")
    _, dl_tokens = _validate_family_vs_models(family, data["selected_models"])
    dl_config = _fill_dl_config_defaults(dl_tokens, data.get("dl_config"))

    plan = TrainingPlan(
        name=data["name"],
        description=data.get("description"),
        task_type=data.get("task_type", "classification"),
        strategy_type=data.get("strategy_type", "baseline"),
        model_family=family,
        selected_models=data["selected_models"],
        search_space=data.get("search_space") or None,
        dl_config=dl_config or None,
        budget_config=data.get("budget_config") or None,
        eval_metrics=data.get("eval_metrics") or None,
        default_objective_metric=data.get("default_objective_metric"),
        default_objective_direction=data.get("default_objective_direction"),
    )
    db.add(plan)
    await db.flush()
    await db.refresh(plan)
    logger.info(
        "Created TrainingPlan %s (%s / %s / family=%s, %d DL models)",
        plan.id, plan.task_type, plan.strategy_type, plan.model_family, len(dl_tokens),
    )
    return _serialise(plan)


async def get_plan(db: AsyncSession, plan_id: str) -> dict[str, Any]:
    return _serialise(await _get_or_404(db, plan_id))


async def update_plan(
    db: AsyncSession, plan_id: str, data: dict[str, Any]
) -> dict[str, Any]:
    _validate_shape(data)
    plan = await _get_or_404(db, plan_id)

    # If selected_models or model_family change, re-validate the combination
    # against the registries so we never persist an inconsistent plan.
    next_family = data.get("model_family", plan.model_family or "ml")
    next_models = data.get("selected_models", plan.selected_models or [])
    if "model_family" in data or "selected_models" in data:
        _, dl_tokens = _validate_family_vs_models(next_family, next_models)
        # Backfill dl_config for any newly added DL tokens; preserve existing entries.
        merged_dl = _fill_dl_config_defaults(
            dl_tokens, data.get("dl_config", plan.dl_config)
        )
        if merged_dl:
            data = {**data, "dl_config": merged_dl}

    # Only the substantive "recipe" fields should bump the version counter;
    # cosmetic edits (name, description) leave version untouched so consumers
    # can rely on version changes to mean "the plan is materially different".
    _VERSION_BUMPING_KEYS = {
        "task_type", "strategy_type", "model_family", "selected_models",
        "search_space", "dl_config", "budget_config", "eval_metrics",
        "default_objective_metric", "default_objective_direction",
    }
    bump_version = any(k in data for k in _VERSION_BUMPING_KEYS)

    for key in (
        "name", "description", "task_type", "strategy_type",
        "model_family", "selected_models", "search_space", "dl_config",
        "budget_config", "eval_metrics",
        "default_objective_metric", "default_objective_direction",
    ):
        if key in data:
            setattr(plan, key, data[key])
    if bump_version:
        plan.version = (plan.version or 1) + 1
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

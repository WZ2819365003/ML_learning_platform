"""
Training Plan API — /api/platform/training-plans

CRUD for reusable training-plan templates.  A plan is dataset-agnostic and
captures (strategy_type, selected_models, search_space, budget_config,
eval_metrics).  Consumers (workbench batch-config UI) call /list to
populate a picker and /{id} to load full details for form prefill.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.database import get_db
from app.services import training_plan_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/platform/training-plans", tags=["training-plans"])


@router.get("", summary="List training plans (optionally filtered by task_type)")
async def list_plans(
    task_type: str | None = Query(default=None),
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=50, ge=1, le=200),
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    return await training_plan_service.list_plans(
        db, task_type=task_type, page=page, page_size=page_size
    )


@router.post("", summary="Create a training plan", status_code=201)
async def create_plan(
    payload: dict[str, Any],
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    return await training_plan_service.create_plan(db, payload)


@router.get("/{plan_id}", summary="Get one training plan")
async def get_plan(
    plan_id: str,
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    return await training_plan_service.get_plan(db, plan_id)


@router.patch("/{plan_id}", summary="Update a training plan (partial)")
async def update_plan(
    plan_id: str,
    payload: dict[str, Any],
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    return await training_plan_service.update_plan(db, plan_id, payload)


@router.delete("/{plan_id}", summary="Delete a training plan")
async def delete_plan(
    plan_id: str,
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    await training_plan_service.delete_plan(db, plan_id)
    return {"deleted": True, "id": plan_id}


@router.post("/{plan_id}/mark-used", summary="Bump use_count after applying")
async def mark_used(
    plan_id: str,
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    return await training_plan_service.mark_used(db, plan_id)

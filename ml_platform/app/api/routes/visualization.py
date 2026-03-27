"""Visualization routes — charts and explainability data."""

from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.database import get_db
from app.services.viz_service import (
    get_confusion_matrix,
    get_roc_curve,
    get_feature_importance,
    get_learning_curve,
    get_shap_summary,
)

router = APIRouter(prefix="/viz", tags=["Visualization"])


@router.get("/{task_id}/confusion_matrix")
async def confusion_matrix_route(
    task_id: str,
    normalize: bool = Query(False, description="Normalize by row"),
    db: AsyncSession = Depends(get_db),
):
    """Return confusion matrix data for a completed training task."""
    return await get_confusion_matrix(task_id, db, normalize=normalize)


@router.get("/{task_id}/roc_curve")
async def roc_curve_route(
    task_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Return ROC curve data (FPR, TPR, AUC)."""
    return await get_roc_curve(task_id, db)


@router.get("/{task_id}/feature_importance")
async def feature_importance_route(
    task_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Return feature importance scores."""
    return await get_feature_importance(task_id, db)


@router.get("/{task_id}/learning_curve")
async def learning_curve_route(
    task_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Return per-step/fold training metrics for learning curve."""
    return await get_learning_curve(task_id, db)


@router.get("/{task_id}/shap_summary")
async def shap_summary_route(
    task_id: str,
    max_samples: int = Query(100, ge=10, le=500, description="Max samples for SHAP"),
    db: AsyncSession = Depends(get_db),
):
    """Return SHAP values for model explainability."""
    return await get_shap_summary(task_id, db, max_samples=max_samples)

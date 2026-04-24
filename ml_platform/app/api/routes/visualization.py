"""Visualization routes — charts and explainability data."""

from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.database import get_db
from app.services.viz_service import (
    get_calibration_curve,
    get_confusion_matrix,
    get_feature_importance,
    get_learning_curve,
    get_per_class_metrics,
    get_pr_curve,
    get_prediction_distribution,
    get_predicted_vs_actual,
    get_residual_plot,
    get_roc_curve,
    get_shap_summary,
    get_threshold_analysis,
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


@router.get("/{task_id}/residual_plot")
async def residual_plot_route(
    task_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Return residuals vs predicted values for regression tasks."""
    return await get_residual_plot(task_id, db)


@router.get("/{task_id}/predicted_vs_actual")
async def predicted_vs_actual_route(
    task_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Return predicted vs actual scatter data for regression tasks."""
    return await get_predicted_vs_actual(task_id, db)


# ---------------------------------------------------------------------------
# Advanced classification viz — per-class metrics, PR curve, calibration,
# threshold tuning, prediction distribution.
# ---------------------------------------------------------------------------


@router.get("/{task_id}/per_class")
async def per_class_route(
    task_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Per-class precision / recall / F1 / support (classification only)."""
    return await get_per_class_metrics(task_id, db)


@router.get("/{task_id}/pr_curve")
async def pr_curve_route(
    task_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Precision-Recall curve + Average Precision + best-F1 threshold."""
    return await get_pr_curve(task_id, db)


@router.get("/{task_id}/calibration")
async def calibration_route(
    task_id: str,
    n_bins: int = Query(10, ge=3, le=50),
    db: AsyncSession = Depends(get_db),
):
    """Calibration (reliability) curve + ECE + Brier score (binary only)."""
    return await get_calibration_curve(task_id, db, n_bins=n_bins)


@router.get("/{task_id}/threshold")
async def threshold_route(
    task_id: str,
    step: float = Query(0.05, ge=0.01, le=0.5),
    db: AsyncSession = Depends(get_db),
):
    """Sweep binary classification thresholds; returns P/R/F1/accuracy per step."""
    return await get_threshold_analysis(task_id, db, step=step)


@router.get("/{task_id}/distribution")
async def distribution_route(
    task_id: str,
    bins: int = Query(30, ge=10, le=100),
    db: AsyncSession = Depends(get_db),
):
    """Prediction distribution — probability histogram (classification)
    or residual histogram (regression)."""
    return await get_prediction_distribution(task_id, db, bins=bins)

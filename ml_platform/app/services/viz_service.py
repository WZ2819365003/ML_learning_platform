"""Visualization service — computes data for charts and plots.

Task resolution and SHAP computation live in `resolver.py` / `shap_service.py`;
this module stays focused on the per-chart metric computation. Back-compat
aliases (`_get_task_and_dataset`, `_TaskFacade`, `_is_regressor`,
`_load_and_split_data`, `_load_task_model_data`) are preserved so existing
in-function imports (e.g. `model_mgmt.py`) keep working.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import numpy as np
from fastapi import HTTPException
from sklearn.metrics import auc, confusion_matrix, roc_curve
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.services import shap_service
from app.services.resolver import (
    TaskFacade,
    is_regressor,
    load_and_split_data_no_stratify,
    load_and_split_data_stratified,
    load_model,
    resolve_legacy_id_candidates,
    resolve_task_and_dataset,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Back-compat aliases — kept so existing `from .viz_service import _*` work.
# New code should import directly from `app.services.resolver`.
# ---------------------------------------------------------------------------

_TaskFacade = TaskFacade
_is_regressor = is_regressor
_load_model = load_model
_load_and_split_data = load_and_split_data_stratified
_get_task_and_dataset = resolve_task_and_dataset
_resolve_metrics_candidate_ids = resolve_legacy_id_candidates


def _load_task_model_data(task, dataset, test_size: float = 0.2):
    """Non-stratified loader compatible with older callers.

    Returns (model, X_train, X_test, y_train, y_test, feature_names).
    """
    X_train, X_test, y_train, y_test, feature_names = load_and_split_data_no_stratify(
        dataset.file_path, task.target_column, test_size
    )
    model = load_model(task.model_path)
    return model, X_train, X_test, y_train, y_test, feature_names


# ---------------------------------------------------------------------------
# Chart data endpoints
# ---------------------------------------------------------------------------


async def get_confusion_matrix(task_id: str, db: AsyncSession, normalize: bool = False) -> dict:
    """Compute confusion matrix for a completed training task."""
    task, dataset = await resolve_task_and_dataset(task_id, db)
    if is_regressor(task.model_type):
        raise HTTPException(
            status_code=400,
            detail="该任务为回归任务，不支持混淆矩阵。请查看残差图或预测-真实值散点。",
        )
    model = load_model(task.model_path)
    _, X_test, _, y_test, _, class_labels = load_and_split_data_stratified(
        dataset.file_path, task.target_column, task.test_size
    )

    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    if normalize:
        cm = cm.astype(float)
        row_sums = cm.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        cm = np.round(cm / row_sums, 4)

    return {
        "labels": class_labels,
        "matrix": cm.tolist(),
        "normalized": normalize,
    }


async def get_roc_curve(task_id: str, db: AsyncSession) -> dict:
    """Compute ROC curve data for a completed training task."""
    task, dataset = await resolve_task_and_dataset(task_id, db)
    if is_regressor(task.model_type):
        raise HTTPException(
            status_code=400,
            detail="该任务为回归任务，不支持 ROC 曲线。请查看残差图或预测-真实值散点。",
        )
    model = load_model(task.model_path)
    _, X_test, _, y_test, _, class_labels = load_and_split_data_stratified(
        dataset.file_path, task.target_column, task.test_size
    )

    if not hasattr(model, "predict_proba"):
        raise HTTPException(status_code=400, detail="Model does not support probability prediction for ROC")

    y_proba = model.predict_proba(X_test)

    if y_proba.shape[1] == 2:
        fpr, tpr, thresholds = roc_curve(y_test, y_proba[:, 1])
        roc_auc = float(auc(fpr, tpr))
        return {
            "fpr": [round(float(x), 4) for x in fpr],
            "tpr": [round(float(x), 4) for x in tpr],
            "auc": round(roc_auc, 4),
            "thresholds": [round(float(x), 4) if np.isfinite(x) else 1.0 for x in thresholds],
        }

    from sklearn.preprocessing import label_binarize
    classes = np.unique(y_test)
    y_test_bin = label_binarize(y_test, classes=classes)
    curves = []
    for i, cls in enumerate(classes):
        fpr_i, tpr_i, _ = roc_curve(y_test_bin[:, i], y_proba[:, i])
        auc_i = float(auc(fpr_i, tpr_i))
        curves.append({
            "class": class_labels[i] if i < len(class_labels) else str(cls),
            "fpr": [round(float(x), 4) for x in fpr_i],
            "tpr": [round(float(x), 4) for x in tpr_i],
            "auc": round(auc_i, 4),
        })
    return {"curves": curves, "multiclass": True}


async def get_feature_importance(task_id: str, db: AsyncSession) -> dict:
    """Feature importance extracted from the model (tree-based or linear)."""
    task, dataset = await resolve_task_and_dataset(task_id, db)
    model = load_model(task.model_path)
    # Non-stratified split — we only need feature names. Works for regression too.
    _, _, _, _, feature_names = load_and_split_data_no_stratify(
        dataset.file_path, task.target_column, task.test_size
    )

    if hasattr(model, "feature_importances_"):
        importance = np.asarray(model.feature_importances_, dtype=np.float64)
    elif hasattr(model, "coef_"):
        coef = np.asarray(model.coef_, dtype=np.float64)
        importance = np.abs(coef).mean(axis=0) if coef.ndim > 1 else np.abs(coef)
    else:
        raise HTTPException(status_code=400, detail="Model does not provide feature importance")

    indices = np.argsort(importance)[::-1]
    sorted_features = [feature_names[i] for i in indices]
    sorted_importance = [round(float(importance[i]), 6) for i in indices]

    return {
        "features": sorted_features,
        "importance": sorted_importance,
    }


async def get_learning_curve(task_id: str, db: AsyncSession) -> dict:
    """Pick the step-metric log file for this task via the id candidate chain."""
    settings = get_settings()

    metrics_file = None
    resolved_id = task_id
    for cid in await resolve_legacy_id_candidates(task_id, db):
        candidate = settings.storage_logs / f"{cid}_metrics.json"
        if candidate.exists():
            metrics_file = candidate
            resolved_id = cid
            break

    if metrics_file is None:
        raise HTTPException(status_code=404, detail="No metrics data found")

    with open(metrics_file, "r") as f:
        data = json.load(f)

    steps = data.get("steps", [])
    if not steps:
        raise HTTPException(status_code=404, detail="No training steps recorded")

    return {
        "task_id": task_id,
        "resolved_id": resolved_id,
        "model_type": data.get("model_type", ""),
        "steps": steps,
    }


async def get_shap_summary(task_id: str, db: AsyncSession, max_samples: int = 200) -> dict[str, Any]:
    """Thin wrapper around `shap_service.compute_shap_summary` — keeps viz_service
    as the single import surface for route handlers.
    """
    try:
        return await shap_service.compute_shap_summary(task_id, db, max_samples=max_samples)
    except HTTPException:
        raise
    except ImportError as exc:
        raise HTTPException(status_code=503, detail=f"SHAP library not installed: {exc}")
    except Exception as exc:
        logger.exception("SHAP computation failed")
        raise HTTPException(status_code=500, detail=f"SHAP computation failed: {exc}")


# ---------------------------------------------------------------------------
# Regression-specific viz
# ---------------------------------------------------------------------------


async def get_residual_plot(task_id: str, db: AsyncSession) -> dict:
    """Return residuals (y_true - y_pred) and predicted values for residual plot."""
    task, dataset = await resolve_task_and_dataset(task_id, db)
    X_train, X_test, y_train, y_test, _ = load_and_split_data_no_stratify(
        dataset.file_path, task.target_column, task.test_size
    )
    model = load_model(task.model_path)

    y_pred = model.predict(X_test)
    residuals = (y_test - y_pred).tolist()
    y_pred_list = y_pred.tolist()

    return {
        "task_id": task_id,
        "predicted": [round(float(v), 4) for v in y_pred_list],
        "residuals": [round(float(v), 4) for v in residuals],
        "mean_residual": round(float(np.mean(residuals)), 4),
        "std_residual": round(float(np.std(residuals)), 4),
    }


async def get_predicted_vs_actual(task_id: str, db: AsyncSession) -> dict:
    """Return predicted vs actual values for scatter plot."""
    task, dataset = await resolve_task_and_dataset(task_id, db)
    X_train, X_test, y_train, y_test, _ = load_and_split_data_no_stratify(
        dataset.file_path, task.target_column, task.test_size
    )
    model = load_model(task.model_path)

    y_pred = model.predict(X_test)

    return {
        "task_id": task_id,
        "actual": [round(float(v), 4) for v in y_test.tolist()],
        "predicted": [round(float(v), 4) for v in y_pred.tolist()],
    }

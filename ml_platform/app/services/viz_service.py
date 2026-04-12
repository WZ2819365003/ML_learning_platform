"""Visualization service — computes data for charts and plots."""

import json
import logging
from typing import Any

import joblib
import numpy as np
import pandas as pd
from fastapi import HTTPException
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.model_selection import train_test_split
from sqlalchemy import select

from app.config import get_settings
from app.models.database import AsyncSession, Dataset, TrainingTask
from app.services.prediction_service import load_dataframe, prepare_training_frame
from app.utils.storage_paths import resolve_runtime_path

logger = logging.getLogger(__name__)


def _load_model(model_path: str):
    """Load a saved sklearn model."""
    path = resolve_runtime_path(model_path)
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Model file not found: {model_path}")
    return joblib.load(path)


def _load_and_split_data(file_path: str, target_column: str, test_size: float = 0.2):
    """Load dataset and split into train/test, mirroring training logic."""
    df = load_dataframe(file_path)
    X, y, _, target_encoder = prepare_training_frame(df, target_column)

    X_train, X_test, y_train, y_test = train_test_split(
        X.values, y.values, test_size=test_size, random_state=42, stratify=y.values
    )

    feature_names = list(X.columns)
    class_labels = (
        target_encoder.classes_.tolist()
        if target_encoder is not None
        else sorted([str(c) for c in np.unique(y.values)])
    )

    return X_train, X_test, y_train, y_test, feature_names, class_labels


async def _get_task_and_dataset(task_id: str, db: AsyncSession):
    """Load training task and its dataset."""
    result = await db.execute(select(TrainingTask).where(TrainingTask.id == task_id))
    task = result.scalar_one_or_none()
    if task is None:
        raise HTTPException(status_code=404, detail="Training task not found")
    if task.status != "SUCCESS":
        raise HTTPException(status_code=400, detail=f"Task not completed (status={task.status})")
    if not task.model_path:
        raise HTTPException(status_code=400, detail="No model saved for this task")

    result = await db.execute(select(Dataset).where(Dataset.id == task.dataset_id))
    dataset = result.scalar_one_or_none()
    if dataset is None:
        raise HTTPException(status_code=404, detail="Dataset not found")

    return task, dataset


async def get_confusion_matrix(task_id: str, db: AsyncSession, normalize: bool = False) -> dict:
    """Compute confusion matrix for a completed training task."""
    task, dataset = await _get_task_and_dataset(task_id, db)
    model = _load_model(task.model_path)
    _, X_test, _, y_test, _, class_labels = _load_and_split_data(
        dataset.file_path, task.target_column, task.test_size
    )

    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    if normalize:
        cm = cm.astype(float)
        row_sums = cm.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # avoid division by zero
        cm = np.round(cm / row_sums, 4)

    return {
        "labels": class_labels,
        "matrix": cm.tolist(),
        "normalized": normalize,
    }


async def get_roc_curve(task_id: str, db: AsyncSession) -> dict:
    """Compute ROC curve data for a completed training task."""
    task, dataset = await _get_task_and_dataset(task_id, db)
    model = _load_model(task.model_path)
    _, X_test, _, y_test, _, class_labels = _load_and_split_data(
        dataset.file_path, task.target_column, task.test_size
    )

    if not hasattr(model, 'predict_proba'):
        raise HTTPException(status_code=400, detail="Model does not support probability prediction for ROC")

    y_proba = model.predict_proba(X_test)

    # Binary classification
    if y_proba.shape[1] == 2:
        fpr, tpr, thresholds = roc_curve(y_test, y_proba[:, 1])
        roc_auc = float(auc(fpr, tpr))
        return {
            "fpr": [round(float(x), 4) for x in fpr],
            "tpr": [round(float(x), 4) for x in tpr],
            "auc": round(roc_auc, 4),
            "thresholds": [round(float(x), 4) if np.isfinite(x) else 1.0 for x in thresholds],
        }

    # Multi-class: one-vs-rest
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
    """Get feature importance from the trained model."""
    task, dataset = await _get_task_and_dataset(task_id, db)
    model = _load_model(task.model_path)
    _, _, _, _, feature_names, _ = _load_and_split_data(
        dataset.file_path, task.target_column, task.test_size
    )

    importance = None

    # Tree-based models
    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
    # Linear models
    elif hasattr(model, 'coef_'):
        coef = model.coef_
        if coef.ndim > 1:
            importance = np.abs(coef).mean(axis=0)
        else:
            importance = np.abs(coef)
    else:
        raise HTTPException(status_code=400, detail="Model does not provide feature importance")

    # Sort by importance descending
    indices = np.argsort(importance)[::-1]
    sorted_features = [feature_names[i] for i in indices]
    sorted_importance = [round(float(importance[i]), 6) for i in indices]

    return {
        "features": sorted_features,
        "importance": sorted_importance,
    }


async def get_learning_curve(task_id: str, db: AsyncSession) -> dict:
    """Get learning curve data from metrics log."""
    settings = get_settings()
    metrics_file = settings.storage_logs / f"{task_id}_metrics.json"

    if not metrics_file.exists():
        raise HTTPException(status_code=404, detail="No metrics data found")

    with open(metrics_file, "r") as f:
        data = json.load(f)

    steps = data.get("steps", [])
    if not steps:
        raise HTTPException(status_code=404, detail="No training steps recorded")

    return {
        "task_id": task_id,
        "model_type": data.get("model_type", ""),
        "steps": steps,
    }


async def get_shap_summary(task_id: str, db: AsyncSession, max_samples: int = 100) -> dict:
    """Compute SHAP values for model explainability."""
    task, dataset = await _get_task_and_dataset(task_id, db)
    model = _load_model(task.model_path)

    X_train, X_test, _, _, feature_names, _ = _load_and_split_data(
        dataset.file_path, task.target_column, task.test_size
    )

    # Limit samples
    if len(X_test) > max_samples:
        indices = np.random.RandomState(42).choice(len(X_test), max_samples, replace=False)
        X_sample = X_test[indices]
    else:
        X_sample = X_test

    try:
        import shap

        # Choose appropriate explainer
        model_type = task.model_type
        if model_type in ('random_forest', 'xgboost', 'lightgbm'):
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_sample)
        else:
            # Use KernelExplainer for other models (slower)
            background = shap.sample(pd.DataFrame(X_train, columns=feature_names), min(50, len(X_train)))
            explainer = shap.KernelExplainer(model.predict_proba, background)
            shap_values = explainer.shap_values(X_sample)

        # Handle multi-output (binary classification returns list of 2)
        if isinstance(shap_values, list):
            # Take the positive class (index 1) for binary, or first class for multi
            shap_values = shap_values[1] if len(shap_values) == 2 else shap_values[0]

        # Compute mean absolute SHAP per feature
        mean_abs_shap = np.abs(shap_values).mean(axis=0).tolist()

        return {
            "features": feature_names,
            "mean_abs_shap": [round(float(v), 6) for v in mean_abs_shap],
            "shap_values": [[round(float(v), 4) for v in row] for row in shap_values.tolist()],
            "feature_values": [[round(float(v), 4) for v in row] for row in X_sample.tolist()],
            "num_samples": len(X_sample),
        }

    except ImportError:
        raise HTTPException(status_code=503, detail="SHAP library not installed")
    except Exception as e:
        logger.error("SHAP computation failed: %s", str(e))
        raise HTTPException(status_code=500, detail=f"SHAP computation failed: {str(e)}")


# ---------------------------------------------------------------------------
# Regression visualization helpers
# ---------------------------------------------------------------------------

def _load_task_model_data(task, dataset, test_size: float = 0.2):
    """Load model + split data for regression tasks (no stratify on continuous target).

    This is a separate helper from _load_and_split_data which uses stratify=y.values
    (only valid for classification). Regression targets are continuous so stratify
    must be omitted.
    """
    df = load_dataframe(dataset.file_path)
    X, y, _, target_encoder = prepare_training_frame(df, task.target_column)

    # No stratify — regression targets are continuous
    X_train, X_test, y_train, y_test = train_test_split(
        X.values, y.values, test_size=test_size, random_state=42
    )

    model = _load_model(task.model_path)
    feature_names = list(X.columns)
    return model, X_train, X_test, y_train, y_test, feature_names


async def get_residual_plot(task_id: str, db: AsyncSession) -> dict:
    """Return residuals (y_true - y_pred) and predicted values for residual plot."""
    task, dataset = await _get_task_and_dataset(task_id, db)
    model, _, X_test, _, y_test, _ = _load_task_model_data(task, dataset, task.test_size)

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
    task, dataset = await _get_task_and_dataset(task_id, db)
    model, _, X_test, _, y_test, _ = _load_task_model_data(task, dataset, task.test_size)

    y_pred = model.predict(X_test)

    return {
        "task_id": task_id,
        "actual": [round(float(v), 4) for v in y_test.tolist()],
        "predicted": [round(float(v), 4) for v in y_pred.tolist()],
    }

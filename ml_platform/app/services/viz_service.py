"""Visualization service — computes data for charts and plots.

Task resolution and SHAP computation live in `resolver.py` / `shap_service.py`;
this module stays focused on the per-chart metric computation.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import numpy as np
from fastapi import HTTPException
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    auc,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_curve,
)
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.services import shap_service
from app.services.resolver import (
    is_regressor,
    resolve_and_load,
    resolve_legacy_id_candidates,
)
from app.services.object_storage import restore_file

logger = logging.getLogger(__name__)




# ---------------------------------------------------------------------------
# Chart data endpoints
# ---------------------------------------------------------------------------


async def get_confusion_matrix(task_id: str, db: AsyncSession, normalize: bool = False) -> dict:
    """Compute confusion matrix for a completed training task."""
    prepared = await resolve_and_load(task_id, db)
    task = prepared["task"]
    if is_regressor(task.model_type):
        raise HTTPException(
            status_code=400,
            detail="该任务为回归任务，不支持混淆矩阵。请查看残差图或预测-真实值散点。",
        )
    model = prepared["model"]
    X_test = prepared["X_test"]
    y_test = prepared["y_test"]
    class_labels = prepared["class_labels"]

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
    prepared = await resolve_and_load(task_id, db)
    task = prepared["task"]
    if is_regressor(task.model_type):
        raise HTTPException(
            status_code=400,
            detail="该任务为回归任务，不支持 ROC 曲线。请查看残差图或预测-真实值散点。",
        )
    model = prepared["model"]
    X_test = prepared["X_test"]
    y_test = prepared["y_test"]
    class_labels = prepared["class_labels"]

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
    prepared = await resolve_and_load(task_id, db, stratified=False)
    model = prepared["model"]
    feature_names = prepared["feature_names"]

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
        if not candidate.exists():
            restore_file(candidate, [f"logs/{cid}_metrics.json"])
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


# Where a computed explanation is parked on the task row, and the shape that
# tells us whether a cached one still answers the current request.
_SHAP_CACHE_KEY = "shap_cache"


def _cached_shap(task: Any, max_samples: int) -> dict[str, Any] | None:
    """Return a previously computed summary if it matches this request."""
    metrics = getattr(task, "result_metrics", None) or {}
    cached = metrics.get(_SHAP_CACHE_KEY)
    if not isinstance(cached, dict):
        return None
    # A summary computed over fewer samples is not the one being asked for.
    if cached.get("max_samples") != max_samples:
        return None
    payload = cached.get("payload")
    return payload if isinstance(payload, dict) else None


async def _store_shap(db: AsyncSession, task: Any, max_samples: int, payload: dict) -> None:
    """Park the computed summary on the task row. Never raises.

    Losing the cache costs a recomputation; failing the request the user just
    waited minutes for would be worse, so a storage problem is logged and
    swallowed.
    """
    try:
        if not hasattr(task, "result_metrics"):
            return   # an on-disk orphan facade has no row to write back to
        metrics = dict(task.result_metrics or {})
        metrics[_SHAP_CACHE_KEY] = {"max_samples": max_samples, "payload": payload}
        task.result_metrics = metrics
        await db.flush()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Caching SHAP summary for %s failed: %s", getattr(task, "id", "?"), exc)


async def get_shap_summary(
    task_id: str,
    db: AsyncSession,
    max_samples: int = 200,
    *,
    refresh: bool = False,
) -> dict[str, Any]:
    """Return a SHAP summary, computing it only when there is no usable cache.

    SHAP is the most expensive thing this service does — a TreeExplainer on a
    deep forest once took six minutes in production — and the result does not
    change unless the model does. Recomputing it every time the tab is opened
    made an already slow operation feel broken, so the payload is parked on the
    task row and returned directly next time. `refresh=True` forces a new run.
    """
    if not refresh:
        try:
            task, _dataset = await shap_service.resolve_task_and_dataset(task_id, db)
            cached = _cached_shap(task, max_samples)
            if cached is not None:
                return {**cached, "cached": True}
        except HTTPException:
            raise
        except Exception as exc:  # noqa: BLE001 — a cache miss must not break the request
            logger.warning("SHAP cache lookup for %s failed: %s", task_id, exc)

    try:
        payload = await shap_service.compute_shap_summary(task_id, db, max_samples=max_samples)
        try:
            task, _dataset = await shap_service.resolve_task_and_dataset(task_id, db)
            await _store_shap(db, task, max_samples, payload)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not cache SHAP summary for %s: %s", task_id, exc)
        return {**payload, "cached": False}
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


def _tail_evaluation_rows(X_test, y_test, max_samples: int):
    """Return a deterministic tail window without aggregating chart values.

    Visualization endpoints do not need to predict an entire large hold-out
    set merely to draw at most 1,000 points.  This is a view window rather than
    statistical resampling: rows keep their original order and values.
    """
    total = int(len(y_test))
    limit = max(1, min(int(max_samples), total)) if total else 0
    start = max(0, total - limit)
    return X_test[start:], y_test[start:], total, start


async def get_residual_plot(
    task_id: str, db: AsyncSession, max_samples: int = 1000
) -> dict:
    """Return residuals (y_true - y_pred) and predicted values for residual plot."""
    prepared = await resolve_and_load(task_id, db, stratified=False)
    model = prepared["model"]
    X_test, y_test, total_count, sample_offset = _tail_evaluation_rows(
        prepared["X_test"], prepared["y_test"], max_samples
    )

    y_pred = model.predict(X_test)
    residuals = (y_test - y_pred).tolist()
    y_pred_list = y_pred.tolist()

    return {
        "task_id": task_id,
        "predicted": [round(float(v), 4) for v in y_pred_list],
        "residuals": [round(float(v), 4) for v in residuals],
        "mean_residual": round(float(np.mean(residuals)), 4),
        "std_residual": round(float(np.std(residuals)), 4),
        "sample_count": len(y_pred_list),
        "total_count": total_count,
        "sample_offset": sample_offset,
        "truncated": len(y_pred_list) < total_count,
    }


async def get_predicted_vs_actual(
    task_id: str, db: AsyncSession, max_samples: int = 1000
) -> dict:
    """Return a bounded predicted/actual window for scatter and line charts."""
    prepared = await resolve_and_load(task_id, db, stratified=False)
    model = prepared["model"]
    X_test, y_test, total_count, sample_offset = _tail_evaluation_rows(
        prepared["X_test"], prepared["y_test"], max_samples
    )

    y_pred = model.predict(X_test)

    return {
        "task_id": task_id,
        "actual": [round(float(v), 4) for v in y_test.tolist()],
        "predicted": [round(float(v), 4) for v in y_pred.tolist()],
        "sample_count": int(len(y_pred)),
        "total_count": total_count,
        "sample_offset": sample_offset,
        "truncated": int(len(y_pred)) < total_count,
    }


# ---------------------------------------------------------------------------
# Advanced viz endpoints — per-class metrics, PR curve, calibration,
# threshold tuning, prediction distribution.  These back the new
# "professional" tabs in Results.jsx and RunInspector.
# ---------------------------------------------------------------------------


def _guard_classification(model_type: str | None, endpoint: str) -> None:
    if is_regressor(model_type):
        raise HTTPException(
            status_code=400,
            detail=f"{endpoint} 仅适用于分类任务，当前任务为回归。",
        )


async def get_per_class_metrics(task_id: str, db: AsyncSession) -> dict:
    """Return sklearn.classification_report as a structured dict.

    Each class gets precision / recall / f1 / support; macro + weighted
    averages are included.  The frontend table can render one row per
    class with the last two rows pinned as aggregate.
    """
    prepared = await resolve_and_load(task_id, db)
    task = prepared["task"]
    _guard_classification(task.model_type, "per_class metrics")
    X_test = prepared["X_test"]
    y_test = prepared["y_test"]
    class_labels = prepared["class_labels"]
    model = prepared["model"]
    y_pred = model.predict(X_test)

    report = classification_report(
        y_test, y_pred, target_names=class_labels, output_dict=True, zero_division=0
    )

    rows: list[dict[str, Any]] = []
    aggregate_keys = {"accuracy", "macro avg", "weighted avg"}
    for label, vals in report.items():
        if label in aggregate_keys or not isinstance(vals, dict):
            continue
        rows.append({
            "label": label,
            "precision": round(float(vals.get("precision", 0.0)), 4),
            "recall": round(float(vals.get("recall", 0.0)), 4),
            "f1": round(float(vals.get("f1-score", 0.0)), 4),
            "support": int(vals.get("support", 0)),
        })

    def _agg(key: str) -> dict[str, Any] | None:
        v = report.get(key)
        if not isinstance(v, dict):
            return None
        return {
            "label": key,
            "precision": round(float(v.get("precision", 0.0)), 4),
            "recall": round(float(v.get("recall", 0.0)), 4),
            "f1": round(float(v.get("f1-score", 0.0)), 4),
            "support": int(v.get("support", 0)),
        }

    return {
        "task_id": task_id,
        "rows": rows,
        "macro_avg": _agg("macro avg"),
        "weighted_avg": _agg("weighted avg"),
        "accuracy": round(float(report.get("accuracy", 0.0)), 4),
    }


async def get_pr_curve(task_id: str, db: AsyncSession) -> dict:
    """Precision-Recall curve + Average Precision + best-F1 threshold.

    For binary tasks, returns the full (precision, recall, thresholds)
    arrays; the frontend plots PR + vertical line at best_threshold.
    Multiclass is one-vs-rest per class.
    """
    prepared = await resolve_and_load(task_id, db)
    task = prepared["task"]
    _guard_classification(task.model_type, "PR curve")
    X_test = prepared["X_test"]
    y_test = prepared["y_test"]
    class_labels = prepared["class_labels"]
    model = prepared["model"]
    if not hasattr(model, "predict_proba"):
        raise HTTPException(
            status_code=400,
            detail="该模型不支持 predict_proba，无法绘制 PR 曲线。",
        )

    y_proba = model.predict_proba(X_test)
    classes = np.asarray(model.classes_) if hasattr(model, "classes_") else np.unique(y_test)

    if y_proba.shape[1] == 2:
        positive = classes[1]
        y_true_bin = (np.asarray(y_test) == positive).astype(int)
        precision, recall, thresholds = precision_recall_curve(y_true_bin, y_proba[:, 1])
        ap = float(auc(recall, precision))
        # Best F1 threshold: walk the curve, handle length-mismatch (precision
        # has one more point than thresholds by sklearn convention).
        f1_scores = np.asarray([
            (2 * p * r / (p + r)) if (p + r) > 0 else 0.0
            for p, r in zip(precision[:-1], recall[:-1])
        ])
        best_idx = int(np.argmax(f1_scores)) if f1_scores.size else 0
        return {
            "task_id": task_id,
            "multiclass": False,
            "positive_label": str(positive),
            "precision": [round(float(x), 4) for x in precision],
            "recall": [round(float(x), 4) for x in recall],
            "thresholds": [round(float(x), 4) for x in thresholds],
            "average_precision": round(ap, 4),
            "best_threshold": round(float(thresholds[best_idx]) if thresholds.size else 0.5, 4),
            "best_f1": round(float(f1_scores[best_idx]) if f1_scores.size else 0.0, 4),
        }

    # Multiclass: one-vs-rest per class.
    from sklearn.preprocessing import label_binarize
    y_true_bin = label_binarize(y_test, classes=classes)
    curves = []
    for i, cls in enumerate(classes):
        precision_i, recall_i, _ = precision_recall_curve(y_true_bin[:, i], y_proba[:, i])
        ap_i = float(auc(recall_i, precision_i))
        curves.append({
            "class": class_labels[i] if i < len(class_labels) else str(cls),
            "precision": [round(float(x), 4) for x in precision_i],
            "recall": [round(float(x), 4) for x in recall_i],
            "average_precision": round(ap_i, 4),
        })
    return {"task_id": task_id, "multiclass": True, "curves": curves}


async def get_calibration_curve(task_id: str, db: AsyncSession, n_bins: int = 10) -> dict:
    """Calibration (reliability) curve + Expected Calibration Error + Brier score.

    Binary-only.  ECE is the mean of |prob_pred - prob_true| weighted by
    per-bin count.  Good calibration means the curve hugs the diagonal.
    """
    prepared = await resolve_and_load(task_id, db)
    task = prepared["task"]
    _guard_classification(task.model_type, "calibration curve")
    X_test = prepared["X_test"]
    y_test = prepared["y_test"]
    model = prepared["model"]
    if not hasattr(model, "predict_proba"):
        raise HTTPException(
            status_code=400,
            detail="该模型不支持 predict_proba，无法绘制校准曲线。",
        )

    y_proba = model.predict_proba(X_test)
    classes = np.asarray(model.classes_) if hasattr(model, "classes_") else np.unique(y_test)
    if y_proba.shape[1] != 2:
        raise HTTPException(
            status_code=400,
            detail="校准曲线目前仅支持二分类任务。",
        )

    positive = classes[1]
    y_true = (np.asarray(y_test) == positive).astype(int)
    scores = y_proba[:, 1]

    prob_true, prob_pred = calibration_curve(y_true, scores, n_bins=n_bins, strategy="uniform")

    # ECE: per-bin weighted |gap|
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_idx = np.digitize(scores, bin_edges[1:-1], right=False)
    ece = 0.0
    total = len(scores)
    for b in range(n_bins):
        mask = bin_idx == b
        if not mask.any():
            continue
        avg_conf = float(np.mean(scores[mask]))
        avg_acc = float(np.mean(y_true[mask]))
        ece += (mask.sum() / total) * abs(avg_conf - avg_acc)

    brier = float(brier_score_loss(y_true, scores))

    return {
        "task_id": task_id,
        "positive_label": str(positive),
        "prob_pred": [round(float(x), 4) for x in prob_pred],
        "prob_true": [round(float(x), 4) for x in prob_true],
        "n_bins": n_bins,
        "ece": round(ece, 4),
        "brier": round(brier, 4),
    }


async def get_threshold_analysis(
    task_id: str, db: AsyncSession, step: float = 0.05
) -> dict:
    """Sweep classification thresholds and return per-threshold metrics.

    Binary classification only.  Produces ~19 rows (0.05 → 0.95, step=0.05
    by default) with precision / recall / F1 / accuracy.  The UI picks the
    best-F1 row to highlight.
    """
    prepared = await resolve_and_load(task_id, db)
    task = prepared["task"]
    _guard_classification(task.model_type, "threshold analysis")
    X_test = prepared["X_test"]
    y_test = prepared["y_test"]
    model = prepared["model"]
    if not hasattr(model, "predict_proba"):
        raise HTTPException(
            status_code=400,
            detail="该模型不支持 predict_proba，无法做阈值分析。",
        )

    y_proba = model.predict_proba(X_test)
    classes = np.asarray(model.classes_) if hasattr(model, "classes_") else np.unique(y_test)
    if y_proba.shape[1] != 2:
        raise HTTPException(
            status_code=400,
            detail="阈值分析目前仅支持二分类任务。",
        )
    positive = classes[1]
    y_true = (np.asarray(y_test) == positive).astype(int)
    scores = y_proba[:, 1]

    step = max(0.01, min(0.5, float(step)))
    thresholds = np.arange(step, 1.0, step)
    rows = []
    for thr in thresholds:
        y_pred = (scores >= thr).astype(int)
        precision = float(precision_score(y_true, y_pred, zero_division=0))
        recall = float(recall_score(y_true, y_pred, zero_division=0))
        f1 = float(f1_score(y_true, y_pred, zero_division=0))
        accuracy = float(np.mean(y_pred == y_true))
        rows.append({
            "threshold": round(float(thr), 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "accuracy": round(accuracy, 4),
        })

    best_row = max(rows, key=lambda r: r["f1"]) if rows else None
    return {
        "task_id": task_id,
        "positive_label": str(positive),
        "rows": rows,
        "best_threshold": best_row["threshold"] if best_row else 0.5,
    }


async def get_prediction_distribution(
    task_id: str, db: AsyncSession, bins: int = 30, max_samples: int = 5000
) -> dict:
    """Prediction distribution for classification (probability histogram)
    or regression (residual histogram).

    For classification: histogram of positive-class probabilities, split
    by true class to see separation.  For regression: residual histogram
    + summary stats.
    """
    prepared = await resolve_and_load(task_id, db)
    task = prepared["task"]
    model = prepared["model"]
    X_test, y_test, total_count, sample_offset = _tail_evaluation_rows(
        prepared["X_test"], prepared["y_test"], max_samples
    )
    bins = max(10, min(100, int(bins)))

    if is_regressor(task.model_type):
        y_pred = model.predict(X_test)
        residuals = np.asarray(y_test) - np.asarray(y_pred)
        counts, edges = np.histogram(residuals, bins=bins)
        return {
            "task_id": task_id,
            "kind": "regression_residuals",
            "bin_edges": [round(float(e), 4) for e in edges],
            "counts": [int(c) for c in counts],
            "mean": round(float(np.mean(residuals)), 4),
            "std": round(float(np.std(residuals)), 4),
            "min": round(float(np.min(residuals)), 4),
            "max": round(float(np.max(residuals)), 4),
            "sample_count": int(len(residuals)),
            "total_count": total_count,
            "sample_offset": sample_offset,
            "truncated": int(len(residuals)) < total_count,
        }

    class_labels = prepared["class_labels"]
    if not hasattr(model, "predict_proba"):
        raise HTTPException(
            status_code=400,
            detail="该模型不支持 predict_proba，无法绘制概率分布。",
        )

    y_proba = model.predict_proba(X_test)
    classes = np.asarray(model.classes_) if hasattr(model, "classes_") else np.unique(y_test)
    if y_proba.shape[1] != 2:
        # Multiclass: histogram of max-probability per sample
        max_probs = np.max(y_proba, axis=1)
        counts, edges = np.histogram(max_probs, bins=bins, range=(0.0, 1.0))
        return {
            "task_id": task_id,
            "kind": "classification_confidence_multiclass",
            "bin_edges": [round(float(e), 4) for e in edges],
            "counts": [int(c) for c in counts],
            "n_classes": int(y_proba.shape[1]),
            "sample_count": int(len(max_probs)),
            "total_count": total_count,
            "truncated": int(len(max_probs)) < total_count,
        }

    positive = classes[1]
    positive_scores = y_proba[:, 1]
    y_true = np.asarray(y_test) == positive

    pos_counts, edges = np.histogram(positive_scores[y_true], bins=bins, range=(0.0, 1.0))
    neg_counts, _ = np.histogram(positive_scores[~y_true], bins=bins, range=(0.0, 1.0))

    return {
        "task_id": task_id,
        "kind": "classification_binary_proba",
        "positive_label": str(positive),
        "bin_edges": [round(float(e), 4) for e in edges],
        "positive_counts": [int(c) for c in pos_counts],
        "negative_counts": [int(c) for c in neg_counts],
        "sample_count": int(len(positive_scores)),
        "total_count": total_count,
        "truncated": int(len(positive_scores)) < total_count,
    }

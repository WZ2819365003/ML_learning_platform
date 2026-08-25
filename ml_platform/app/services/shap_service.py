"""
Unified SHAP / feature-importance service.

Replaces the two previously-duplicated SHAP paths (`viz_service.get_shap_summary`
for the ML Results page and `explain_service.run_shap_explanation` for V3 runs).
Both now delegate here so the explanations they return are computed identically.

Fallback ladder (generic → specialised):

    1. **TreeExplainer** — for `random_forest / xgboost / lightgbm` (fast, exact).
    2. **KernelExplainer** — for any model that exposes `predict` / `predict_proba`
       with a 50-sample background set. Slower but model-agnostic.
    3. **Permutation importance** — pure `sklearn.inspection.permutation_importance`
       tail when SHAP itself errors. No per-sample `shap_values`, but still
       gives a reliable importance ranking so the UI can label
       "SHAP 不可用，展示排列重要度" rather than fabricating feature_importances.

Every numeric array is serialized via `np.asarray(..., dtype=np.float64)` to
sidestep the numpy 2.0 `np.inexact` dtype-cast rejection that was silently
collapsing SHAP output to `feature_importances_`.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import tempfile
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.database import DLTrainingTask
from app.services.dl_shap_adapter import build_dl_shap_context
from app.services.resolver import (
    TaskFacade,
    is_regressor,
    load_and_split_data_for_model,
    load_model,
    resolve_task_and_dataset,
)

logger = logging.getLogger(__name__)


# Method labels — kept stable so FE can switch UI text on them.
METHOD_TREE = "tree"
METHOD_KERNEL = "kernel"
METHOD_PERMUTATION = "permutation"


_TREE_MODEL_PREFIXES = ("random_forest", "xgboost", "lightgbm", "gradient_boost", "extra_trees")
_DL_MAX_BACKGROUND = 20
_DL_MAX_SAMPLES = 50


def _to_f64_list(arr) -> list:
    """numpy 2.0-safe: cast to np.float64 before .tolist().

    Raw `arr.astype(float)` refuses when `arr.dtype` is a `np.inexact` subclass
    under numpy 2.0+. `np.asarray(..., dtype=np.float64)` is always safe.
    """
    return np.asarray(arr, dtype=np.float64).tolist()


def _round_list_2d(arr, decimals: int = 6) -> list[list[float]]:
    a = np.asarray(arr, dtype=np.float64)
    return np.round(a, decimals).tolist()


def _normalize_shap_values(shap_values):
    """Coerce SHAP output into (sv_per_sample 2D, base_value_scalar) regardless of
    SHAP's legacy-list or new-array multiclass shape.

    Returns:
        sv_per_sample: np.ndarray of shape (n_samples, n_features)
        selected_class_idx: int | None — which class we picked for multiclass
    """
    selected = None
    if isinstance(shap_values, list):
        # legacy SHAP API: list of per-class 2D arrays
        arr = np.asarray(shap_values)
        if arr.ndim == 3:
            selected = 1 if arr.shape[0] == 2 else 0
            sv = arr[selected]
        else:
            sv = arr
    else:
        arr = np.asarray(shap_values)
        if arr.ndim == 3:
            # new API: (n_samples, n_features, n_classes)
            selected = 1 if arr.shape[-1] == 2 else 0
            sv = arr[:, :, selected]
        else:
            sv = arr
    return sv, selected


def _normalize_base_value(base, selected_class_idx: int | None):
    if hasattr(base, "tolist"):
        try:
            base = base.tolist()
        except Exception:
            return None
    if isinstance(base, list) and selected_class_idx is not None:
        if selected_class_idx >= len(base):
            return None
        base = base[selected_class_idx]
    try:
        return float(base) if isinstance(base, (int, float)) else base
    except Exception:
        return None


def _compute_tree(
    model, X_sample
) -> tuple[np.ndarray, float | list | None, int | None]:
    """Tree-based SHAP — returns (sv_per_sample 2D, base_value)."""
    import shap
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(X_sample)
    sv, selected = _normalize_shap_values(sv)
    base = getattr(explainer, "expected_value", None)
    return sv, _normalize_base_value(base, selected), selected


def _compute_kernel(
    model, X_train, X_sample, feature_names
) -> tuple[np.ndarray, float | list | None, int | None]:
    """Kernel SHAP — model-agnostic, slower, needs background samples."""
    import shap
    predict_fn = getattr(model, "predict_proba", None) or model.predict
    bg_df = pd.DataFrame(X_train, columns=feature_names)
    background = shap.sample(bg_df, min(50, len(bg_df)))
    explainer = shap.KernelExplainer(predict_fn, background)
    sv = explainer.shap_values(X_sample, nsamples=50)
    sv, selected = _normalize_shap_values(sv)
    base = getattr(explainer, "expected_value", None)
    return sv, _normalize_base_value(base, selected), selected


def _compute_permutation(model, X_sample, y_sample, feature_names) -> np.ndarray:
    """Permutation-importance tail — returns 1D mean-importance vector only.

    No per-sample SHAP; UI must hide beeswarm/waterfall and show the
    importance bar chart with a "SHAP 不可用" note.
    """
    from sklearn.inspection import permutation_importance
    result = permutation_importance(
        model, X_sample, y_sample,
        n_repeats=5, random_state=42, n_jobs=1,
    )
    return np.asarray(result.importances_mean, dtype=np.float64)


def _is_tree_model(model_type: str | None, model: Any) -> bool:
    """Identify tree models robustly — prefer model_type, fall back to class name."""
    mt = (model_type or "").lower()
    if any(mt.startswith(p) for p in _TREE_MODEL_PREFIXES):
        return True
    cls_name = type(model).__name__.lower()
    for kw in ("randomforest", "xgb", "xgboost", "lgbm", "lightgbm", "gradientboost", "extratrees"):
        if kw in cls_name:
            return True
    return False


def _build_payload(
    *,
    method: str,
    feature_names: list[str],
    mean_abs_shap: np.ndarray,
    sv_per_sample: np.ndarray | None = None,
    feature_values: np.ndarray | None = None,
    base_value: float | list | None = None,
    sample_count: int,
    class_index: int | None = None,
    task_kind: str = "classification",
    max_ui_samples: int = 300,
) -> dict[str, Any]:
    """Pack everything the UI needs into one dict with consistent shape.

    `sv_per_sample`/`feature_values` are downsampled to `max_ui_samples` rows so
    the JSON blob stays under ~50KB for typical tabular data (permutation
    method returns both as None — the bar chart alone still renders).
    """
    importances_sorted = sorted(
        zip(feature_names, [float(v) for v in mean_abs_shap]),
        key=lambda x: abs(x[1]),
        reverse=True,
    )
    feature_importances = dict(importances_sorted)

    top_features = []
    # Direction-aware if we have per-sample SHAP (tree/kernel); permutation ≥ 0 always.
    for feat, imp in importances_sorted[:10]:
        direction = None
        if sv_per_sample is not None and method != METHOD_PERMUTATION:
            try:
                idx = feature_names.index(feat)
                mean_signed = float(np.asarray(sv_per_sample[:, idx], dtype=np.float64).mean())
                if mean_signed > 1e-9:
                    direction = "up"
                elif mean_signed < -1e-9:
                    direction = "down"
                else:
                    direction = "neutral"
            except Exception:
                pass
        top_features.append({
            "feature": feat,
            "importance": float(imp),
            "direction": direction,
        })

    payload: dict[str, Any] = {
        "status": "ready",
        "method": method,
        "task_kind": task_kind,
        "base_value": base_value,
        "sample_count": sample_count,
        "feature_count": len(feature_names),
        "feature_names": feature_names,
        "feature_importances": feature_importances,
        "mean_abs_shap": [float(v) for v in mean_abs_shap],
        "top_features": top_features,
        "class_index": class_index,
    }

    if sv_per_sample is not None and feature_values is not None:
        n = int(min(len(sv_per_sample), max_ui_samples))
        if n < len(sv_per_sample):
            rng = np.random.default_rng(42)
            idx = rng.choice(len(sv_per_sample), size=n, replace=False)
            idx.sort()
            sv_ui = sv_per_sample[idx]
            fv_ui = feature_values[idx]
        else:
            sv_ui = sv_per_sample
            fv_ui = feature_values

        payload["shap_values"] = _round_list_2d(sv_ui, 6)
        # Preserve NaN as None so JSON is valid
        fv_f64 = np.asarray(fv_ui, dtype=np.float64)
        payload["feature_values"] = np.where(
            np.isnan(fv_f64), None, np.round(fv_f64, 6)
        ).tolist()
    else:
        payload["shap_values"] = None
        payload["feature_values"] = None

    return payload


async def compute_shap_summary(
    task_id: str,
    db: AsyncSession,
    *,
    max_samples: int = 200,
) -> dict[str, Any]:
    """Compute a SHAP summary for `task_id` using the fallback ladder.

    `task_id` may be a legacy TrainingTask id, a V3 ExperimentRun id, a
    PlatformTask id, or an orphan — `resolver.resolve_task_and_dataset`
    handles the branching.

    Only the task/dataset lookup runs on the event loop (it needs the async
    DB session). Everything after that — model loading, data prep, and the
    SHAP computation itself — is synchronous, CPU-bound scikit-learn/shap
    code with no `await` in it, so it runs in a worker thread via
    ``asyncio.to_thread``. Without this, one slow explanation (e.g.
    TreeExplainer on a deep, unbounded-depth RandomForest — its cost scales
    roughly quadratically with tree depth) blocks the *entire* event loop:
    every other request, including the container health check, hangs for as
    long as the computation runs. That is what "the whole backend crashed"
    actually was on 2026-08-25 — a 6-minute SHAP call with nothing in this
    file ever yielding control back to the loop.
    """
    task, dataset = await resolve_task_and_dataset(task_id, db)
    return await asyncio.to_thread(_compute_shap_summary_sync, task_id, task, dataset, max_samples)


def _compute_shap_summary_sync(
    task_id: str,
    task: Any,
    dataset: Any,
    max_samples: int,
) -> dict[str, Any]:
    """Synchronous body of ``compute_shap_summary`` — safe to run in a thread."""
    if isinstance(task, DLTrainingTask):
        context = build_dl_shap_context(
            task,
            dataset,
            max_background=_DL_MAX_BACKGROUND,
            max_samples=min(max(1, int(max_samples)), _DL_MAX_SAMPLES),
        )
        sv_per_sample, base_value, selected_class_idx = _compute_kernel(
            context.model,
            context.X_background,
            context.X_sample,
            context.feature_names,
        )
        mean_abs_shap = np.abs(
            np.asarray(sv_per_sample, dtype=np.float64)
        ).mean(axis=0)
        payload = _build_payload(
            method=METHOD_KERNEL,
            feature_names=context.feature_names,
            mean_abs_shap=mean_abs_shap,
            sv_per_sample=sv_per_sample,
            feature_values=context.X_sample,
            base_value=base_value,
            sample_count=len(context.X_sample),
            class_index=selected_class_idx,
            task_kind=context.task_kind,
        )
        payload["task_id"] = task.id
        return payload

    loaded_model = load_model(task.model_path)
    prepared = load_and_split_data_for_model(
        dataset.file_path,
        task.target_column,
        task.test_size,
        loaded_model,
        stratified=False,
    )
    X_train = prepared["X_train"]
    X_test = prepared["X_test"]
    y_train = prepared["y_train"]
    y_test = prepared["y_test"]
    feature_names = prepared["feature_names"]
    model = prepared["model"]

    # Subsample X_test for SHAP — bounded for performance
    if len(X_test) > max_samples:
        rng = np.random.RandomState(42)
        idx = rng.choice(len(X_test), max_samples, replace=False)
        X_sample = X_test[idx]
        y_sample = y_test[idx]
    else:
        X_sample = X_test
        y_sample = y_test

    model_type = getattr(task, "model_type", None)
    task_kind = "regression" if is_regressor(model_type) else "classification"

    method = None
    sv_per_sample = None
    base_value = None
    mean_abs_shap = None
    selected_class_idx = None

    # ---- Ladder rung 1: TreeExplainer -------------------------------------
    if _is_tree_model(model_type, model):
        try:
            sv_per_sample, base_value, selected_class_idx = _compute_tree(
                model, X_sample
            )
            mean_abs_shap = np.abs(np.asarray(sv_per_sample, dtype=np.float64)).mean(axis=0)
            method = METHOD_TREE
        except Exception as exc:
            logger.warning("TreeExplainer failed for %s: %s", task_id, exc)

    # ---- Ladder rung 2: KernelExplainer -----------------------------------
    if method is None:
        try:
            sv_per_sample, base_value, selected_class_idx = _compute_kernel(
                model, X_train, X_sample, feature_names
            )
            mean_abs_shap = np.abs(np.asarray(sv_per_sample, dtype=np.float64)).mean(axis=0)
            method = METHOD_KERNEL
        except Exception as exc:
            logger.warning("KernelExplainer failed for %s: %s", task_id, exc)

    # ---- Ladder rung 3: permutation_importance (no shap_values) -----------
    if method is None:
        try:
            mean_abs_shap = _compute_permutation(model, X_sample, y_sample, feature_names)
            sv_per_sample = None
            base_value = None
            method = METHOD_PERMUTATION
        except Exception as exc:
            logger.error("Permutation importance also failed for %s: %s", task_id, exc)
            raise

    payload = _build_payload(
        method=method,
        feature_names=feature_names,
        mean_abs_shap=mean_abs_shap,
        sv_per_sample=sv_per_sample,
        feature_values=np.asarray(X_sample, dtype=np.float64) if sv_per_sample is not None else None,
        base_value=base_value,
        sample_count=len(X_sample),
        class_index=selected_class_idx,
        task_kind=task_kind,
    )
    payload["task_id"] = getattr(task, "id", task_id)
    return payload


# ---------------------------------------------------------------------------
# V3 run integration — compute + persist + MinIO cache
# ---------------------------------------------------------------------------


async def compute_and_persist_shap_for_run(run_id: str) -> dict[str, Any]:
    """Run the SHAP ladder for a V3 ExperimentRun and persist the result.

    Stores a compact summary inline on `run.metrics["shap_importances"]` +
    `shap_method` + `shap_sample_size`, and if MinIO/S3 is configured uploads
    the full per-sample payload as `explanations/{run_id}/shap_summary.json`
    and records the object key on `run.artifacts_uri`.
    """
    from app.models.database import ExperimentRun, async_session_factory
    from app.services.object_storage import upload_file
    from app.config import get_settings

    async with async_session_factory() as db:
        payload = await compute_shap_summary(run_id, db)

    object_key: str | None = None
    try:
        settings = get_settings()
        if settings.s3_enabled:
            object_key = f"explanations/{run_id}/shap_summary.json"
            with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
                json.dump(payload, f)
                tmp_path = f.name
            try:
                upload_file(tmp_path, object_key)
                logger.info("SHAP results uploaded to %s", object_key)
            finally:
                os.unlink(tmp_path)
    except Exception as exc:
        logger.warning("MinIO upload for SHAP failed (%s); storing inline only", exc)
        object_key = None

    async with async_session_factory() as db:
        run = (await db.execute(select(ExperimentRun).where(ExperimentRun.id == run_id))).scalar_one_or_none()
        if run is not None:
            merged = dict(run.metrics or {})
            merged["shap_importances"] = payload["feature_importances"]
            merged["shap_sample_size"] = payload["sample_count"]
            merged["shap_method"] = payload["method"]
            merged["shap_base_value"] = payload.get("base_value")
            merged["shap_task_kind"] = payload.get("task_kind")
            run.metrics = merged
            if object_key:
                run.artifacts_uri = object_key
            await db.commit()

    return {
        "run_id": run_id,
        "feature_importances": payload["feature_importances"],
        "shap_values_uri": object_key,
        "metrics": {"feature_count": payload["feature_count"]},
        "explanation_method": payload["method"],
    }

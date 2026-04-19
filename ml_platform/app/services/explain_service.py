"""
V3 Model Explainability Service — SHAP-based feature importance.

Called by the Celery explain task after a successful training run.
Results are stored in MinIO and referenced via ExperimentRun.artifacts_uri.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from typing import Any

from sqlalchemy import select

logger = logging.getLogger(__name__)

# Default sample size to keep SHAP computation fast
_SHAP_SAMPLE_SIZE = int(os.getenv("SHAP_SAMPLE_SIZE", "500"))


def _fallback_feature_importances(model: Any, X, y=None) -> tuple[dict[str, float], str]:
    """Fallback when SHAP is unavailable in the current runtime."""
    import numpy as np

    if hasattr(model, "feature_importances_"):
        importances = np.asarray(model.feature_importances_, dtype=float)
        return (
            {col: float(val) for col, val in zip(X.columns.tolist(), importances)},
            "model_feature_importances",
        )

    if hasattr(model, "coef_"):
        coef = np.asarray(model.coef_, dtype=float)
        if coef.ndim > 1:
            coef = np.abs(coef).mean(axis=0)
        else:
            coef = np.abs(coef)
        return (
            {col: float(val) for col, val in zip(X.columns.tolist(), coef)},
            "model_coefficients",
        )

    if y is not None:
        from sklearn.inspection import permutation_importance

        result = permutation_importance(
            model,
            X,
            y,
            n_repeats=5,
            random_state=42,
            n_jobs=1,
        )
        importances = np.asarray(result.importances_mean, dtype=float)
        return (
            {col: float(val) for col, val in zip(X.columns.tolist(), importances)},
            "permutation_importance",
        )

    raise ValueError("SHAP is unavailable and the model does not expose fallback feature importance attributes")


async def run_shap_explanation(run_id: str, platform_task_id: str) -> dict[str, Any]:
    """
    Compute SHAP values for the model produced by `run_id`.

    Returns a dict with:
      - feature_importances: {col: importance_score}
      - shap_values_uri: MinIO object key for the full SHAP values JSON
    """
    from app.models.database import ExperimentRun, TrainingTask, Dataset, async_session_factory
    from app.services.object_storage import upload_file

    async with async_session_factory() as db:
        run_result = await db.execute(
            select(ExperimentRun).where(ExperimentRun.id == run_id)
        )
        run = run_result.scalar_one_or_none()
        if run is None:
            raise ValueError(f"ExperimentRun {run_id!r} not found")

        params = run.params or {}
        # payload_ref on the linked PlatformTask encodes the domain task id
        # We store dataset_id and model_path in run.params during AutoML submission
        dataset_id = params.get("dataset_id")
        model_path = params.get("model_path")

        if not dataset_id or not model_path:
            # Try to resolve from the linked training task via task.payload_ref
            if run.task_id:
                from app.models.database import PlatformTask
                ptask_result = await db.execute(
                    select(PlatformTask).where(PlatformTask.id == run.task_id)
                )
                ptask = ptask_result.scalar_one_or_none()
                if ptask and ptask.payload_ref:
                    _, _, domain_task_id = ptask.payload_ref.partition(":")
                    tt_result = await db.execute(
                        select(TrainingTask).where(TrainingTask.id == domain_task_id)
                    )
                    tt = tt_result.scalar_one_or_none()
                    if tt:
                        dataset_id = tt.dataset_id
                        model_path = tt.model_path
                        target_column = tt.target_column

        if not dataset_id or not model_path:
            raise ValueError(f"Cannot resolve dataset_id or model_path for run {run_id!r}")

        ds_result = await db.execute(select(Dataset).where(Dataset.id == dataset_id))
        dataset = ds_result.scalar_one_or_none()
        if dataset is None:
            raise ValueError(f"Dataset {dataset_id!r} not found")

        target_column = params.get("target_column", locals().get("target_column", ""))

    # ── Compute SHAP (synchronous, CPU-bound) ──────────────────────────────
    import joblib
    from app.services.prediction_service import load_dataframe, prepare_training_frame

    model = joblib.load(model_path)
    df = load_dataframe(dataset.file_path)
    y = None
    if target_column:
        X, y, _, _ = prepare_training_frame(df, target_column)
    else:
        X = df.select_dtypes(include=["number", "bool"]).copy()
        if X.empty:
            raise ValueError("SHAP explanation requires numeric features or a resolvable target column")

    # Sample for performance
    if len(X) > _SHAP_SAMPLE_SIZE:
        sampled = X.sample(_SHAP_SAMPLE_SIZE, random_state=42)
        if y is not None:
            y = y.loc[sampled.index]
        X = sampled

    explanation_method = "shap"

    # Select explainer based on model type
    try:
        import shap

        try:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X)
        except Exception:
            try:
                explainer = shap.LinearExplainer(model, X)
                shap_values = explainer.shap_values(X)
            except Exception:
                background = shap.sample(X, min(50, len(X)))
                explainer = shap.KernelExplainer(model.predict, background)
                shap_values = explainer.shap_values(X, nsamples=50)
    except Exception as shap_exc:
        logger.warning("SHAP unavailable for run %s, falling back to model-native importance: %s", run_id, shap_exc)
        feature_importances, explanation_method = _fallback_feature_importances(model, X, y)
        shap_values = None

    # Per-sample SHAP array for beeswarm/dependence plots. Shape normalised to
    # (n_samples, n_features) — for multiclass we pick the positive class
    # (index 1 for binary, index 0 otherwise). UI can later add a class
    # selector once we plumb it through.
    sv_per_sample: "np.ndarray | None" = None

    if shap_values is not None:
        import numpy as np
        try:
            if isinstance(shap_values, list):
                # Legacy SHAP API: list of per-class 2D arrays
                arr = np.array(shap_values)  # (n_classes, n_samples, n_features)
                class_idx = 1 if arr.shape[0] == 2 else 0
                sv_per_sample = arr[class_idx]
                sv_array = np.abs(arr).mean(axis=0)  # mean across classes for importance
            elif shap_values.ndim == 3:
                # New SHAP API: (n_samples, n_features, n_classes)
                class_idx = 1 if shap_values.shape[-1] == 2 else 0
                sv_per_sample = shap_values[:, :, class_idx]
                sv_array = np.abs(shap_values).mean(axis=-1)
            else:
                # 2D: (n_samples, n_features) — regression or already-selected class
                sv_per_sample = shap_values
                sv_array = np.abs(shap_values)

            mean_importances = sv_array.mean(axis=0) if sv_array.ndim == 2 else sv_array
            feature_importances = {
                col: float(imp)
                for col, imp in zip(X.columns.tolist(), mean_importances)
            }
        except Exception as exc:
            logger.warning("SHAP post-processing failed for run %s, using fallback: %s", run_id, exc)
            feature_importances, explanation_method = _fallback_feature_importances(model, X, y)
            sv_per_sample = None

    # Sort descending (preserves insertion order in Python ≥3.7)
    feature_importances = dict(
        sorted(feature_importances.items(), key=lambda x: x[1], reverse=True)
    )

    # ── Build the full MinIO payload ──────────────────────────────────────
    # The inline `run.metrics["shap_importances"]` keeps the aggregated bar
    # chart cheap to render. The MinIO payload carries the per-sample matrix
    # for beeswarm + dependence plots — too big to keep inline.
    object_key = f"explanations/{run_id}/shap_summary.json"
    feature_names = X.columns.tolist()
    payload_json: dict[str, Any] = {
        "run_id": run_id,
        "feature_names": feature_names,
        "feature_importances": feature_importances,
        "sample_size": len(X),
        "feature_count": len(X.columns),
        "explanation_method": explanation_method,
    }
    # Downsample per-sample arrays for UI: 300 is enough for a readable
    # beeswarm and keeps the JSON blob under ~50KB for typical tabular data.
    _UI_SAMPLE_CAP = 300
    if sv_per_sample is not None:
        try:
            import numpy as np

            n = int(min(len(X), _UI_SAMPLE_CAP))
            if n < len(X):
                rng = np.random.default_rng(42)
                idx = rng.choice(len(X), size=n, replace=False)
                idx.sort()
                sv_ui = sv_per_sample[idx]
                fv_ui = X.iloc[idx].to_numpy(dtype=float, na_value=float("nan"))
            else:
                sv_ui = sv_per_sample
                fv_ui = X.to_numpy(dtype=float, na_value=float("nan"))

            # NaN-safe + compact: round to 6 decimals to shrink the blob
            payload_json["samples"] = {
                "feature_values": np.where(np.isnan(fv_ui), None, np.round(fv_ui, 6)).tolist(),
                "shap_values": np.round(sv_ui.astype(float), 6).tolist(),
            }
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not serialise per-sample SHAP for run %s: %s", run_id, exc)
    try:
        from app.config import get_settings
        settings = get_settings()
        if settings.s3_enabled:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
                json.dump(payload_json, f)
                tmp_path = f.name
            try:
                upload_file(tmp_path, object_key)
                logger.info("SHAP results uploaded to %s", object_key)
            finally:
                os.unlink(tmp_path)
        else:
            object_key = None  # MinIO not configured
    except Exception as e:
        logger.warning("MinIO upload for SHAP failed (%s); storing inline only", e)
        object_key = None

    # ── Update ExperimentRun — store importances inline + optional MinIO key ──
    async with async_session_factory() as db:
        run_result = await db.execute(select(ExperimentRun).where(ExperimentRun.id == run_id))
        run = run_result.scalar_one_or_none()
        if run:
            # Merge into existing metrics so training scores are preserved
            merged = dict(run.metrics or {})
            merged["shap_importances"] = feature_importances
            merged["shap_sample_size"] = len(X)
            merged["shap_method"] = explanation_method
            run.metrics = merged
            if object_key:
                run.artifacts_uri = object_key
            await db.commit()

    return {
        "run_id": run_id,
        "feature_importances": feature_importances,
        "shap_values_uri": object_key,
        "metrics": {"feature_count": len(feature_importances)},
        "explanation_method": explanation_method,
    }


async def get_shap_result(run_id: str) -> dict[str, Any] | None:
    """
    Fetch the stored SHAP result for a run.

    Priority:
      1. Inline: `run.metrics["shap_importances"]` — always available after computation.
      2. MinIO presigned URL — only if S3 is configured and artifacts_uri is set.

    Returns None if no explanation has been computed yet.
    """
    from app.models.database import ExperimentRun, async_session_factory

    async with async_session_factory() as db:
        run_result = await db.execute(select(ExperimentRun).where(ExperimentRun.id == run_id))
        run = run_result.scalar_one_or_none()
        if run is None:
            return None

        # Primary path: importances stored inline in run.metrics
        shap_importances = (run.metrics or {}).get("shap_importances")
        if shap_importances:
            return {
                "run_id": run_id,
                "feature_importances": shap_importances,
                "sample_size": (run.metrics or {}).get("shap_sample_size"),
                "artifacts_uri": run.artifacts_uri,
                "explanation_method": (run.metrics or {}).get("shap_method", "shap"),
                "source": "inline",
            }

        # Fallback: MinIO presigned URL
        if not run.artifacts_uri:
            return None
        artifacts_uri = run.artifacts_uri

    try:
        from app.services.object_storage import get_presigned_url
        url = get_presigned_url(artifacts_uri)
        if not url:
            return None
        return {
            "run_id": run_id,
            "download_url": url,
            "artifacts_uri": artifacts_uri,
            "source": "minio",
        }
    except Exception as exc:
        logger.warning("Could not get presigned URL for SHAP artifact: %s", exc)
        return None

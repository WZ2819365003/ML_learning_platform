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

        if not dataset_id or not model_path:
            raise ValueError(f"Cannot resolve dataset_id or model_path for run {run_id!r}")

        ds_result = await db.execute(select(Dataset).where(Dataset.id == dataset_id))
        dataset = ds_result.scalar_one_or_none()
        if dataset is None:
            raise ValueError(f"Dataset {dataset_id!r} not found")

        target_column = params.get("target_column", "")

    # ── Compute SHAP (synchronous, CPU-bound) ──────────────────────────────
    import joblib
    import pandas as pd
    import shap
    from app.utils.storage_paths import resolve_storage_path

    model = joblib.load(model_path)
    df = pd.read_csv(resolve_storage_path(dataset.file_path))

    if target_column and target_column in df.columns:
        X = df.drop(columns=[target_column])
    else:
        X = df

    # Sample for performance
    if len(X) > _SHAP_SAMPLE_SIZE:
        X = X.sample(_SHAP_SAMPLE_SIZE, random_state=42)

    # Select explainer based on model type
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

    # Multi-class: take mean absolute across classes
    import numpy as np
    if isinstance(shap_values, list):
        sv_array = np.abs(np.array(shap_values)).mean(axis=0)
    else:
        sv_array = np.abs(shap_values)

    mean_importances = sv_array.mean(axis=0)
    feature_importances = {
        col: float(imp)
        for col, imp in zip(X.columns.tolist(), mean_importances)
    }

    # Sort descending
    feature_importances = dict(
        sorted(feature_importances.items(), key=lambda x: x[1], reverse=True)
    )

    # ── Try to upload to MinIO (optional) ─────────────────────────────────
    object_key = f"explanations/{run_id}/shap_summary.json"
    payload_json = {
        "run_id": run_id,
        "feature_importances": feature_importances,
        "sample_size": len(X),
        "feature_count": len(X.columns),
    }
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
            run.metrics = merged
            if object_key:
                run.artifacts_uri = object_key
            await db.commit()

    return {
        "run_id": run_id,
        "feature_importances": feature_importances,
        "shap_values_uri": object_key,
        "metrics": {"feature_count": len(feature_importances)},
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

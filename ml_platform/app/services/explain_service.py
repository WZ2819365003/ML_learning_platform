"""
V3 Model Explainability Service — SHAP persistence + MinIO cache.

The actual SHAP computation (Tree → Kernel → Permutation ladder, dtype-safe
serialization) lives in ``shap_service``. This module is the V3-run-specific
integration layer that:

  - Presents the ``run_shap_explanation`` executor signature expected by
    Celery / scheduler / platform_experiments routes
  - Reads back stored explanations via ``get_shap_result``
  - Exposes ``_fallback_feature_importances`` for legacy tests

All paths share the same underlying service, so the ``explanation_method``
reported by V3 inspector matches the method served by the ML Results page.
"""

from __future__ import annotations

import logging
from typing import Any

from sqlalchemy import select

from app.services.shap_service import (
    METHOD_PERMUTATION,
    compute_and_persist_shap_for_run,
)

logger = logging.getLogger(__name__)


def _fallback_feature_importances(model: Any, X, y=None) -> tuple[dict[str, float], str]:
    """Legacy test-only entry point retained for ``tests/v3/test_explain_fallback``.

    New code should call `shap_service.compute_shap_summary` instead — it
    already encapsulates TreeExplainer → KernelExplainer → permutation fallback.
    """
    import numpy as np

    if hasattr(model, "feature_importances_"):
        importances = np.asarray(model.feature_importances_, dtype=np.float64)
        return (
            {col: float(val) for col, val in zip(X.columns.tolist(), importances)},
            "model_feature_importances",
        )

    if hasattr(model, "coef_"):
        coef = np.asarray(model.coef_, dtype=np.float64)
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
            model, X, y, n_repeats=5, random_state=42, n_jobs=1,
        )
        importances = np.asarray(result.importances_mean, dtype=np.float64)
        return (
            {col: float(val) for col, val in zip(X.columns.tolist(), importances)},
            "permutation_importance",
        )

    raise ValueError(
        "SHAP is unavailable and the model does not expose fallback feature importance attributes"
    )


async def run_shap_explanation(run_id: str, platform_task_id: str) -> dict[str, Any]:
    """Compute + persist SHAP values for the model produced by `run_id`.

    Thin wrapper around ``shap_service.compute_and_persist_shap_for_run``; the
    ``platform_task_id`` parameter is kept for signature compatibility with
    the Celery / scheduler executor contract but isn't needed for computation.
    """
    return await compute_and_persist_shap_for_run(run_id)


async def get_shap_result(run_id: str) -> dict[str, Any] | None:
    """Fetch the stored SHAP result for a V3 run.

    Priority:
      1. Inline: ``run.metrics["shap_importances"]`` — always available after computation.
      2. MinIO presigned URL — only if S3 is configured and artifacts_uri is set.

    Returns None if no explanation has been computed yet.
    """
    from app.models.database import ExperimentRun, async_session_factory

    async with async_session_factory() as db:
        run_result = await db.execute(select(ExperimentRun).where(ExperimentRun.id == run_id))
        run = run_result.scalar_one_or_none()
        if run is None:
            return None

        metrics = run.metrics or {}
        shap_importances = metrics.get("shap_importances")
        method = metrics.get("shap_method", "shap")
        sample_size = metrics.get("shap_sample_size")
        base_value = metrics.get("shap_base_value")

        if shap_importances:
            return {
                "run_id": run_id,
                "feature_importances": shap_importances,
                "sample_size": sample_size,
                "base_value": base_value,
                "artifacts_uri": run.artifacts_uri,
                "explanation_method": method,
                "source": "inline",
            }

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


# ---------------------------------------------------------------------------
# Executor registration — V3 Phase 2
# ---------------------------------------------------------------------------

async def _explain_executor(domain_id: str, platform_task_id: str) -> dict:
    """Executor entrypoint that run_id-drives the full SHAP ladder."""
    result = await run_shap_explanation(domain_id, platform_task_id)
    importances = result.get("feature_importances") or {}
    metrics = {"feature_count": len(importances)}
    return {**result, "metrics": metrics}


from app.scheduler.executors import register_executor as _register_executor  # noqa: E402
_register_executor("explain", _explain_executor)


# Sentinel export so downstream modules can discover the permutation-tail label.
__all__ = [
    "run_shap_explanation",
    "get_shap_result",
    "_fallback_feature_importances",
    "METHOD_PERMUTATION",
]

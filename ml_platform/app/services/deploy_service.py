"""Model deployment service — LRU model cache and inference logic."""

from __future__ import annotations

import logging
from collections import OrderedDict
from datetime import datetime, timezone
from typing import Any

import joblib
from fastapi import HTTPException
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.database import InferenceJob, ModelDeployment, TrainingTask, Dataset
from app.services.prediction_service import load_dataframe, prepare_prediction_frame, prepare_training_frame
from app.utils.storage_paths import resolve_runtime_path

logger = logging.getLogger(__name__)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# LRU model cache
# ---------------------------------------------------------------------------

class _ModelCache:
    """Simple LRU cache for loaded sklearn models (keyed by deployment_id)."""

    def __init__(self, max_size: int = 10):
        self._cache: OrderedDict[str, Any] = OrderedDict()
        self.max_size = max_size

    def get(self, deployment_id: str, model_path: str) -> Any:
        if deployment_id in self._cache:
            self._cache.move_to_end(deployment_id)
            return self._cache[deployment_id]
        model = joblib.load(model_path)
        self._cache[deployment_id] = model
        self._cache.move_to_end(deployment_id)
        if len(self._cache) > self.max_size:
            self._cache.popitem(last=False)
        return model

    def evict(self, deployment_id: str) -> None:
        self._cache.pop(deployment_id, None)


_model_cache = _ModelCache(max_size=10)


# ---------------------------------------------------------------------------
# Service functions
# ---------------------------------------------------------------------------

async def create_deployment(
    task_id: str,
    name: str,
    db: AsyncSession,
    description: str | None = None,
    max_batch_size: int = 100,
) -> dict:
    # Verify task exists and is complete
    result = await db.execute(select(TrainingTask).where(TrainingTask.id == task_id))
    task = result.scalar_one_or_none()
    if task is None:
        raise HTTPException(status_code=404, detail="Training task not found")
    if task.status != "SUCCESS":
        raise HTTPException(status_code=400, detail=f"Task not completed (status={task.status})")
    if not task.model_path or not resolve_runtime_path(task.model_path).exists():
        raise HTTPException(status_code=400, detail="Model file not found on disk")

    deployment = ModelDeployment(
        task_id=task_id,
        name=name,
        description=description,
        max_batch_size=max_batch_size,
        status="active",
        request_count=0,
    )
    db.add(deployment)
    await db.flush()
    await db.refresh(deployment)

    base = "http://127.0.0.1:8000"
    endpoints = {
        "predict": f"{base}/inference/{deployment.id}/predict",
        "result": f"{base}/inference/{deployment.id}/result/{{job_id}}",
    }

    return {
        "deployment_id": deployment.id,
        "task_id": task_id,
        "name": deployment.name,
        "status": deployment.status,
        "request_count": deployment.request_count,
        "created_at": deployment.created_at,
        "endpoints": endpoints,
    }


async def list_deployments(db: AsyncSession, page: int = 1, page_size: int = 20) -> dict:
    count_result = await db.execute(select(func.count(ModelDeployment.id)))
    total: int = count_result.scalar_one()

    offset = (page - 1) * page_size
    result = await db.execute(
        select(ModelDeployment)
        .order_by(ModelDeployment.created_at.desc())
        .offset(offset)
        .limit(page_size)
    )
    deployments = result.scalars().all()

    base = "http://127.0.0.1:8000"
    items = [
        {
            "deployment_id": d.id,
            "task_id": d.task_id,
            "name": d.name,
            "status": d.status,
            "request_count": d.request_count,
            "created_at": d.created_at,
            "endpoints": {
                "predict": f"{base}/inference/{d.id}/predict",
                "result": f"{base}/inference/{d.id}/result/{{job_id}}",
            },
        }
        for d in deployments
    ]

    return {"deployments": items, "total": total, "page": page, "page_size": page_size}


async def delete_deployment(deployment_id: str, db: AsyncSession) -> None:
    result = await db.execute(
        select(ModelDeployment).where(ModelDeployment.id == deployment_id)
    )
    deployment = result.scalar_one_or_none()
    if deployment is None:
        raise HTTPException(status_code=404, detail="Deployment not found")
    _model_cache.evict(deployment_id)
    await db.delete(deployment)
    await db.flush()


async def update_deployment_status(
    deployment_id: str, status: str, db: AsyncSession
) -> None:
    result = await db.execute(
        select(ModelDeployment).where(ModelDeployment.id == deployment_id)
    )
    deployment = result.scalar_one_or_none()
    if deployment is None:
        raise HTTPException(status_code=404, detail="Deployment not found")
    deployment.status = status
    await db.flush()


async def run_inference(
    deployment_id: str,
    rows: list[dict],
    db: AsyncSession,
    include_probabilities: bool = True,
) -> dict:
    result = await db.execute(
        select(ModelDeployment).where(ModelDeployment.id == deployment_id)
    )
    deployment = result.scalar_one_or_none()
    if deployment is None:
        raise HTTPException(status_code=404, detail="Deployment not found")
    if deployment.status != "active":
        raise HTTPException(status_code=400, detail="Deployment is paused")

    # Load task and model
    task_result = await db.execute(
        select(TrainingTask).where(TrainingTask.id == deployment.task_id)
    )
    task = task_result.scalar_one_or_none()
    if task is None or not task.model_path:
        raise HTTPException(status_code=404, detail="Associated task or model not found")

    model = _model_cache.get(deployment_id, str(resolve_runtime_path(task.model_path)))

    # Load training data for prepare_prediction_frame
    ds_result = await db.execute(select(Dataset).where(Dataset.id == task.dataset_id))
    dataset = ds_result.scalar_one_or_none()
    if dataset is None:
        raise HTTPException(status_code=404, detail="Dataset not found")

    training_df = load_dataframe(dataset.file_path)
    X_pred = prepare_prediction_frame(training_df, rows, task.target_column)
    predictions_raw = model.predict(X_pred.values).tolist()

    # Decode classification labels if applicable
    _, _, _, target_encoder = prepare_training_frame(training_df, task.target_column)
    if target_encoder is not None:
        try:
            predictions = target_encoder.inverse_transform(
                [int(p) for p in predictions_raw]
            ).tolist()
        except Exception:
            predictions = predictions_raw
    else:
        predictions = predictions_raw

    probabilities = None
    if include_probabilities and hasattr(model, "predict_proba"):
        try:
            probabilities = model.predict_proba(X_pred).tolist()
        except Exception:
            pass

    # Update request count
    deployment.request_count = (deployment.request_count or 0) + 1

    job = InferenceJob(
        deployment_id=deployment_id,
        status="completed",
        input_rows=len(rows),
        predictions=predictions,
        probabilities=probabilities,
        completed_at=_utcnow(),
    )
    db.add(job)
    await db.flush()

    return {
        "job_id": job.id,
        "deployment_id": deployment_id,
        "status": "completed",
        "predictions": predictions,
        "probabilities": probabilities,
        "input_rows": len(rows),
        "error_message": None,
        "created_at": job.created_at,
        "completed_at": job.completed_at,
    }


async def get_inference_result(
    deployment_id: str, job_id: str, db: AsyncSession
) -> dict:
    result = await db.execute(
        select(InferenceJob).where(
            InferenceJob.id == job_id,
            InferenceJob.deployment_id == deployment_id,
        )
    )
    job = result.scalar_one_or_none()
    if job is None:
        raise HTTPException(status_code=404, detail="Inference job not found")

    return {
        "job_id": job.id,
        "deployment_id": deployment_id,
        "status": job.status,
        "predictions": job.predictions,
        "probabilities": job.probabilities,
        "input_rows": job.input_rows,
        "error_message": job.error_message,
        "created_at": job.created_at,
        "completed_at": job.completed_at,
    }

"""Model deployment service — LRU model cache and inference logic."""

from __future__ import annotations

import logging
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
from fastapi import HTTPException
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.database import InferenceJob, ModelDeployment, TrainingTask, Dataset
from app.services.object_storage import restore_dataset_file, restore_model_bundle
from app.services.prediction_service import load_dataframe, predict_with_model

logger = logging.getLogger(__name__)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _deployment_endpoints(deployment_id: str) -> dict[str, str]:
    """Return gateway-safe paths without assuming a host or backend port."""
    return {
        "predict": f"/inference/{deployment_id}/predict",
        "result": f"/inference/{deployment_id}/result/{{job_id}}",
    }


# ---------------------------------------------------------------------------
# LRU model cache
# ---------------------------------------------------------------------------

class _ModelCache:
    """Simple LRU cache for loaded sklearn models (keyed by deployment_id)."""

    def __init__(self, max_size: int = 10):
        self._cache: OrderedDict[str, Any] = OrderedDict()
        self.max_size = max_size

    def get(self, deployment_id: str, model_path: str | Path) -> Any:
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
    owner_username: str | None = None,
) -> dict:
    # Verify task exists and is complete
    task_stmt = select(TrainingTask).where(TrainingTask.id == task_id)
    if owner_username:
        task_stmt = task_stmt.where(TrainingTask.owner_username == owner_username)
    result = await db.execute(task_stmt)
    task = result.scalar_one_or_none()
    if task is None:
        raise HTTPException(status_code=404, detail="Training task not found")
    if task.status != "SUCCESS":
        raise HTTPException(status_code=400, detail=f"Task not completed (status={task.status})")
    # Read-through: a rebuilt container has empty local storage — pull the
    # model (and DL sidecars) back from MinIO before declaring it missing.
    if not task.model_path or restore_model_bundle(task.model_path) is None:
        raise HTTPException(status_code=400, detail="模型文件缺失，且对象存储中无副本")

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

    return {
        "deployment_id": deployment.id,
        "task_id": task_id,
        "name": deployment.name,
        "status": deployment.status,
        "request_count": deployment.request_count,
        "created_at": deployment.created_at,
        "endpoints": _deployment_endpoints(deployment.id),
    }


async def list_deployments(
    db: AsyncSession,
    page: int = 1,
    page_size: int = 20,
    owner_username: str | None = None,
) -> dict:
    count_stmt = select(func.count(ModelDeployment.id))
    stmt = select(ModelDeployment)
    if owner_username:
        count_stmt = (
            count_stmt
            .join(TrainingTask, TrainingTask.id == ModelDeployment.task_id)
            .where(TrainingTask.owner_username == owner_username)
        )
        stmt = (
            stmt
            .join(TrainingTask, TrainingTask.id == ModelDeployment.task_id)
            .where(TrainingTask.owner_username == owner_username)
        )
    count_result = await db.execute(count_stmt)
    total: int = count_result.scalar_one()

    offset = (page - 1) * page_size
    result = await db.execute(
        stmt
        .order_by(ModelDeployment.created_at.desc())
        .offset(offset)
        .limit(page_size)
    )
    deployments = result.scalars().all()

    items = [
        {
            "deployment_id": d.id,
            "task_id": d.task_id,
            "name": d.name,
            "status": d.status,
            "request_count": d.request_count,
            "created_at": d.created_at,
            "endpoints": _deployment_endpoints(d.id),
        }
        for d in deployments
    ]

    return {"deployments": items, "total": total, "page": page, "page_size": page_size}


async def delete_deployment(
    deployment_id: str,
    db: AsyncSession,
    owner_username: str | None = None,
) -> None:
    result = await db.execute(
        select(ModelDeployment).where(ModelDeployment.id == deployment_id)
    )
    deployment = result.scalar_one_or_none()
    if deployment is None:
        raise HTTPException(status_code=404, detail="Deployment not found")
    task_stmt = select(TrainingTask).where(TrainingTask.id == deployment.task_id)
    if owner_username:
        task_stmt = task_stmt.where(TrainingTask.owner_username == owner_username)
    task = (await db.execute(task_stmt)).scalar_one_or_none()
    if task is None:
        raise HTTPException(status_code=404, detail="Deployment not found")
    _model_cache.evict(deployment_id)
    await db.delete(deployment)
    await db.flush()


async def update_deployment_status(
    deployment_id: str,
    status: str,
    db: AsyncSession,
    owner_username: str | None = None,
) -> None:
    result = await db.execute(
        select(ModelDeployment).where(ModelDeployment.id == deployment_id)
    )
    deployment = result.scalar_one_or_none()
    if deployment is None:
        raise HTTPException(status_code=404, detail="Deployment not found")
    task_stmt = select(TrainingTask).where(TrainingTask.id == deployment.task_id)
    if owner_username:
        task_stmt = task_stmt.where(TrainingTask.owner_username == owner_username)
    task = (await db.execute(task_stmt)).scalar_one_or_none()
    if task is None:
        raise HTTPException(status_code=404, detail="Deployment not found")
    deployment.status = status
    await db.flush()


async def run_inference(
    deployment_id: str,
    rows: list[dict],
    db: AsyncSession,
    include_probabilities: bool = True,
    owner_username: str | None = None,
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
    task_stmt = select(TrainingTask).where(TrainingTask.id == deployment.task_id)
    if owner_username:
        task_stmt = task_stmt.where(TrainingTask.owner_username == owner_username)
    task_result = await db.execute(task_stmt)
    task = task_result.scalar_one_or_none()
    if task is None or not task.model_path:
        raise HTTPException(status_code=404, detail="Associated task or model not found")

    model_path = restore_model_bundle(task.model_path)
    if model_path is None:
        raise HTTPException(
            status_code=404,
            detail="模型文件不存在，且对象存储中无可恢复副本",
        )
    model = _model_cache.get(deployment_id, model_path)

    # Legacy estimators still need their training dataframe; new artifacts do not refit it.
    ds_result = await db.execute(select(Dataset).where(Dataset.id == task.dataset_id))
    dataset = ds_result.scalar_one_or_none()
    if dataset is None:
        raise HTTPException(status_code=404, detail="Dataset not found")

    dataset_path = restore_dataset_file(dataset.id, dataset.file_path)
    if dataset_path is None:
        raise HTTPException(
            status_code=404,
            detail="数据集文件不存在，且对象存储中无可恢复副本",
        )
    training_df = load_dataframe(dataset_path)
    prediction = predict_with_model(
        model,
        training_df,
        rows,
        task.target_column,
        include_probabilities=include_probabilities,
    )
    predictions = prediction["predictions"]
    probabilities = prediction["probabilities"]

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
    deployment_id: str,
    job_id: str,
    db: AsyncSession,
    owner_username: str | None = None,
) -> dict:
    if owner_username:
        deployment_result = await db.execute(
            select(ModelDeployment)
            .join(TrainingTask, TrainingTask.id == ModelDeployment.task_id)
            .where(
                ModelDeployment.id == deployment_id,
                TrainingTask.owner_username == owner_username,
            )
        )
        if deployment_result.scalar_one_or_none() is None:
            raise HTTPException(status_code=404, detail="Inference job not found")
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

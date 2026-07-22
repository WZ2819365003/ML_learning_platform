"""Batch prediction — CSV in, CSV out, executed asynchronously (M3-2).

The synchronous path (`deploy_service.run_inference`) already accepts a list of
rows, so "batch" was half-present. What it could not do is the thing users
actually have: a file with more rows than fit in an HTTP request, stored
somewhere they can download afterwards. Two properties of that path make it
unsuitable for scale rather than merely slow:

* results land in ``InferenceJob.predictions`` (a JSON column), which cannot
  hold a 100k-row result and cannot be streamed back;
* ``max_batch_size`` is stored on the deployment but never enforced, so a large
  request blocks the event loop for as long as it takes.

This module therefore keeps the same ``InferenceJob`` record but moves the data
to files, and runs through the normal scheduler so it inherits M2c's claim /
terminal write-back / stalled-task recovery instead of inventing its own.

Rows are processed in chunks. The input is never fully materialised, so memory
stays flat regardless of file size — the reason a 2 GB upload does not become a
2 GB dataframe.
"""
from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.models.database import (
    Dataset,
    InferenceJob,
    ModelDeployment,
    TrainingTask,
    async_session_factory,
)
from app.services.object_storage import (
    restore_dataset_file,
    restore_file,
    restore_model_bundle,
    upload_file,
)

logger = logging.getLogger(__name__)

CHUNK_SIZE = 5_000
_PREDICTION_COLUMN = "prediction"


def _input_key(job_id: str) -> str:
    return f"predictions/{job_id}-input.csv"


def _result_key(job_id: str) -> str:
    return f"predictions/{job_id}-result.csv"


def restore_batch_file(local_path: str | Path, job_id: str, *, kind: str) -> Path | None:
    """Read-through for batch prediction files.

    Uploading without a restore path only protects against *deleting* the file
    by hand; it does nothing for the case that matters — the storage volume is
    gone and the row still points at a path that no longer exists.
    """
    key = _input_key(job_id) if kind == "input" else _result_key(job_id)
    return restore_file(Path(local_path), [key])


def _predictions_dir() -> Path:
    path = get_settings().project_root / "storage" / "predictions"
    path.mkdir(parents=True, exist_ok=True)
    return path


async def create_batch_job(
    db: AsyncSession,
    *,
    deployment_id: str,
    filename: str,
    content: bytes,
) -> dict[str, Any]:
    """Persist the upload, create a PENDING job, and dispatch it.

    Returns immediately — the caller polls the job for progress. The row count
    is deliberately *not* computed here: doing so would read the whole file in
    the request handler, which is exactly the blocking behaviour this path
    exists to avoid.
    """
    deployment = (
        await db.execute(
            select(ModelDeployment).where(ModelDeployment.id == deployment_id)
        )
    ).scalar_one_or_none()
    if deployment is None:
        raise HTTPException(status_code=404, detail="部署不存在")
    if deployment.status != "active":
        raise HTTPException(status_code=400, detail="部署已暂停，无法提交预测任务")
    if not filename.lower().endswith(".csv"):
        raise HTTPException(status_code=422, detail="批量预测目前仅支持 CSV 文件")
    if not content.strip():
        raise HTTPException(status_code=422, detail="上传的文件为空")

    job = InferenceJob(deployment_id=deployment_id, status="pending")
    db.add(job)
    await db.flush()

    input_path = _predictions_dir() / f"{job.id}-input.csv"
    input_path.write_bytes(content)
    job.input_path = str(input_path)

    # Upload the input too, not just the result. Without it the volume is the
    # only copy: a rebuilt container cannot re-run the job (the executor hard
    # -fails on a missing input), and there is no record of what was actually
    # scored. Best-effort like every other write-through — the local file is
    # the source of truth for this request.
    try:
        upload_file(input_path, _input_key(job.id))
    except Exception as exc:  # noqa: BLE001
        logger.warning("Batch prediction input upload failed for %s: %s", job.id, exc)

    from app.scheduler.task_runner import register_domain_task

    task = await register_domain_task(db, kind="predict", payload_ref=f"predict:{job.id}")
    await db.commit()

    from app.scheduler.scheduler import get_scheduler

    await get_scheduler("predict").submit(task.id)

    return {
        "job_id": job.id,
        "platform_task_id": task.id,
        "deployment_id": deployment_id,
        "status": "pending",
    }


async def _load_predictor(db: AsyncSession, deployment_id: str):
    """Resolve model + training frame once per job, not once per chunk."""
    deployment = (
        await db.execute(
            select(ModelDeployment).where(ModelDeployment.id == deployment_id)
        )
    ).scalar_one_or_none()
    if deployment is None:
        raise HTTPException(status_code=404, detail="部署不存在")

    task = (
        await db.execute(
            select(TrainingTask).where(TrainingTask.id == deployment.task_id)
        )
    ).scalar_one_or_none()
    if task is None or not task.model_path:
        raise HTTPException(status_code=404, detail="关联的训练任务或模型不存在")

    model_path = restore_model_bundle(task.model_path)
    if model_path is None:
        raise HTTPException(status_code=404, detail="模型文件不存在，且对象存储中无可恢复副本")

    dataset = (
        await db.execute(select(Dataset).where(Dataset.id == task.dataset_id))
    ).scalar_one_or_none()
    if dataset is None:
        raise HTTPException(status_code=404, detail="数据集不存在")
    dataset_path = restore_dataset_file(dataset.id, dataset.file_path)
    if dataset_path is None:
        raise HTTPException(status_code=404, detail="数据集文件不存在，且对象存储中无可恢复副本")

    from app.services.deploy_service import _model_cache
    from app.services.prediction_service import load_dataframe

    model = _model_cache.get(deployment_id, model_path)
    training_df = load_dataframe(dataset_path)
    return model, training_df, task.target_column


async def run_batch_prediction(domain_task_id: str, platform_task_id: str) -> dict[str, Any]:
    """Scheduler executor for ``kind='predict'``.

    Streams the input in chunks and appends to the result CSV as it goes, so a
    job that dies partway leaves a partial file and an accurate
    ``processed_rows`` rather than nothing at all.
    """
    from app.services.prediction_service import predict_with_model

    async with async_session_factory() as db:
        job = (
            await db.execute(select(InferenceJob).where(InferenceJob.id == domain_task_id))
        ).scalar_one_or_none()
        if job is None:
            raise ValueError(f"InferenceJob {domain_task_id} not found")
        deployment_id = job.deployment_id
        input_path = Path(job.input_path or "")

        job.status = "running"
        await db.commit()

    if not input_path.exists():
        # The volume may have been rebuilt since submission. Try object storage
        # before giving up — otherwise stalled-task recovery can never succeed.
        if restore_batch_file(input_path, domain_task_id, kind="input") is None:
            raise FileNotFoundError(
                f"批量预测输入文件缺失，且对象存储中无可恢复副本: {input_path}"
            )

    async with async_session_factory() as db:
        model, training_df, target_column = await _load_predictor(db, deployment_id)

    result_path = _predictions_dir() / f"{domain_task_id}-result.csv"
    processed = 0
    header_written = False

    with result_path.open("w", newline="", encoding="utf-8-sig") as out:
        writer: csv.DictWriter | None = None
        for chunk in pd.read_csv(input_path, chunksize=CHUNK_SIZE):
            rows = chunk.to_dict(orient="records")
            if not rows:
                continue
            prediction = predict_with_model(
                model,
                training_df,
                rows,
                target_column,
                # Probabilities would double the file width and are not part
                # of the CSV contract; the synchronous endpoint still offers them.
                include_probabilities=False,
            )
            preds = prediction["predictions"]
            for row, pred in zip(rows, preds):
                row[_PREDICTION_COLUMN] = pred
            if not header_written:
                writer = csv.DictWriter(out, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                header_written = True
            writer.writerows(rows)
            processed += len(rows)

            async with async_session_factory() as db:
                job = (
                    await db.execute(
                        select(InferenceJob).where(InferenceJob.id == domain_task_id)
                    )
                ).scalar_one()
                job.processed_rows = processed
                await db.commit()

    try:
        upload_file(result_path, _result_key(domain_task_id))
    except Exception as exc:  # noqa: BLE001 — object storage is a copy, not the source
        logger.warning("Batch prediction result upload failed for %s: %s", domain_task_id, exc)

    async with async_session_factory() as db:
        job = (
            await db.execute(select(InferenceJob).where(InferenceJob.id == domain_task_id))
        ).scalar_one()
        job.status = "completed"
        job.input_rows = processed
        job.processed_rows = processed
        job.result_path = str(result_path)
        from app.services.deploy_service import _utcnow

        job.completed_at = _utcnow()
        await db.commit()

    return {"metrics": {"processed_rows": processed}, "result_path": str(result_path)}


async def get_batch_job(db: AsyncSession, deployment_id: str, job_id: str) -> InferenceJob:
    job = (
        await db.execute(
            select(InferenceJob).where(
                InferenceJob.id == job_id,
                InferenceJob.deployment_id == deployment_id,
            )
        )
    ).scalar_one_or_none()
    if job is None:
        raise HTTPException(status_code=404, detail="预测任务不存在")
    return job


def serialize_batch_job(job: InferenceJob) -> dict[str, Any]:
    return {
        "job_id": job.id,
        "deployment_id": job.deployment_id,
        "status": job.status,
        "input_rows": job.input_rows,
        "processed_rows": job.processed_rows,
        "has_result": bool(job.result_path),
        "error_message": job.error_message,
        "created_at": job.created_at.isoformat() if job.created_at else None,
        "completed_at": job.completed_at.isoformat() if job.completed_at else None,
    }


from app.scheduler.executors import register_executor as _register_executor  # noqa: E402

_register_executor("predict", run_batch_prediction)

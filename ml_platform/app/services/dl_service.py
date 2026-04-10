"""Deep-learning training service.

Orchestrates DLTrainingTask lifecycle:
  start_dl_training  → create DB row, launch _execute_dl_training coroutine
  _execute_dl_training → async: set RUNNING, submit sync work to ThreadPoolExecutor
  _run_dl_sync         → sync: actual PyTorch training; publishes epoch events via EventBus
  stop_dl_training    → cancel async task
  get_dl_status / list_dl_tasks → DB queries
"""
from __future__ import annotations

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from fastapi import HTTPException
from sklearn.model_selection import train_test_split
from sqlalchemy import func, select

from app.config import get_settings
from app.core.dl_registry import get_dl_trainer
from app.core.logger import event_bus
from app.models.database import (
    AsyncSession, Dataset, DLTrainingTask, async_session_factory,
)
from app.services.prediction_service import load_dataframe, prepare_training_frame

logger = logging.getLogger(__name__)

_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="dl-train")
_running_tasks: dict[str, asyncio.Task] = {}

# ---------------------------------------------------------------------------
# Task-type auto-detection
# ---------------------------------------------------------------------------

def _detect_task_type(y: np.ndarray, requested: str) -> str:
    if requested in ("classification", "regression"):
        return requested
    # heuristic: ≤20 unique int-valued classes → classification
    unique = np.unique(y)
    if len(unique) <= 20 and np.issubdtype(y.dtype, np.integer):
        return "classification"
    return "regression"


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

def _prepare_dl_data(file_path: str, target_column: str, test_size: float, task_type: str):
    df = load_dataframe(file_path)
    X, y, _, _ = prepare_training_frame(df, target_column)
    stratify = y.values if task_type == "classification" else None
    X_train, X_val, y_train, y_val = train_test_split(
        X.values, y.values, test_size=test_size, random_state=42, stratify=stratify
    )
    return X_train, X_val, y_train, y_val


# ---------------------------------------------------------------------------
# Sync training function (runs in ThreadPoolExecutor)
# ---------------------------------------------------------------------------

def _run_dl_sync(
    task_id:      str,
    file_path:    str,
    target_column: str,
    model_type:   str,
    task_type:    str,
    arch_config:  dict,
    opt_config:   dict,
    train_config: dict,
    model_save_dir: str,
    loop: asyncio.AbstractEventLoop,
) -> dict:
    epochs = int(train_config.get("epochs", 50))
    test_size = float(train_config.get("test_size", 0.2))

    logger.info("[DL %s] Starting %s | epochs=%d", task_id, model_type, epochs)

    X_train, X_val, y_train, y_val = _prepare_dl_data(
        file_path, target_column, test_size, task_type)

    # Resolve task_type from data if "auto"
    task_type = _detect_task_type(y_train, task_type)

    def epoch_callback(epoch: int, data: dict):
        """Called from training thread → safely publish to asyncio EventBus."""
        progress = round(epoch / epochs * 100, 1)
        msg = {
            "type":       "epoch",
            "epoch":      epoch,
            "total":      epochs,
            "progress":   progress,
            **data,
        }
        loop.call_soon_threadsafe(event_bus.publish, f"dl:{task_id}", msg)
        # Also update task progress in DB synchronously is not possible here;
        # the async _execute_dl_training polls after training completes.

    trainer = get_dl_trainer(model_type)
    result = trainer.train(
        X_train, y_train, X_val, y_val,
        arch_config=arch_config,
        opt_config=opt_config,
        train_config=train_config,
        task_type=task_type,
        epoch_callback=epoch_callback,
    )

    # Save model
    save_dir = Path(model_save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    model_path = str(save_dir / f"dl_{task_id}.pt")
    trainer.save(model_path)
    logger.info("[DL %s] Saved → %s", task_id, model_path)

    return {"result_metrics": result, "model_path": model_path, "task_type": task_type}


# ---------------------------------------------------------------------------
# Async orchestration
# ---------------------------------------------------------------------------

async def _execute_dl_training(
    task_id:      str,
    file_path:    str,
    target_column: str,
    model_type:   str,
    task_type:    str,
    arch_config:  dict,
    opt_config:   dict,
    train_config: dict,
    model_save_dir: str,
):
    # Mark RUNNING
    async with async_session_factory() as db:
        result = await db.execute(select(DLTrainingTask).where(DLTrainingTask.id == task_id))
        task = result.scalar_one_or_none()
        if task:
            task.status = "RUNNING"
            task.started_at = datetime.now(timezone.utc)
            task.total_epochs = int(train_config.get("epochs", 50))
            await db.commit()

    loop = asyncio.get_event_loop()
    try:
        training_result = await loop.run_in_executor(
            _executor,
            _run_dl_sync,
            task_id, file_path, target_column, model_type, task_type,
            arch_config, opt_config, train_config, model_save_dir, loop,
        )

        # Strip history from stored metrics (keep it lightweight)
        stored_metrics = {k: v for k, v in training_result["result_metrics"].items()
                          if k != "history"}

        async with async_session_factory() as db:
            result = await db.execute(select(DLTrainingTask).where(DLTrainingTask.id == task_id))
            task = result.scalar_one_or_none()
            if task:
                task.status = "SUCCESS"
                task.progress = 100.0
                task.task_type = training_result["task_type"]
                task.result_metrics = stored_metrics
                task.model_path = training_result["model_path"]
                task.finished_at = datetime.now(timezone.utc)
                await db.commit()

        # Notify WebSocket clients that training is done
        loop.call_soon_threadsafe(event_bus.publish, f"dl:{task_id}", {
            "type": "done", "status": "SUCCESS", "metrics": stored_metrics,
        })

    except Exception as exc:
        logger.error("[DL %s] Failed: %s", task_id, exc, exc_info=True)
        async with async_session_factory() as db:
            result = await db.execute(select(DLTrainingTask).where(DLTrainingTask.id == task_id))
            task = result.scalar_one_or_none()
            if task:
                task.status = "FAILED"
                task.error_message = str(exc)
                task.finished_at = datetime.now(timezone.utc)
                await db.commit()
        event_bus.publish(f"dl:{task_id}", {"type": "done", "status": "FAILED", "error": str(exc)})
    finally:
        _running_tasks.pop(task_id, None)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def start_dl_training(request_data: dict, db: AsyncSession) -> DLTrainingTask:
    settings = get_settings()

    # Validate dataset
    dataset_id = request_data["dataset_id"]
    res = await db.execute(select(Dataset).where(Dataset.id == dataset_id))
    dataset = res.scalar_one_or_none()
    if dataset is None:
        raise HTTPException(status_code=404, detail="Dataset not found")

    # Validate model type
    from app.core.dl_registry import get_dl_trainer_registry
    available = list(get_dl_trainer_registry().keys())
    if request_data["model_type"] not in available:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown DL model '{request_data['model_type']}'. Available: {available}",
        )

    import uuid as _uuid_mod
    short_id = str(_uuid_mod.uuid4())[:8]
    task = DLTrainingTask(
        dataset_id=dataset_id,
        name=f"{request_data['model_type']}_{short_id}",
        target_column=request_data["target_column"],
        model_type=request_data["model_type"],
        task_type=request_data.get("task_type", "auto"),
        arch_config=request_data.get("arch_config", {}),
        opt_config=request_data.get("opt_config", {}),
        train_config=request_data.get("train_config", {}),
        status="PENDING",
    )
    db.add(task)
    await db.flush()
    await db.refresh(task)

    bg = asyncio.create_task(_execute_dl_training(
        task_id=task.id,
        file_path=dataset.file_path,
        target_column=request_data["target_column"],
        model_type=request_data["model_type"],
        task_type=request_data.get("task_type", "auto"),
        arch_config=request_data.get("arch_config", {}),
        opt_config=request_data.get("opt_config", {}),
        train_config=request_data.get("train_config", {}),
        model_save_dir=str(settings.storage_models),
    ))
    _running_tasks[task.id] = bg
    return task


async def stop_dl_training(task_id: str, db: AsyncSession) -> DLTrainingTask:
    res = await db.execute(select(DLTrainingTask).where(DLTrainingTask.id == task_id))
    task = res.scalar_one_or_none()
    if task is None:
        raise HTTPException(status_code=404, detail="DL task not found")
    if task.status not in ("PENDING", "RUNNING"):
        raise HTTPException(status_code=400, detail=f"Cannot stop task with status '{task.status}'")
    if task_id in _running_tasks:
        _running_tasks[task_id].cancel()
        del _running_tasks[task_id]
    task.status = "FAILED"
    task.error_message = "Manually stopped"
    task.finished_at = datetime.now(timezone.utc)
    await db.flush()
    return task


async def get_dl_status(task_id: str, db: AsyncSession) -> DLTrainingTask:
    res = await db.execute(select(DLTrainingTask).where(DLTrainingTask.id == task_id))
    task = res.scalar_one_or_none()
    if task is None:
        raise HTTPException(status_code=404, detail="DL task not found")
    return task


async def list_dl_tasks(
    db: AsyncSession, page: int = 1, page_size: int = 20, status_filter: str | None = None
) -> dict:
    stmt = select(DLTrainingTask)
    count_stmt = select(func.count(DLTrainingTask.id))
    if status_filter:
        stmt = stmt.where(DLTrainingTask.status == status_filter)
        count_stmt = count_stmt.where(DLTrainingTask.status == status_filter)

    total = (await db.execute(count_stmt)).scalar_one()
    offset = (page - 1) * page_size
    stmt = stmt.order_by(DLTrainingTask.created_at.desc()).offset(offset).limit(page_size)
    tasks = (await db.execute(stmt)).scalars().all()
    return {"items": tasks, "total": total, "page": page, "page_size": page_size}


async def rename_dl_task(task_id: str, name: str, db: AsyncSession) -> DLTrainingTask:
    """Rename a DL training task."""
    res = await db.execute(select(DLTrainingTask).where(DLTrainingTask.id == task_id))
    task = res.scalar_one_or_none()
    if task is None:
        raise HTTPException(status_code=404, detail="DL task not found")
    task.name = name
    await db.flush()
    return task


async def delete_dl_task(task_id: str, db: AsyncSession) -> None:
    """Delete a DL training task (not allowed while RUNNING)."""
    res = await db.execute(select(DLTrainingTask).where(DLTrainingTask.id == task_id))
    task = res.scalar_one_or_none()
    if task is None:
        raise HTTPException(status_code=404, detail="DL task not found")
    if task.status == "RUNNING":
        raise HTTPException(status_code=422, detail="Cannot delete a running task. Stop it first.")
    await db.delete(task)
    await db.flush()


async def list_dl_trained_models(
    db: AsyncSession, page: int = 1, page_size: int = 20
) -> dict:
    """Return paginated list of successfully completed DL tasks (i.e. trained models)."""
    stmt = select(DLTrainingTask).where(DLTrainingTask.status == "SUCCESS")
    count_stmt = select(func.count(DLTrainingTask.id)).where(DLTrainingTask.status == "SUCCESS")
    total = (await db.execute(count_stmt)).scalar_one()
    offset = (page - 1) * page_size
    stmt = stmt.order_by(DLTrainingTask.created_at.desc()).offset(offset).limit(page_size)
    tasks = (await db.execute(stmt)).scalars().all()
    return {"items": tasks, "total": total, "page": page, "page_size": page_size}

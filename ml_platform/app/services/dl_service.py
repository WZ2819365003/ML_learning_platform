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
import pandas as pd
from fastapi import HTTPException
from sklearn.model_selection import train_test_split
from sqlalchemy import func, select

from app.config import get_settings
from app.core.dl_registry import get_dl_trainer
from app.core.logger import TrainingLogger, event_bus
from app.models.database import (
    AsyncSession,
    Dataset,
    DLModelDeployment,
    DLTrainingEpoch,
    DLTrainingLog,
    DLTrainingTask,
    async_session_factory,
)
from app.services.prediction_service import load_dataframe, prepare_prediction_frame, prepare_training_frame

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


def _format_dl_log_number(value: float) -> str:
    return f"{float(value):.6f}"


def _select_dl_log_metrics(metrics: dict, keys: list[str]) -> dict[str, str]:
    selected: dict[str, str] = {}
    for key in keys:
        value = metrics.get(key)
        if isinstance(value, (int, float)):
            selected[key] = _format_dl_log_number(value)
    return selected


def _build_dl_epoch_log_entry(epoch: int, total_epochs: int, metrics: dict) -> tuple[str, dict[str, str]]:
    priority = [
        "train_loss",
        "val_loss",
        "val_acc",
        "val_f1_macro",
        "val_auc_roc",
        "val_rmse",
        "val_mae",
        "val_r2",
    ]
    return f"Epoch {epoch}/{total_epochs} 指标更新", _select_dl_log_metrics(metrics, priority)


def _build_dl_completion_log_entry(metrics: dict) -> tuple[str, dict[str, str]]:
    priority = [
        "best_val_loss",
        "val_loss",
        "val_acc",
        "val_f1_macro",
        "val_auc_roc",
        "val_precision",
        "val_recall",
        "val_rmse",
        "val_mae",
        "val_r2",
        "val_mape",
    ]
    return "训练完成，最终验证指标", _select_dl_log_metrics(metrics, priority)


async def _store_dl_epoch_progress(task_id: str, epoch: int, progress: float) -> None:
    """Persist the latest DL epoch so the REST status endpoint stays current."""
    async with async_session_factory() as db:
        result = await db.execute(select(DLTrainingTask).where(DLTrainingTask.id == task_id))
        task = result.scalar_one_or_none()
        if task is None:
            return
        task.current_epoch = epoch
        task.progress = progress
        await db.commit()


async def _store_dl_epoch_record(task_id: str, epoch: int, total_epochs: int, metrics: dict) -> None:
    async with async_session_factory() as db:
        db.add(
            DLTrainingEpoch(
                task_id=task_id,
                epoch=epoch,
                total_epochs=total_epochs,
                train_loss=metrics.get("train_loss"),
                val_loss=metrics.get("val_loss"),
                val_acc=metrics.get("val_acc"),
                val_f1_macro=metrics.get("val_f1_macro"),
                val_rmse=metrics.get("val_rmse"),
                val_mae=metrics.get("val_mae"),
                val_r2=metrics.get("val_r2"),
                lr=metrics.get("lr"),
            )
        )
        await db.commit()


async def _store_dl_log_record(
    task_id: str,
    level: str,
    message: str,
    extra: dict | None = None,
) -> None:
    async with async_session_factory() as db:
        db.add(
            DLTrainingLog(
                task_id=task_id,
                level=level,
                message=message,
                extra=extra or None,
            )
        )
        await db.commit()


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
    epochs    = int(train_config.get("epochs", 50))
    test_size = float(train_config.get("test_size", 0.2))
    best_logged_val_loss: float | None = None

    task_logger = TrainingLogger(task_id, model_type=model_type)

    def emit_log(level: str, message: str, **extra) -> None:
        task_logger.log(level, message, **extra)
        asyncio.run_coroutine_threadsafe(
            _store_dl_log_record(
                task_id=task_id,
                level=level,
                message=message,
                extra=extra or None,
            ),
            loop,
        )

    emit_log("INFO", "深度学习训练启动", model=model_type, task=task_type, epochs=str(epochs))

    logger.info("[DL %s] Starting %s | epochs=%d", task_id, model_type, epochs)

    X_train, X_val, y_train, y_val = _prepare_dl_data(
        file_path, target_column, test_size, task_type)

    # Infer feature column names (best-effort from DataFrame)
    try:
        df = load_dataframe(file_path)
        feature_columns = [c for c in df.columns if c != target_column]
    except Exception:
        feature_columns = []

    # Resolve task_type from data if "auto"
    task_type = _detect_task_type(y_train, task_type)
    emit_log(
        "INFO",
        f"任务类型确认: {task_type}",
        train_samples=str(len(X_train)),
        val_samples=str(len(X_val)),
        feature_count=str(X_train.shape[1]),
    )

    def epoch_callback(epoch: int, data: dict):
        """Called from training thread → safely publish to asyncio EventBus."""
        nonlocal best_logged_val_loss
        progress = round(epoch / epochs * 100, 1)
        msg = {
            "type":     "epoch",
            "epoch":    epoch,
            "total":    epochs,
            "progress": progress,
            **data,
        }
        loop.call_soon_threadsafe(event_bus.publish, f"dl:{task_id}", msg)
        # Persist epoch progress (current_epoch, progress) on the task row
        asyncio.run_coroutine_threadsafe(
            _store_dl_epoch_progress(task_id=task_id, epoch=epoch, progress=progress),
            loop,
        )
        # Persist full epoch record (every epoch) for page-refresh recovery
        asyncio.run_coroutine_threadsafe(
            _store_dl_epoch_record(task_id=task_id, epoch=epoch, total_epochs=epochs, metrics=data),
            loop,
        )

        # Track best val_loss for log annotation
        current_val_loss = data.get("val_loss")
        improved = (
            isinstance(current_val_loss, (int, float))
            and (best_logged_val_loss is None or float(current_val_loss) < best_logged_val_loss - 1e-7)
        )
        if improved:
            best_logged_val_loss = float(current_val_loss)

        # Log EVERY epoch (user requirement: 1 epoch per log line)
        log_message, scalar_data = _build_dl_epoch_log_entry(epoch=epoch, total_epochs=epochs, metrics=data)
        if improved and best_logged_val_loss is not None:
            log_message = f"{log_message}（当前最优）"
            scalar_data["best_val_loss"] = _format_dl_log_number(best_logged_val_loss)
        emit_log("INFO", log_message, **scalar_data)

    trainer = get_dl_trainer(model_type)
    result = trainer.train(
        X_train, y_train, X_val, y_val,
        arch_config=arch_config,
        opt_config=opt_config,
        train_config=train_config,
        task_type=task_type,
        epoch_callback=epoch_callback,
    )

    # Save model (with arch metadata for future inference)
    save_dir = Path(model_save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    model_path = str(save_dir / f"dl_{task_id}.pt")
    trainer.save(
        model_path,
        arch_config=arch_config,
        input_dim=X_train.shape[1],
        task_type=task_type,
        feature_columns=feature_columns,
    )
    logger.info("[DL %s] Saved → %s", task_id, model_path)

    # Log final metrics
    final_message, scalar_metrics = _build_dl_completion_log_entry(result)
    emit_log("INFO", final_message, **scalar_metrics)

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

        # Store full metrics including training history (needed for result charts)
        stored_metrics = training_result["result_metrics"]

        async with async_session_factory() as db:
            result = await db.execute(select(DLTrainingTask).where(DLTrainingTask.id == task_id))
            task = result.scalar_one_or_none()
            if task:
                task.status = "SUCCESS"
                task.progress = 100.0
                task.current_epoch = int(stored_metrics.get("final_epoch") or task.total_epochs or 0)
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
        await _store_dl_log_record(
            task_id=task_id,
            level="ERROR",
            message=f"训练失败: {exc}",
            extra=None,
        )
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


async def list_dl_epochs(
    task_id: str, db: AsyncSession, page: int = 1, page_size: int = 100
) -> dict:
    """Return paginated epoch records for a DL task (ordered by epoch asc)."""
    count_stmt = select(func.count(DLTrainingEpoch.id)).where(DLTrainingEpoch.task_id == task_id)
    total = (await db.execute(count_stmt)).scalar_one()
    offset = (page - 1) * page_size
    stmt = (
        select(DLTrainingEpoch)
        .where(DLTrainingEpoch.task_id == task_id)
        .order_by(DLTrainingEpoch.epoch.asc())
        .offset(offset)
        .limit(page_size)
    )
    epochs = (await db.execute(stmt)).scalars().all()
    return {"items": epochs, "total": total, "page": page, "page_size": page_size}


async def list_dl_epoch_history(
    task_id: str, db: AsyncSession, page: int = 1, page_size: int = 20
) -> dict:
    """Return paginated epoch records ordered by newest epoch first."""
    count_stmt = select(func.count(DLTrainingEpoch.id)).where(DLTrainingEpoch.task_id == task_id)
    total = (await db.execute(count_stmt)).scalar_one()
    offset = (page - 1) * page_size
    stmt = (
        select(DLTrainingEpoch)
        .where(DLTrainingEpoch.task_id == task_id)
        .order_by(DLTrainingEpoch.epoch.desc(), DLTrainingEpoch.created_at.desc())
        .offset(offset)
        .limit(page_size)
    )
    epochs = (await db.execute(stmt)).scalars().all()
    return {"items": epochs, "total": total, "page": page, "page_size": page_size}


async def list_dl_logs(
    task_id: str,
    db: AsyncSession,
    level: str | None = None,
    page: int = 1,
    page_size: int = 200,
) -> dict:
    """Return paginated persisted DL log entries ordered by creation time."""
    stmt = select(DLTrainingLog).where(DLTrainingLog.task_id == task_id)
    count_stmt = select(func.count(DLTrainingLog.id)).where(DLTrainingLog.task_id == task_id)
    if level:
        normalized_level = level.upper()
        stmt = stmt.where(DLTrainingLog.level == normalized_level)
        count_stmt = count_stmt.where(DLTrainingLog.level == normalized_level)

    total = (await db.execute(count_stmt)).scalar_one()
    offset = (page - 1) * page_size
    stmt = stmt.order_by(DLTrainingLog.created_at.asc()).offset(offset).limit(page_size)
    entries = (await db.execute(stmt)).scalars().all()
    return {
        "task_id": task_id,
        "entries": entries,
        "total": total,
        "page": page,
        "page_size": page_size,
    }


async def update_dl_task_meta(
    task_id: str, notes: str | None, tags: list[str] | None, db: AsyncSession
) -> DLTrainingTask:
    """Update notes and/or tags of a DL training task."""
    res = await db.execute(select(DLTrainingTask).where(DLTrainingTask.id == task_id))
    task = res.scalar_one_or_none()
    if task is None:
        raise HTTPException(status_code=404, detail="DL task not found")
    if notes is not None:
        task.notes = notes
    if tags is not None:
        task.tags = tags
    await db.flush()
    return task


# ---------------------------------------------------------------------------
# DL Deployment service
# ---------------------------------------------------------------------------

async def create_dl_deployment(
    dl_task_id: str, name: str, description: str | None, db: AsyncSession
) -> DLModelDeployment:
    res = await db.execute(select(DLTrainingTask).where(DLTrainingTask.id == dl_task_id))
    task = res.scalar_one_or_none()
    if task is None:
        raise HTTPException(status_code=404, detail="DL task not found")
    if task.status != "SUCCESS":
        raise HTTPException(status_code=422, detail="只有训练成功的模型才能部署")
    dep = DLModelDeployment(
        dl_task_id=dl_task_id, name=name, description=description,
    )
    db.add(dep)
    await db.flush()
    await db.refresh(dep)
    return dep


async def list_dl_deployments(db: AsyncSession) -> dict:
    stmt = select(DLModelDeployment).order_by(DLModelDeployment.created_at.desc())
    deps = (await db.execute(stmt)).scalars().all()
    return {"deployments": deps, "total": len(deps)}


async def delete_dl_deployment(dep_id: str, db: AsyncSession) -> None:
    res = await db.execute(select(DLModelDeployment).where(DLModelDeployment.id == dep_id))
    dep = res.scalar_one_or_none()
    if dep is None:
        raise HTTPException(status_code=404, detail="DL deployment not found")
    await db.delete(dep)
    await db.flush()


async def toggle_dl_deployment_status(dep_id: str, status: str, db: AsyncSession) -> DLModelDeployment:
    res = await db.execute(select(DLModelDeployment).where(DLModelDeployment.id == dep_id))
    dep = res.scalar_one_or_none()
    if dep is None:
        raise HTTPException(status_code=404, detail="DL deployment not found")
    if status not in ("active", "paused"):
        raise HTTPException(status_code=422, detail="status must be 'active' or 'paused'")
    dep.status = status
    await db.flush()
    return dep


async def predict_dl_deployment(
    dep_id: str, rows: list[dict], db: AsyncSession
) -> dict:
    """Load DL model and run inference for the given deployment."""

    # Load deployment + task info
    res = await db.execute(
        select(DLModelDeployment).where(DLModelDeployment.id == dep_id)
    )
    dep = res.scalar_one_or_none()
    if dep is None:
        raise HTTPException(status_code=404, detail="DL deployment not found")
    if dep.status == "paused":
        raise HTTPException(status_code=422, detail="部署已暂停，请先恢复部署")

    res2 = await db.execute(
        select(DLTrainingTask).where(DLTrainingTask.id == dep.dl_task_id)
    )
    task = res2.scalar_one_or_none()
    if task is None or not task.model_path:
        raise HTTPException(status_code=404, detail="模型文件不存在")

    # Load training dataset for reference preprocessing (encoding, column order)
    ds_res = await db.execute(select(Dataset).where(Dataset.id == task.dataset_id))
    dataset = ds_res.scalar_one_or_none()
    if dataset is None:
        raise HTTPException(status_code=404, detail="训练数据集不存在，无法推断特征编码")

    loop = asyncio.get_event_loop()

    def _load_and_predict():
        # Build properly-encoded feature matrix (mirrors training preprocessing)
        training_df = load_dataframe(dataset.file_path)
        X_df = prepare_prediction_frame(training_df, rows, task.target_column)
        # Explicitly coerce all columns to numeric (handles object dtype after label encoding)
        X = X_df.apply(pd.to_numeric, errors='coerce').fillna(0).values.astype(np.float32)
        feature_cols = list(X_df.columns)

        trainer = get_dl_trainer(task.model_type)
        meta = trainer.load_for_inference(task.model_path)
        task_type = meta["task_type"]

        preds, probas = trainer.predict(X, task_type)
        return task_type, preds.tolist(), probas, feature_cols

    task_type, predictions, probas, feature_columns = await loop.run_in_executor(
        _executor, _load_and_predict
    )

    # Increment request count
    dep.request_count = (dep.request_count or 0) + 1
    await db.flush()

    prob_list = None
    if probas is not None:
        prob_list = [
            {str(i): round(float(p), 6) for i, p in enumerate(row)}
            for row in probas
        ]

    return {
        "deployment_id":   dep_id,
        "task_type":       task_type,
        "rows":            len(rows),
        "predictions":     predictions,
        "probabilities":   prob_list,
        "feature_columns": feature_columns,
    }


async def predict_dl_task_direct(
    task_id: str, rows: list[dict], db: AsyncSession
) -> dict:
    """Direct inference against a DL task without requiring a deployment."""
    res = await db.execute(select(DLTrainingTask).where(DLTrainingTask.id == task_id))
    task = res.scalar_one_or_none()
    if task is None:
        raise HTTPException(status_code=404, detail="DL task not found")
    if task.status != "SUCCESS" or not task.model_path:
        raise HTTPException(status_code=422, detail="模型未就绪（训练未完成或文件不存在）")

    # Load training dataset for reference preprocessing (encoding, column order)
    ds_res = await db.execute(select(Dataset).where(Dataset.id == task.dataset_id))
    dataset = ds_res.scalar_one_or_none()
    if dataset is None:
        raise HTTPException(status_code=404, detail="训练数据集不存在，无法推断特征编码")

    loop = asyncio.get_event_loop()

    def _predict():
        # Build properly-encoded feature matrix (mirrors training preprocessing)
        training_df = load_dataframe(dataset.file_path)
        X_df = prepare_prediction_frame(training_df, rows, task.target_column)
        # Explicitly coerce all columns to numeric (handles object dtype after label encoding)
        X = X_df.apply(pd.to_numeric, errors='coerce').fillna(0).values.astype(np.float32)
        feature_cols = list(X_df.columns)

        trainer = get_dl_trainer(task.model_type)
        meta = trainer.load_for_inference(task.model_path)
        task_type = meta["task_type"]

        preds, probas = trainer.predict(X, task_type)
        return task_type, preds.tolist(), probas, feature_cols

    task_type, predictions, probas, feature_columns = await loop.run_in_executor(
        _executor, _predict
    )

    prob_list = None
    if probas is not None:
        prob_list = [
            {str(i): round(float(p), 6) for i, p in enumerate(row)}
            for row in probas
        ]

    return {
        "task_id":         task_id,
        "task_type":       task_type,
        "rows":            len(rows),
        "predictions":     predictions,
        "probabilities":   prob_list,
        "feature_columns": feature_columns,
    }

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
from app.core.model_artifact import fit_dl_preprocessing_artifact
from app.models.database import (
    AsyncSession,
    Dataset,
    DLModelDeployment,
    DLTrainingEpoch,
    DLTrainingLog,
    DLTrainingTask,
    async_session_factory,
)
from app.services.prediction_service import (
    load_dataframe,
    prepare_prediction_frame,
    prepare_raw_training_frame,
)
from app.utils.storage_paths import to_portable_storage_path
from app.services.object_storage import (
    restore_dataset_file,
    restore_model_bundle,
    upload_training_artifacts,
)

logger = logging.getLogger(__name__)

_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="dl-train")
_running_tasks: dict[str, asyncio.Task] = {}

# ---------------------------------------------------------------------------
# Task-type auto-detection
# ---------------------------------------------------------------------------

def _detect_task_type(y, requested: str) -> str:
    if requested in ("classification", "regression"):
        return requested

    series = pd.Series(y).dropna()
    if series.empty:
        raise ValueError("目标列没有可用于训练的非空值。")
    if (
        series.dtype == "object"
        or series.dtype.name in {"category", "string", "bool", "boolean"}
    ):
        return "classification"

    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.isna().any():
        return "classification"
    integer_like = np.allclose(numeric.to_numpy(), np.round(numeric.to_numpy()))
    if series.nunique() <= 20 and integer_like:
        return "classification"
    return "regression"


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

# Inner validation ratio used in selection mode — carved out of the OUTER
# training portion so the sealed hold-out is never touched during selection.
_SELECTION_INNER_VAL_RATIO = 0.2


def _classification_stratify(y, resolved_task_type: str, target_column: str):
    """Stratify labels for the OUTER split, with actionable validation errors."""
    if resolved_task_type != "classification":
        return None
    counts = pd.Series(y).value_counts(dropna=True)
    unique_count = int(counts.shape[0])
    min_class_count = int(counts.min()) if unique_count else 0
    if unique_count < 2:
        raise ValueError(
            f"分类目标列 {target_column!r} 只有 {unique_count} 个类别，至少需要 2 个类别。"
        )
    if min_class_count < 2:
        raise ValueError(
            f"分类目标列 {target_column!r} 的最小类别样本数为 {min_class_count}，"
            "无法进行分层切分；请不要选择 ID/序号列，改选真实标签列。"
        )
    return y.values


def _outer_split(file_path: str, target_column: str, test_size: float, task_type: str):
    """The canonical OUTER split (seed 42) shared by training AND finalize.

    final_evaluation_service replays this exact function to recover the sealed
    hold-out, so split parity is guaranteed by construction — never inline a
    second train_test_split with the same arguments elsewhere.
    """
    df = load_dataframe(file_path)
    X, y = prepare_raw_training_frame(df, target_column)
    resolved_task_type = _detect_task_type(y, task_type)
    stratify = _classification_stratify(y, resolved_task_type, target_column)
    raw_X_train, raw_X_val, raw_y_train, raw_y_val = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=stratify
    )
    return raw_X_train, raw_X_val, raw_y_train, raw_y_val, resolved_task_type


def split_raw_holdout(file_path: str, target_column: str, test_size: float, task_type: str):
    """Return (raw_X_holdout, raw_y_holdout, resolved_task_type) — the sealed
    outer hold-out, untouched by selection-mode training."""
    _, raw_X_val, _, raw_y_val, resolved = _outer_split(
        file_path, target_column, test_size, task_type
    )
    return raw_X_val, raw_y_val, resolved


def _prepare_dl_data(
    file_path: str,
    target_column: str,
    test_size: float,
    task_type: str,
    evaluation_mode: str = "standard",
):
    """Split + transform for DL training.

    standard  — legacy V2 behaviour: outer split (seed 42), per-epoch
                validation AND final metrics on the outer val set.
    selection — B1 semantics: the outer val set is the SEALED hold-out and is
                discarded here; an inner split of the training portion serves
                early stopping + selection metrics. final_evaluation_service
                replays the same outer split at finalize time.
    """
    raw_X_train, raw_X_val, raw_y_train, raw_y_val, resolved_task_type = _outer_split(
        file_path, target_column, test_size, task_type
    )

    if evaluation_mode == "selection":
        # Seal the outer hold-out: replace (train, val) with an inner split of
        # the training portion. Stratify when possible; tiny classes fall back
        # to a plain split rather than failing the whole run.
        inner_stratify = (
            raw_y_train.values
            if resolved_task_type == "classification"
            and pd.Series(raw_y_train).value_counts(dropna=True).min() >= 2
            else None
        )
        raw_X_train, raw_X_val, raw_y_train, raw_y_val = train_test_split(
            raw_X_train,
            raw_y_train,
            test_size=_SELECTION_INNER_VAL_RATIO,
            random_state=42,
            stratify=inner_stratify,
        )

    artifact = fit_dl_preprocessing_artifact(
        raw_X_train,
        raw_y_train,
        task_kind=resolved_task_type,
    )
    return (
        artifact.transform_features(raw_X_train),
        artifact.transform_features(raw_X_val),
        artifact.encode_target(raw_y_train),
        artifact.encode_target(raw_y_val),
        artifact,
        resolved_task_type,
    )


def _prepare_dl_prediction_input(
    metadata: dict,
    training_df: pd.DataFrame | None,
    rows: list[dict],
    target_column: str,
) -> tuple[np.ndarray, list[str]]:
    artifact = metadata.get("preprocessing_artifact")
    if artifact is not None:
        values = artifact.transform_features(pd.DataFrame(rows))
        return values.astype(np.float32), artifact.feature_names

    if training_df is None:
        raise ValueError("Legacy DL inference requires the original training dataset")
    prediction_frame = prepare_prediction_frame(
        training_df,
        rows,
        target_column,
    )
    values = (
        prediction_frame.apply(pd.to_numeric, errors="coerce")
        .fillna(0)
        .to_numpy(dtype=np.float32)
    )
    return values, list(prediction_frame.columns)


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
    platform_task_id: str | None = None,
    evaluation_mode: str = "standard",
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

    X_train, X_val, y_train, y_val, preprocessing_artifact, task_type = _prepare_dl_data(
        file_path, target_column, test_size, task_type, evaluation_mode)

    feature_columns = preprocessing_artifact.feature_names
    emit_log(
        "INFO",
        f"任务类型确认: {task_type}"
        + ("（selection 模式：外层 hold-out 已封存，验证用内层切分）"
           if evaluation_mode == "selection" else ""),
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
        if platform_task_id:
            from app.scheduler.task_runner import update_platform_task_status

            asyncio.run_coroutine_threadsafe(
                update_platform_task_status(
                    platform_task_id,
                    "RUNNING",
                    metrics=data,
                    progress=max(0.0, min(1.0, progress / 100.0)),
                ),
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
    model_file = save_dir / f"dl_{task_id}.pt"
    trainer.save(
        str(model_file),
        arch_config=arch_config,
        input_dim=X_train.shape[1],
        task_type=task_type,
        feature_columns=feature_columns,
        preprocessing_artifact=preprocessing_artifact,
    )
    model_path = to_portable_storage_path(model_file)
    logger.info("[DL %s] Saved → %s", task_id, model_path)

    # Log final metrics
    final_message, scalar_metrics = _build_dl_completion_log_entry(result)
    emit_log("INFO", final_message, **scalar_metrics)

    # Upload artifacts to object storage (MinIO)
    settings = get_settings()
    scaler_file = Path(str(model_file) + ".scaler.joblib")
    preprocessing_file = Path(str(model_file) + ".preprocessor.joblib")
    log_file = settings.storage_logs / f"{task_id}.log"
    metrics_file = settings.storage_logs / f"{task_id}_metrics.json"
    upload_training_artifacts(
        task_id=task_id,
        model_files=[
            f for f in [model_file, scaler_file, preprocessing_file] if f.exists()
        ],
        log_files=[f for f in [log_file, metrics_file] if f.exists()],
    )

    return {
        "result_metrics": result,
        "model_path": model_path,
        "task_type": task_type,
        "evaluation_mode": evaluation_mode,
    }


# ---------------------------------------------------------------------------
# Celery entry-point: _run_dl_training_by_id
# ---------------------------------------------------------------------------

async def _run_dl_training_by_id(
    dl_task_id: str,
    platform_task_id: str | None = None,
) -> dict:
    """
    Celery entry-point: load a DLTrainingTask by ID and run the training pipeline.

    Returns {"metrics": {...}, "model_path": "..."}.
    Mirrors _run_training_sync_by_id in training_service.
    """
    settings = get_settings()

    # Load domain task parameters
    async with async_session_factory() as db:
        result = await db.execute(select(DLTrainingTask).where(DLTrainingTask.id == dl_task_id))
        task = result.scalar_one_or_none()
        if task is None:
            raise ValueError(f"DLTrainingTask {dl_task_id!r} not found")
        # Load related dataset
        dataset_result = await db.execute(select(Dataset).where(Dataset.id == task.dataset_id))
        dataset = dataset_result.scalar_one_or_none()
        if dataset is None:
            raise ValueError(f"Dataset {task.dataset_id!r} not found for DLTask {dl_task_id!r}")
        file_path = restore_dataset_file(dataset.id, dataset.file_path)
        if file_path is None:
            raise ValueError(f"Dataset artifact {dataset.id!r} not found for DLTask {dl_task_id!r}")
        target_column = task.target_column
        model_type    = task.model_type
        task_type_req = task.task_type or "auto"
        arch_config   = task.arch_config or {}
        opt_config    = task.opt_config or {}
        train_config  = task.train_config or {}

        # B1: V3 runs dispatched with evaluation_mode=selection seal the outer
        # hold-out (same resolution as classic ML in training_service).
        from app.services.training_service import _resolve_evaluation_mode

        evaluation_mode = await _resolve_evaluation_mode(db, platform_task_id)

    # Mark domain task RUNNING
    async with async_session_factory() as db:
        result = await db.execute(select(DLTrainingTask).where(DLTrainingTask.id == dl_task_id))
        t = result.scalar_one_or_none()
        if t:
            t.status = "RUNNING"
            t.started_at = datetime.now(timezone.utc)
            t.total_epochs = int(train_config.get("epochs", 50))
            await db.commit()

    loop = asyncio.get_event_loop()
    try:
        training_result = await loop.run_in_executor(
            _executor,
            _run_dl_sync,
            dl_task_id, file_path, target_column, model_type, task_type_req,
            arch_config, opt_config, train_config, str(settings.storage_models), loop,
            platform_task_id, evaluation_mode,
        )
        stored_metrics = training_result["result_metrics"]

        async with async_session_factory() as db:
            result = await db.execute(select(DLTrainingTask).where(DLTrainingTask.id == dl_task_id))
            t = result.scalar_one_or_none()
            if t:
                t.status = "SUCCESS"
                t.progress = 100.0
                t.current_epoch = int(stored_metrics.get("final_epoch") or t.total_epochs or 0)
                t.task_type = training_result["task_type"]
                t.result_metrics = stored_metrics
                t.model_path = training_result["model_path"]
                t.finished_at = datetime.now(timezone.utc)
                await db.commit()

        return {
            "metrics": stored_metrics,
            "model_path": training_result["model_path"],
            "evaluation_mode": training_result.get("evaluation_mode", "standard"),
        }

    except Exception as exc:
        async with async_session_factory() as db:
            result = await db.execute(select(DLTrainingTask).where(DLTrainingTask.id == dl_task_id))
            t = result.scalar_one_or_none()
            if t:
                t.status = "FAILED"
                t.error_message = str(exc)
                t.finished_at = datetime.now(timezone.utc)
                await db.commit()
        raise


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
    platform_task_id: str | None = None,
):
    """Background coroutine that runs DL training and updates both domain + platform tables."""
    from app.scheduler.task_runner import update_platform_task_status

    # ── Mark RUNNING ──────────────────────────────────────────────────────────
    async with async_session_factory() as db:
        result = await db.execute(select(DLTrainingTask).where(DLTrainingTask.id == task_id))
        task = result.scalar_one_or_none()
        if task:
            task.status = "RUNNING"
            task.started_at = datetime.now(timezone.utc)
            task.total_epochs = int(train_config.get("epochs", 50))
            await db.commit()

    if platform_task_id:
        await update_platform_task_status(platform_task_id, "RUNNING")

    loop = asyncio.get_event_loop()
    try:
        training_result = await loop.run_in_executor(
            _executor,
            _run_dl_sync,
            task_id, file_path, target_column, model_type, task_type,
            arch_config, opt_config, train_config, model_save_dir, loop,
            platform_task_id,
        )

        stored_metrics = training_result["result_metrics"]

        # ── Mark SUCCESS (domain) ─────────────────────────────────────────────
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

        # ── Write-back: PlatformTask SUCCESS ─────────────────────────────────
        if platform_task_id:
            await update_platform_task_status(
                platform_task_id, "SUCCESS",
                metrics=stored_metrics,
            )

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
        # ── Mark FAILED (domain) ──────────────────────────────────────────────
        async with async_session_factory() as db:
            result = await db.execute(select(DLTrainingTask).where(DLTrainingTask.id == task_id))
            task = result.scalar_one_or_none()
            if task:
                task.status = "FAILED"
                task.error_message = str(exc)
                task.finished_at = datetime.now(timezone.utc)
                await db.commit()

        # ── Write-back: PlatformTask FAILED ──────────────────────────────────
        if platform_task_id:
            await update_platform_task_status(platform_task_id, "FAILED", error=str(exc))

        event_bus.publish(f"dl:{task_id}", {"type": "done", "status": "FAILED", "error": str(exc)})
    finally:
        _running_tasks.pop(task_id, None)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def start_dl_training(request_data: dict, db: AsyncSession) -> DLTrainingTask:
    # Validate dataset
    dataset_id = request_data["dataset_id"]
    res = await db.execute(select(Dataset).where(Dataset.id == dataset_id))
    dataset = res.scalar_one_or_none()
    if dataset is None:
        raise HTTPException(status_code=404, detail="Dataset not found")
    restored_dataset = restore_dataset_file(dataset.id, dataset.file_path)
    if restored_dataset is None:
        raise HTTPException(status_code=404, detail="Dataset artifact not found")

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

    # ── Write-back contract: register in unified platform task table ──────────
    from app.scheduler.task_runner import register_domain_task
    platform_task = await register_domain_task(
        db=db,
        kind="dl_train",
        payload_ref=f"dl_train:{task.id}",
    )
    platform_task_id = platform_task.id
    await db.commit()

    from app.scheduler.scheduler import get_scheduler

    scheduled = await get_scheduler("dl_train").submit(platform_task_id)
    if isinstance(scheduled, asyncio.Task):
        _running_tasks[task.id] = scheduled
        scheduled.add_done_callback(
            lambda _done, domain_id=task.id: _running_tasks.pop(domain_id, None)
        )
    return task


async def stop_dl_training(task_id: str, db: AsyncSession) -> DLTrainingTask:
    from app.models.database import PlatformTask

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

    ptask_res = await db.execute(
        select(PlatformTask).where(PlatformTask.payload_ref == f"dl_train:{task_id}")
    )
    platform_task = ptask_res.scalar_one_or_none()
    if platform_task is not None:
        platform_task.status = "CANCELLED"
        platform_task.error_message = "Manually stopped"
        platform_task.finished_at = datetime.now(timezone.utc)

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
    if restore_dataset_file(dataset.id, dataset.file_path) is None:
        raise HTTPException(status_code=404, detail="训练数据集文件不存在，无法推断特征编码")
    model_path = restore_model_bundle(task.model_path)
    if model_path is None:
        raise HTTPException(status_code=404, detail="模型文件不存在")

    loop = asyncio.get_event_loop()

    def _load_and_predict():
        trainer = get_dl_trainer(task.model_type)
        meta = trainer.load_for_inference(str(model_path))
        task_type = meta["task_type"]
        training_df = (
            load_dataframe(dataset.file_path)
            if meta.get("preprocessing_artifact") is None
            else None
        )
        X, feature_cols = _prepare_dl_prediction_input(
            meta,
            training_df,
            rows,
            task.target_column,
        )

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
    if restore_dataset_file(dataset.id, dataset.file_path) is None:
        raise HTTPException(status_code=404, detail="训练数据集文件不存在，无法推断特征编码")
    model_path = restore_model_bundle(task.model_path)
    if model_path is None:
        raise HTTPException(status_code=404, detail="模型文件不存在")

    loop = asyncio.get_event_loop()

    def _predict():
        trainer = get_dl_trainer(task.model_type)
        meta = trainer.load_for_inference(str(model_path))
        task_type = meta["task_type"]
        training_df = (
            load_dataframe(dataset.file_path)
            if meta.get("preprocessing_artifact") is None
            else None
        )
        X, feature_cols = _prepare_dl_prediction_input(
            meta,
            training_df,
            rows,
            task.target_column,
        )

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


# ---------------------------------------------------------------------------
# Executor registration — V3 Phase 2
# ---------------------------------------------------------------------------
# Register the DL training coroutine in the unified executor registry so the
# Scheduler can dispatch ``kind="dl_train"`` tasks without a special-case
# branch.  Contract matches (domain_id, platform_task_id) → {"metrics", ...}.

from app.scheduler.executors import register_executor as _register_executor  # noqa: E402
_register_executor("dl_train", _run_dl_training_by_id)

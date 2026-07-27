"""Training task service -- orchestrates model training."""

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from fastapi import HTTPException
from sqlalchemy import func, select
from sklearn.model_selection import train_test_split

from app.config import get_settings
from app.core.trainer import detect_task_type, get_trainer, list_available_models
from app.core.logger import TrainingLogger
from app.models.database import (
    AsyncSession,
    Dataset,
    ExperimentRun,
    TrainingTask,
    async_session_factory,
)
from app.services.prediction_service import load_dataframe, prepare_raw_training_frame
from app.utils.storage_paths import to_portable_storage_path
from app.services.object_storage import restore_dataset_file, upload_training_artifacts

logger = logging.getLogger(__name__)

# Thread pool for running training in background
_executor = ThreadPoolExecutor(max_workers=4)

# Track running tasks for cancellation
_running_tasks: dict[str, asyncio.Task] = {}


def _progress_fraction(step: int | float, total: int | float) -> float:
    """Return scheduler progress on the PlatformTask 0..1 scale."""
    try:
        total_f = float(total)
        if total_f <= 0:
            return 0.0
        progress = float(step) / total_f
    except (TypeError, ValueError, ZeroDivisionError):
        return 0.0
    return max(0.0, min(1.0, progress))


def _make_platform_progress_callback(
    platform_task_id: str | None,
    loop: asyncio.AbstractEventLoop,
) -> Callable[[int, int, dict], None] | None:
    if not platform_task_id:
        return None

    from app.scheduler.task_runner import update_platform_task_status

    def _callback(step: int, total: int, metrics: dict) -> None:
        future = asyncio.run_coroutine_threadsafe(
            update_platform_task_status(
                platform_task_id,
                "RUNNING",
                metrics=metrics,
                progress=_progress_fraction(step, total),
            ),
            loop,
        )
        future.add_done_callback(
            lambda f: logger.debug(
                "PlatformTask progress callback failed: %s", f.exception()
            ) if f.exception() else None
        )

    return _callback

def _prepare_data(file_path: str, target_column: str, test_size: float, is_regression: bool = False):
    """Load data and split raw rows before any fitted transformation."""
    df = load_dataframe(file_path)
    X, y = prepare_raw_training_frame(df, target_column)

    # Regression targets are continuous — stratify is not applicable
    stratify = None
    if not is_regression:
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
        stratify = y
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=stratify
    )

    return (
        X_train.reset_index(drop=True),
        X_val.reset_index(drop=True),
        y_train.reset_index(drop=True),
        y_val.reset_index(drop=True),
    )


def _trainer_validation_inputs(evaluation_mode: str, X_val, y_val):
    if evaluation_mode == "selection":
        return None, None
    return X_val, y_val


async def _resolve_evaluation_mode(
    db: AsyncSession,
    platform_task_id: str | None,
) -> str:
    if not platform_task_id:
        return "standard"
    run = (
        await db.execute(
            select(ExperimentRun).where(ExperimentRun.task_id == platform_task_id)
        )
    ).scalar_one_or_none()
    if run is not None and (run.search_meta or {}).get("evaluation_mode") == "selection":
        return "selection"
    return "standard"


def _apply_class_weight(
    hyperparameters: dict,
    model_type: str,
    class_weight: str | None,
    y_train,
) -> dict:
    """Translate the top-level class_weight field into model-specific hyperparameter keys."""
    if not class_weight:
        class_weight = hyperparameters.get("class_weight")
    if not class_weight:
        return hyperparameters

    hp = dict(hyperparameters)

    if model_type == "xgboost":
        # XGBoost uses scale_pos_weight for binary classification
        unique, counts = np.unique(y_train, return_counts=True)
        if len(unique) == 2 and counts[1] > 0:
            hp["scale_pos_weight"] = float(counts[0]) / float(counts[1])
    elif model_type == "lightgbm":
        hp["is_unbalance"] = True
    elif model_type in ("random_forest", "extra_trees", "gradient_boosting",
                        "logistic_regression", "svm"):
        hp["class_weight"] = class_weight
    # decision_tree, knn, gaussian_nb, mlp, neural: not supported — silently ignored

    return hp


def _try_init_mlflow():
    """Try to import and configure MLflow. Returns None if unavailable."""
    try:
        import mlflow
        settings = get_settings()
        tracking_uri = settings.mlflow_tracking_uri
        # Resolve relative paths for file-based backends
        if not tracking_uri.startswith(("http://", "https://", "file://", "sqlite://", "postgresql://", "mysql://")):
            from pathlib import Path
            abs_path = (settings.project_root / tracking_uri).resolve()
            abs_path.mkdir(parents=True, exist_ok=True)
            tracking_uri = abs_path.as_uri()
        elif tracking_uri.startswith("sqlite:///./"):
            # Resolve relative SQLite path
            from pathlib import Path
            rel_path = tracking_uri.replace("sqlite:///./", "")
            abs_path = (settings.project_root / rel_path).resolve()
            abs_path.parent.mkdir(parents=True, exist_ok=True)
            tracking_uri = f"sqlite:///{abs_path}"
        mlflow.set_tracking_uri(tracking_uri)
        return mlflow
    except ImportError:
        logger.warning("MLflow not installed, experiment tracking disabled")
        return None
    except Exception as e:
        logger.warning("MLflow init failed: %s, tracking disabled", e)
        return None


def _log_model_artifact(mlflow, model_file: Path) -> None:
    """Store the self-contained joblib without assuming an sklearn model shape."""
    mlflow.log_artifact(str(model_file), artifact_path="model")


def _run_training_sync(
    task_id: str,
    file_path: str,
    target_column: str,
    model_type: str,
    hyperparameters: dict,
    test_size: float,
    eval_metrics: list[str],
    cv_folds: int,
    model_save_dir: str,
    class_weight: str | None = None,
    progress_callback: Callable[[int, int, dict], None] | None = None,
    evaluation_mode: str = "standard",
) -> dict:
    """Synchronous training function to run in thread pool."""
    # Initialize per-task logger
    tl = TrainingLogger(task_id=task_id, model_type=model_type)
    tl.log("INFO", "Training task started", model=model_type, dataset=file_path)

    try:
        return _run_training_sync_inner(
            tl, task_id, file_path, target_column, model_type,
            hyperparameters, test_size, eval_metrics, cv_folds,
            model_save_dir, class_weight, progress_callback, evaluation_mode,
        )
    except Exception as exc:
        # Record the failure explicitly so the Inspector can show WHY it
        # failed. The exception is re-raised for the caller to handle.
        try:
            tl.log("ERROR", f"Training failed: {exc}", exc_type=type(exc).__name__)
            tl.log_status("FAILED", str(exc))
        except Exception:
            pass
        raise
    finally:
        # Flush buffered log lines regardless of outcome so the
        # `training_logs` table always has a record for successful AND
        # failed runs.
        try:
            tl.flush_to_db()
        except Exception:
            pass


def _run_training_sync_inner(
    tl: TrainingLogger,
    task_id: str,
    file_path: str,
    target_column: str,
    model_type: str,
    hyperparameters: dict,
    test_size: float,
    eval_metrics: list[str],
    cv_folds: int,
    model_save_dir: str,
    class_weight: str | None,
    progress_callback: Callable[[int, int, dict], None] | None,
    evaluation_mode: str = "standard",
) -> dict:
    # Try MLflow integration (optional)
    mlflow = _try_init_mlflow()
    mlflow_run = None

    # Prepare data
    tl.log("INFO", "Loading and preparing data...")
    is_regression = detect_task_type(model_type) == "regression"
    X_train, X_val, y_train, y_val = _prepare_data(file_path, target_column, test_size, is_regression)
    tl.log("INFO", "Data prepared",
           train_samples=len(X_train), val_samples=len(X_val),
           features=X_train.shape[1], target=target_column,
           evaluation_mode=evaluation_mode)

    # Translate class_weight into model-specific hyperparameter keys
    effective_hp = _apply_class_weight(hyperparameters, model_type, class_weight, y_train)

    # Create and configure trainer
    trainer = get_trainer(model_type)
    trainer.configure(effective_hp)
    tl.log("INFO", "Model configured", params=str(hyperparameters))

    # Start MLflow run if available
    if mlflow:
        try:
            mlflow.set_experiment("ml_platform")
            mlflow_run = mlflow.start_run(run_name=f"{model_type}_{task_id[:8]}")
            mlflow.log_params({
                "model_type": model_type,
                "target_column": target_column,
                "test_size": test_size,
                "cv_folds": cv_folds,
                **{k: str(v) for k, v in effective_hp.items()},
            })
            tl.log("INFO", "MLflow run started", run_id=mlflow_run.info.run_id)
        except Exception as e:
            tl.log("WARN", f"MLflow logging failed: {e}")
            mlflow = None

    # Callback that logs each fold/step
    def on_fold_complete(step: int, total: int, metrics: dict):
        tl.log_metrics(step=step, total_steps=total, metrics=metrics)
        if progress_callback:
            progress_callback(step, total, metrics)
        # Log fold metrics to MLflow
        if mlflow:
            try:
                fold_metrics = {k: v for k, v in metrics.items() if k != "fold" and v is not None}
                mlflow.log_metrics(fold_metrics, step=step)
            except Exception:
                pass

    # Train
    tl.log("INFO", f"Starting training with {cv_folds}-fold cross validation")
    trainer_X_val, trainer_y_val = _trainer_validation_inputs(
        evaluation_mode, X_val, y_val
    )
    result_metrics = trainer.train(
        X_train, y_train, trainer_X_val, trainer_y_val,
        eval_metrics=eval_metrics,
        cv_folds=cv_folds,
        callback=on_fold_complete,
    )

    # Ensure model save directory exists
    save_dir = Path(model_save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Save model
    model_file = save_dir / f"{task_id}.joblib"
    trainer.save(str(model_file))
    model_path = to_portable_storage_path(model_file)
    tl.log("INFO", "Model saved", path=model_path)

    # Log final metrics
    clean_metrics = {k: v for k, v in result_metrics.items() if k != "cv_folds"}
    tl.log("INFO", "Training completed", **{k: str(v) for k, v in clean_metrics.items()})
    tl.log_status("SUCCESS", "Training completed successfully", result_metrics=clean_metrics)

    # Upload artifacts to object storage (MinIO)
    upload_training_artifacts(
        task_id=task_id,
        model_files=[model_file],
        log_files=[tl.log_file, tl.metrics_file],
    )

    # Finalize MLflow run
    if mlflow and mlflow_run:
        try:
            final_log = {k: v for k, v in clean_metrics.items() if v is not None}
            mlflow.log_metrics({f"final_{k}": v for k, v in final_log.items() if isinstance(v, (int, float))})
            mlflow.log_artifact(str(tl.log_file))
            mlflow.log_artifact(str(tl.metrics_file))
            _log_model_artifact(mlflow, model_file)
            mlflow.end_run()
            tl.log("INFO", "MLflow run completed")
        except Exception as e:
            tl.log("WARN", f"MLflow finalization failed: {e}")
            try:
                mlflow.end_run(status="FAILED")
            except Exception:
                pass

    return {
        "result_metrics": result_metrics,
        "model_path": model_path,
    }


async def create_training_task_record(db: AsyncSession, candidate: dict) -> TrainingTask:
    """
    Create a TrainingTask DB row from a candidate dict WITHOUT launching execution.

    Used by the V3 batch pipeline to pre-create domain tasks
    before dispatching them to Celery via PlatformTask.

    candidate keys (all optional except dataset_id, model_type, target_column):
      dataset_id, model_type, target_column, hyperparameters, test_size,
      eval_metrics, cv_folds, cross_validation, class_weight
    """
    import uuid as _uuid_mod

    dataset_id = candidate["dataset_id"]
    result = await db.execute(select(Dataset).where(Dataset.id == dataset_id))
    dataset = result.scalar_one_or_none()
    if dataset is None:
        raise HTTPException(status_code=404, detail=f"Dataset {dataset_id!r} not found")
    owner_username = candidate.get("owner_username") or dataset.owner_username

    available = list_available_models()
    model_type = candidate["model_type"]
    if model_type not in available:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown model type {model_type!r}. Available: {available}",
        )

    cv_config = candidate.get("cross_validation") or {}
    cv_folds = int(candidate.get("cv_folds") or cv_config.get("folds") or 5)
    short_id = str(_uuid_mod.uuid4())[:8]
    hyperparameters = dict(candidate.get("hyperparameters", {}))
    if candidate.get("class_weight"):
        # Scheduler executors reconstruct their input solely from this
        # committed domain row, so persist the top-level option with it.
        hyperparameters["class_weight"] = candidate["class_weight"]
    task = TrainingTask(
        owner_username=owner_username,
        dataset_id=dataset_id,
        model_type=model_type,
        name=f"{model_type}_{short_id}",
        hyperparameters=hyperparameters,
        target_column=candidate["target_column"],
        test_size=candidate.get("test_size", 0.2),
        cv_folds=cv_folds,
        eval_metrics=candidate.get("eval_metrics", ["accuracy"]),
        status="PENDING",
    )
    db.add(task)
    await db.flush()
    await db.refresh(task)
    return task


async def _run_training_sync_by_id(
    training_task_id: str,
    platform_task_id: str | None = None,
) -> dict:
    """
    Celery entry-point: load a TrainingTask by ID and run the training pipeline.

    Returns a dict with {"metrics": {...}, "model_path": "..."}.
    Updates both TrainingTask and (optionally) PlatformTask on completion.
    """
    from app.models.database import async_session_factory

    settings = get_settings()

    # Load domain task + associated dataset
    async with async_session_factory() as db:
        result = await db.execute(select(TrainingTask).where(TrainingTask.id == training_task_id))
        task = result.scalar_one_or_none()
        if task is None:
            raise ValueError(f"TrainingTask {training_task_id!r} not found")
        ds_result = await db.execute(select(Dataset).where(Dataset.id == task.dataset_id))
        dataset = ds_result.scalar_one_or_none()
        if dataset is None:
            raise ValueError(f"Dataset {task.dataset_id!r} not found for TrainingTask {training_task_id!r}")
        restored_dataset = restore_dataset_file(dataset.id, dataset.file_path)
        if restored_dataset is None:
            raise ValueError(f"Dataset artifact {dataset.id!r} not found for TrainingTask {training_task_id!r}")
        file_path     = str(restored_dataset)
        target_column = task.target_column
        model_type    = task.model_type
        hyperparams   = task.hyperparameters or {}
        test_size     = task.test_size or 0.2
        eval_metrics  = task.eval_metrics or ["accuracy"]
        cv_folds      = int(getattr(task, "cv_folds", None) or 5)
        evaluation_mode = await _resolve_evaluation_mode(db, platform_task_id)

    # Mark domain task RUNNING
    async with async_session_factory() as db:
        result = await db.execute(select(TrainingTask).where(TrainingTask.id == training_task_id))
        task = result.scalar_one_or_none()
        if task:
            task.status = "RUNNING"
            task.started_at = datetime.now(timezone.utc)
            await db.commit()

    try:
        loop = asyncio.get_event_loop()
        progress_callback = _make_platform_progress_callback(platform_task_id, loop)
        training_result = await loop.run_in_executor(
            _executor,
            _run_training_sync,
            training_task_id, file_path, target_column, model_type,
            hyperparams, test_size, eval_metrics, cv_folds,
            str(settings.storage_models), None, progress_callback, evaluation_mode,
        )
        metrics = {k: v for k, v in training_result["result_metrics"].items() if k != "cv_folds"}

        async with async_session_factory() as db:
            result = await db.execute(select(TrainingTask).where(TrainingTask.id == training_task_id))
            task = result.scalar_one_or_none()
            if task:
                task.status = "SUCCESS"
                task.progress = 100.0
                task.result_metrics = metrics
                task.model_path = training_result["model_path"]
                task.finished_at = datetime.now(timezone.utc)
                await db.commit()

        return {"metrics": metrics, "model_path": training_result["model_path"]}

    except Exception as exc:
        async with async_session_factory() as db:
            result = await db.execute(select(TrainingTask).where(TrainingTask.id == training_task_id))
            task = result.scalar_one_or_none()
            if task:
                task.status = "FAILED"
                task.error_message = str(exc)
                task.finished_at = datetime.now(timezone.utc)
                await db.commit()
        raise


async def start_training(
    request_data: dict,
    db: AsyncSession,
    owner_username: str | None = None,
) -> TrainingTask:
    """Create a training task, register it in the unified platform, and launch it."""
    # Validate dataset exists
    dataset_id = request_data["dataset_id"]
    result = await db.execute(select(Dataset).where(Dataset.id == dataset_id))
    dataset = result.scalar_one_or_none()
    if dataset is None or (owner_username and dataset.owner_username != owner_username):
        raise HTTPException(status_code=404, detail="Dataset not found")
    restored_dataset = restore_dataset_file(dataset.id, dataset.file_path)
    if restored_dataset is None:
        raise HTTPException(status_code=404, detail="Dataset artifact not found")

    # Validate model type
    available = list_available_models()
    if request_data["model_type"] not in available:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown model type '{request_data['model_type']}'. Available: {available}",
        )

    # Create domain task record
    cv_config = request_data.get("cross_validation") or {}
    cv_folds = cv_config.get("folds", 5) if cv_config.get("enabled", True) else 3
    import uuid as _uuid_mod
    short_id = str(_uuid_mod.uuid4())[:8]
    hyperparameters = dict(request_data.get("hyperparameters", {}))
    if request_data.get("class_weight"):
        hyperparameters["class_weight"] = request_data["class_weight"]
    task = TrainingTask(
        owner_username=owner_username or dataset.owner_username,
        dataset_id=dataset_id,
        model_type=request_data["model_type"],
        name=f"{request_data['model_type']}_{short_id}",
        hyperparameters=hyperparameters,
        target_column=request_data["target_column"],
        test_size=request_data.get("test_size", 0.2),
        cv_folds=cv_folds,
        eval_metrics=request_data.get("eval_metrics", ["accuracy"]),
        status="PENDING",
    )
    db.add(task)
    await db.flush()
    await db.refresh(task)

    task_id = task.id
    # ── Write-back contract: register in unified platform task table ──────────
    from app.scheduler.task_runner import register_domain_task
    platform_task = await register_domain_task(
        db=db,
        kind="train",
        payload_ref=f"train:{task_id}",
    )
    platform_task_id = platform_task.id
    # commit both records together
    await db.commit()

    from app.scheduler.scheduler import get_scheduler

    scheduled = await get_scheduler("train").submit(platform_task_id)
    if isinstance(scheduled, asyncio.Task):
        _running_tasks[task_id] = scheduled
        scheduled.add_done_callback(
            lambda _done, domain_id=task_id: _running_tasks.pop(domain_id, None)
        )

    return task


async def get_training_status(
    task_id: str,
    db: AsyncSession,
    owner_username: str | None = None,
) -> TrainingTask:
    """Retrieve the current state of a training task."""
    stmt = select(TrainingTask).where(TrainingTask.id == task_id)
    if owner_username:
        stmt = stmt.where(TrainingTask.owner_username == owner_username)
    result = await db.execute(stmt)
    task = result.scalar_one_or_none()
    if task is None:
        raise HTTPException(status_code=404, detail="Training task not found")
    return task


async def stop_training(
    task_id: str,
    db: AsyncSession,
    owner_username: str | None = None,
) -> TrainingTask:
    """Cancel a pending or running training task."""
    from app.models.database import PlatformTask

    stmt = select(TrainingTask).where(TrainingTask.id == task_id)
    if owner_username:
        stmt = stmt.where(TrainingTask.owner_username == owner_username)
    result = await db.execute(stmt)
    task = result.scalar_one_or_none()
    if task is None:
        raise HTTPException(status_code=404, detail="Training task not found")

    if task.status not in ("PENDING", "RUNNING"):
        raise HTTPException(status_code=400, detail=f"Cannot stop task with status '{task.status}'")

    # Cancel if running
    if task_id in _running_tasks:
        _running_tasks[task_id].cancel()
        del _running_tasks[task_id]

    task.status = "FAILED"
    task.error_message = "Manually stopped by user"
    task.finished_at = datetime.now(timezone.utc)

    platform_result = await db.execute(
        select(PlatformTask).where(PlatformTask.payload_ref == f"train:{task_id}")
    )
    platform_task = platform_result.scalar_one_or_none()
    if platform_task is not None:
        platform_task.status = "CANCELLED"
        platform_task.error_message = "Manually stopped by user"
        platform_task.finished_at = datetime.now(timezone.utc)

    await db.flush()
    return task


async def list_training_tasks(
    db: AsyncSession,
    page: int = 1,
    page_size: int = 20,
    status_filter: str | None = None,
    owner_username: str | None = None,
) -> dict:
    """Return a paginated list of training tasks."""
    stmt = select(TrainingTask)
    count_stmt = select(func.count(TrainingTask.id))

    if status_filter:
        stmt = stmt.where(TrainingTask.status == status_filter)
        count_stmt = count_stmt.where(TrainingTask.status == status_filter)
    if owner_username:
        stmt = stmt.where(TrainingTask.owner_username == owner_username)
        count_stmt = count_stmt.where(TrainingTask.owner_username == owner_username)

    count_result = await db.execute(count_stmt)
    total = count_result.scalar_one()

    offset = (page - 1) * page_size
    stmt = stmt.order_by(TrainingTask.created_at.desc()).offset(offset).limit(page_size)
    result = await db.execute(stmt)
    tasks = result.scalars().all()

    return {
        "items": tasks,
        "total": total,
        "page": page,
        "page_size": page_size,
    }


async def rename_training_task(
    task_id: str,
    name: str,
    db: AsyncSession,
    owner_username: str | None = None,
) -> TrainingTask:
    """Rename a training task."""
    stmt = select(TrainingTask).where(TrainingTask.id == task_id)
    if owner_username:
        stmt = stmt.where(TrainingTask.owner_username == owner_username)
    result = await db.execute(stmt)
    task = result.scalar_one_or_none()
    if task is None:
        raise HTTPException(status_code=404, detail="Training task not found")
    task.name = name
    await db.flush()
    return task


async def delete_training_task(
    task_id: str,
    db: AsyncSession,
    owner_username: str | None = None,
) -> None:
    """Delete a training task (only if not RUNNING)."""
    stmt = select(TrainingTask).where(TrainingTask.id == task_id)
    if owner_username:
        stmt = stmt.where(TrainingTask.owner_username == owner_username)
    result = await db.execute(stmt)
    task = result.scalar_one_or_none()
    if task is None:
        raise HTTPException(status_code=404, detail="Training task not found")
    if task.status == "RUNNING":
        raise HTTPException(status_code=422, detail="Cannot delete a running task. Stop it first.")
    await db.delete(task)
    await db.flush()


async def update_training_task_meta(
    task_id: str,
    notes: str | None,
    tags: list[str] | None,
    db: AsyncSession,
    owner_username: str | None = None,
) -> TrainingTask:
    """Update notes and/or tags of a training task."""
    stmt = select(TrainingTask).where(TrainingTask.id == task_id)
    if owner_username:
        stmt = stmt.where(TrainingTask.owner_username == owner_username)
    result = await db.execute(stmt)
    task = result.scalar_one_or_none()
    if task is None:
        raise HTTPException(status_code=404, detail="Training task not found")
    if notes is not None:
        task.notes = notes
    if tags is not None:
        task.tags = tags
    await db.flush()
    return task


# ---------------------------------------------------------------------------
# Executor registration — V3 Phase 2
# ---------------------------------------------------------------------------
# This lets the Scheduler call us through the registry without importing
# training_service directly.  The executor contract is:
#     async (domain_id, platform_task_id) -> {"metrics": {...}, ...}
# which _run_training_sync_by_id already satisfies — so we register it as-is.

from app.scheduler.executors import register_executor as _register_executor  # noqa: E402
_register_executor("train", _run_training_sync_by_id)

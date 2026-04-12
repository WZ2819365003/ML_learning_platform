"""Training task service -- orchestrates model training."""

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from fastapi import HTTPException
from sqlalchemy import func, select
from sklearn.model_selection import train_test_split

from app.config import get_settings
from app.core.trainer import detect_task_type, get_trainer, list_available_models
from app.core.logger import TrainingLogger
from app.models.database import AsyncSession, Dataset, TrainingTask, async_session_factory
from app.services.prediction_service import load_dataframe, prepare_training_frame
from app.utils.storage_paths import to_portable_storage_path

logger = logging.getLogger(__name__)

# Thread pool for running training in background
_executor = ThreadPoolExecutor(max_workers=4)

# Track running tasks for cancellation
_running_tasks: dict[str, asyncio.Task] = {}

def _prepare_data(file_path: str, target_column: str, test_size: float, is_regression: bool = False):
    """Load data, encode labels, split into train/val sets."""
    df = load_dataframe(file_path)
    X, y, _, _ = prepare_training_frame(df, target_column)

    # Regression targets are continuous — stratify is not applicable
    stratify = None if is_regression else y.values
    X_train, X_val, y_train, y_val = train_test_split(
        X.values, y.values, test_size=test_size, random_state=42, stratify=stratify
    )

    return X_train, X_val, y_train, y_val


def _apply_class_weight(
    hyperparameters: dict,
    model_type: str,
    class_weight: str | None,
    y_train,
) -> dict:
    """Translate the top-level class_weight field into model-specific hyperparameter keys."""
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
) -> dict:
    """Synchronous training function to run in thread pool."""
    # Initialize per-task logger
    tl = TrainingLogger(task_id=task_id, model_type=model_type)
    tl.log("INFO", "Training task started", model=model_type, dataset=file_path)

    # Try MLflow integration (optional)
    mlflow = _try_init_mlflow()
    mlflow_run = None

    # Prepare data
    tl.log("INFO", "Loading and preparing data...")
    is_regression = detect_task_type(model_type) == "regression"
    X_train, X_val, y_train, y_val = _prepare_data(file_path, target_column, test_size, is_regression)
    tl.log("INFO", "Data prepared",
           train_samples=len(X_train), val_samples=len(X_val),
           features=X_train.shape[1], target=target_column)

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
        # Log fold metrics to MLflow
        if mlflow:
            try:
                fold_metrics = {k: v for k, v in metrics.items() if k != "fold" and v is not None}
                mlflow.log_metrics(fold_metrics, step=step)
            except Exception:
                pass

    # Train
    tl.log("INFO", f"Starting training with {cv_folds}-fold cross validation")
    result_metrics = trainer.train(
        X_train, y_train, X_val, y_val,
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

    # Finalize MLflow run
    if mlflow and mlflow_run:
        try:
            final_log = {k: v for k, v in clean_metrics.items() if v is not None}
            mlflow.log_metrics({f"final_{k}": v for k, v in final_log.items() if isinstance(v, (int, float))})
            mlflow.log_artifact(str(tl.log_file))
            mlflow.log_artifact(str(tl.metrics_file))
            mlflow.sklearn.log_model(trainer.model, "model")
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


async def start_training(request_data: dict, db: AsyncSession) -> TrainingTask:
    """Create a training task and launch it in background."""
    settings = get_settings()

    # Validate dataset exists
    dataset_id = request_data["dataset_id"]
    result = await db.execute(select(Dataset).where(Dataset.id == dataset_id))
    dataset = result.scalar_one_or_none()
    if dataset is None:
        raise HTTPException(status_code=404, detail="Dataset not found")

    # Validate model type
    available = list_available_models()
    if request_data["model_type"] not in available:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown model type '{request_data['model_type']}'. Available: {available}",
        )

    # Create task record
    cv_config = request_data.get("cross_validation") or {}
    import uuid as _uuid_mod
    short_id = str(_uuid_mod.uuid4())[:8]
    task = TrainingTask(
        dataset_id=dataset_id,
        model_type=request_data["model_type"],
        name=f"{request_data['model_type']}_{short_id}",
        hyperparameters=request_data.get("hyperparameters", {}),
        target_column=request_data["target_column"],
        test_size=request_data.get("test_size", 0.2),
        eval_metrics=request_data.get("eval_metrics", ["accuracy"]),
        status="PENDING",
    )
    db.add(task)
    await db.flush()
    await db.refresh(task)

    task_id = task.id
    file_path = dataset.file_path

    # Launch background training
    bg_task = asyncio.create_task(_execute_training(
        task_id=task_id,
        file_path=file_path,
        target_column=request_data["target_column"],
        model_type=request_data["model_type"],
        hyperparameters=request_data.get("hyperparameters", {}),
        test_size=request_data.get("test_size", 0.2),
        eval_metrics=request_data.get("eval_metrics", ["accuracy"]),
        cv_folds=cv_config.get("folds", 5) if cv_config.get("enabled", True) else 3,
        model_save_dir=str(settings.storage_models),
        class_weight=request_data.get("class_weight"),
    ))
    _running_tasks[task_id] = bg_task

    return task


async def _execute_training(
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
):
    """Background coroutine that runs training in a thread and updates DB."""
    # Update status to RUNNING
    async with async_session_factory() as db:
        result = await db.execute(select(TrainingTask).where(TrainingTask.id == task_id))
        task = result.scalar_one_or_none()
        if task:
            task.status = "RUNNING"
            task.started_at = datetime.now(timezone.utc)
            await db.commit()

    try:
        # Run training in thread pool
        loop = asyncio.get_event_loop()
        training_result = await loop.run_in_executor(
            _executor,
            _run_training_sync,
            task_id, file_path, target_column, model_type,
            hyperparameters, test_size, eval_metrics, cv_folds, model_save_dir, class_weight,
        )

        # Update with results
        async with async_session_factory() as db:
            result = await db.execute(select(TrainingTask).where(TrainingTask.id == task_id))
            task = result.scalar_one_or_none()
            if task:
                # Remove cv_folds detail from stored metrics to keep it clean
                stored_metrics = {
                    k: v for k, v in training_result["result_metrics"].items() if k != "cv_folds"
                }
                task.status = "SUCCESS"
                task.progress = 100.0
                task.result_metrics = stored_metrics
                task.model_path = training_result["model_path"]
                task.finished_at = datetime.now(timezone.utc)
                await db.commit()

        logger.info("Training task %s completed successfully", task_id)

    except Exception as e:
        logger.error("Training task %s failed: %s", task_id, str(e))
        # Log error to task-specific log file
        try:
            tl = TrainingLogger.__new__(TrainingLogger)
            settings = get_settings()
            tl.task_id = task_id
            tl.model_type = model_type
            tl.log_dir = settings.storage_logs
            tl.log_file = tl.log_dir / f"{task_id}.log"
            tl.metrics_file = tl.log_dir / f"{task_id}_metrics.json"
            tl._metrics_data = {"task_id": task_id, "model_type": model_type, "steps": []}
            tl.log("ERROR", f"Training failed: {str(e)}")
            tl.log_status("FAILED", str(e))
        except Exception:
            pass
        async with async_session_factory() as db:
            result = await db.execute(select(TrainingTask).where(TrainingTask.id == task_id))
            task = result.scalar_one_or_none()
            if task:
                task.status = "FAILED"
                task.error_message = str(e)
                task.finished_at = datetime.now(timezone.utc)
                await db.commit()

    finally:
        _running_tasks.pop(task_id, None)


async def get_training_status(task_id: str, db: AsyncSession) -> TrainingTask:
    """Retrieve the current state of a training task."""
    result = await db.execute(select(TrainingTask).where(TrainingTask.id == task_id))
    task = result.scalar_one_or_none()
    if task is None:
        raise HTTPException(status_code=404, detail="Training task not found")
    return task


async def stop_training(task_id: str, db: AsyncSession) -> TrainingTask:
    """Cancel a pending or running training task."""
    result = await db.execute(select(TrainingTask).where(TrainingTask.id == task_id))
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
    await db.flush()
    return task


async def list_training_tasks(
    db: AsyncSession,
    page: int = 1,
    page_size: int = 20,
    status_filter: str | None = None,
) -> dict:
    """Return a paginated list of training tasks."""
    stmt = select(TrainingTask)
    count_stmt = select(func.count(TrainingTask.id))

    if status_filter:
        stmt = stmt.where(TrainingTask.status == status_filter)
        count_stmt = count_stmt.where(TrainingTask.status == status_filter)

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


async def rename_training_task(task_id: str, name: str, db: AsyncSession) -> TrainingTask:
    """Rename a training task."""
    result = await db.execute(select(TrainingTask).where(TrainingTask.id == task_id))
    task = result.scalar_one_or_none()
    if task is None:
        raise HTTPException(status_code=404, detail="Training task not found")
    task.name = name
    await db.flush()
    return task


async def delete_training_task(task_id: str, db: AsyncSession) -> None:
    """Delete a training task (only if not RUNNING)."""
    result = await db.execute(select(TrainingTask).where(TrainingTask.id == task_id))
    task = result.scalar_one_or_none()
    if task is None:
        raise HTTPException(status_code=404, detail="Training task not found")
    if task.status == "RUNNING":
        raise HTTPException(status_code=422, detail="Cannot delete a running task. Stop it first.")
    await db.delete(task)
    await db.flush()


async def update_training_task_meta(
    task_id: str, notes: str | None, tags: list[str] | None, db: AsyncSession
) -> TrainingTask:
    """Update notes and/or tags of a training task."""
    result = await db.execute(select(TrainingTask).where(TrainingTask.id == task_id))
    task = result.scalar_one_or_none()
    if task is None:
        raise HTTPException(status_code=404, detail="Training task not found")
    if notes is not None:
        task.notes = notes
    if tags is not None:
        task.tags = tags
    await db.flush()
    return task

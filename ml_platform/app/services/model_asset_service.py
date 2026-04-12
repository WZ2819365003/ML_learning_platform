"""Unified model asset and deployment views over ML and DL records."""

from __future__ import annotations

from typing import Any

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.trainer import detect_task_type
from app.models.database import (
    DLModelDeployment,
    DLTrainingTask,
    Dataset,
    ModelDeployment,
    TrainingTask,
)
from app.utils.storage_paths import resolve_runtime_path

_BASE_URL = "http://127.0.0.1:8000"


def _safe_metric(metrics: dict[str, Any] | None, key: str) -> float | None:
    if not metrics:
        return None
    value = metrics.get(key)
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _build_metrics_summary(
    runtime_type: str,
    task_type: str,
    metrics: dict[str, Any] | None,
) -> dict[str, Any]:
    if not metrics:
        return {
            "primary_metric_name": None,
            "primary_metric_value": None,
            "secondary_metric_name": None,
            "secondary_metric_value": None,
        }

    if runtime_type == "dl":
        priority = (
            ["val_acc", "val_f1_macro", "val_auc_roc", "best_val_loss", "val_loss"]
            if task_type == "classification"
            else ["val_rmse", "val_mae", "val_r2", "val_mape", "best_val_loss", "val_loss"]
        )
    else:
        priority = (
            ["accuracy", "f1", "roc_auc", "precision", "recall", "cv_avg_accuracy", "cv_avg_f1"]
            if task_type == "classification"
            else ["rmse", "mae", "r2", "mape"]
        )

    selected: list[tuple[str, float]] = []
    for key in priority:
        value = _safe_metric(metrics, key)
        if value is None:
            continue
        selected.append((key, round(value, 6)))
        if len(selected) == 2:
            break

    if not selected:
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                selected.append((key, round(float(value), 6)))
                if len(selected) == 2:
                    break

    primary_name, primary_value = selected[0] if selected else (None, None)
    secondary_name, secondary_value = selected[1] if len(selected) > 1 else (None, None)
    return {
        "primary_metric_name": primary_name,
        "primary_metric_value": primary_value,
        "secondary_metric_name": secondary_name,
        "secondary_metric_value": secondary_value,
    }


async def list_model_assets(
    db: AsyncSession,
    page: int = 1,
    page_size: int = 20,
    runtime_type: str | None = None,
) -> dict[str, Any]:
    items: list[dict[str, Any]] = []

    ml_tasks: list[TrainingTask] = []
    if runtime_type in (None, "ml"):
        ml_stmt = (
            select(TrainingTask)
            .where(TrainingTask.status == "SUCCESS", TrainingTask.model_path.is_not(None))
            .order_by(TrainingTask.finished_at.desc(), TrainingTask.created_at.desc())
        )
        ml_tasks = (await db.execute(ml_stmt)).scalars().all()

    dl_tasks: list[DLTrainingTask] = []
    if runtime_type in (None, "dl"):
        dl_stmt = (
            select(DLTrainingTask)
            .where(DLTrainingTask.status == "SUCCESS", DLTrainingTask.model_path.is_not(None))
            .order_by(DLTrainingTask.finished_at.desc(), DLTrainingTask.created_at.desc())
        )
        dl_tasks = (await db.execute(dl_stmt)).scalars().all()

    dataset_ids = {task.dataset_id for task in ml_tasks}
    dataset_ids.update(task.dataset_id for task in dl_tasks)
    dataset_map: dict[str, Dataset] = {}
    if dataset_ids:
        dataset_result = await db.execute(select(Dataset).where(Dataset.id.in_(dataset_ids)))
        dataset_map = {dataset.id: dataset for dataset in dataset_result.scalars().all()}

    ml_counts: dict[str, int] = {}
    if ml_tasks:
        ml_count_result = await db.execute(
            select(ModelDeployment.task_id, func.count(ModelDeployment.id))
            .where(ModelDeployment.task_id.in_([task.id for task in ml_tasks]))
            .group_by(ModelDeployment.task_id)
        )
        ml_counts = {task_id: count for task_id, count in ml_count_result.all()}

    dl_counts: dict[str, int] = {}
    if dl_tasks:
        dl_count_result = await db.execute(
            select(DLModelDeployment.dl_task_id, func.count(DLModelDeployment.id))
            .where(DLModelDeployment.dl_task_id.in_([task.id for task in dl_tasks]))
            .group_by(DLModelDeployment.dl_task_id)
        )
        dl_counts = {task_id: count for task_id, count in dl_count_result.all()}

    for task in ml_tasks:
        dataset = dataset_map.get(task.dataset_id)
        model_path = resolve_runtime_path(task.model_path) if task.model_path else None
        if dataset is None or model_path is None or not model_path.exists():
            continue

        task_type = detect_task_type(task.model_type)
        items.append(
            {
                "asset_id": f"ml:{task.id}",
                "runtime_type": "ml",
                "task_id": task.id,
                "name": task.name,
                "dataset_id": task.dataset_id,
                "dataset_name": dataset.name,
                "model_type": task.model_type,
                "task_type": task_type,
                "status": task.status,
                "created_at": task.created_at,
                "finished_at": task.finished_at,
                "notes": task.notes,
                "tags": task.tags,
                "model_path": str(model_path),
                "deployable": True,
                "predictable": True,
                "deployment_count": ml_counts.get(task.id, 0),
                "metrics_summary": _build_metrics_summary("ml", task_type, task.result_metrics),
            }
        )

    for task in dl_tasks:
        dataset = dataset_map.get(task.dataset_id)
        model_path = resolve_runtime_path(task.model_path) if task.model_path else None
        if dataset is None or model_path is None or not model_path.exists():
            continue

        items.append(
            {
                "asset_id": f"dl:{task.id}",
                "runtime_type": "dl",
                "task_id": task.id,
                "name": task.name,
                "dataset_id": task.dataset_id,
                "dataset_name": dataset.name,
                "model_type": task.model_type,
                "task_type": task.task_type,
                "status": task.status,
                "created_at": task.created_at,
                "finished_at": task.finished_at,
                "notes": task.notes,
                "tags": task.tags,
                "model_path": str(model_path),
                "deployable": True,
                "predictable": False,
                "deployment_count": dl_counts.get(task.id, 0),
                "metrics_summary": _build_metrics_summary("dl", task.task_type, task.result_metrics),
            }
        )

    items.sort(key=lambda item: item["finished_at"] or item["created_at"], reverse=True)
    total = len(items)
    offset = (page - 1) * page_size
    paged_items = items[offset: offset + page_size]
    return {
        "items": paged_items,
        "total": total,
        "page": page,
        "page_size": page_size,
    }


async def list_unified_deployments(
    db: AsyncSession,
    page: int = 1,
    page_size: int = 20,
    runtime_type: str | None = None,
) -> dict[str, Any]:
    deployments: list[dict[str, Any]] = []

    ml_deployments: list[ModelDeployment] = []
    if runtime_type in (None, "ml"):
        ml_result = await db.execute(
            select(ModelDeployment).order_by(ModelDeployment.created_at.desc())
        )
        ml_deployments = ml_result.scalars().all()

    dl_deployments: list[DLModelDeployment] = []
    if runtime_type in (None, "dl"):
        dl_result = await db.execute(
            select(DLModelDeployment).order_by(DLModelDeployment.created_at.desc())
        )
        dl_deployments = dl_result.scalars().all()

    ml_task_map: dict[str, TrainingTask] = {}
    if ml_deployments:
        ml_tasks_result = await db.execute(
            select(TrainingTask).where(TrainingTask.id.in_([dep.task_id for dep in ml_deployments]))
        )
        ml_task_map = {task.id: task for task in ml_tasks_result.scalars().all()}

    dl_task_map: dict[str, DLTrainingTask] = {}
    if dl_deployments:
        dl_tasks_result = await db.execute(
            select(DLTrainingTask).where(DLTrainingTask.id.in_([dep.dl_task_id for dep in dl_deployments]))
        )
        dl_task_map = {task.id: task for task in dl_tasks_result.scalars().all()}

    for dep in ml_deployments:
        task = ml_task_map.get(dep.task_id)
        if task is None:
            continue
        deployments.append(
            {
                "deployment_id": dep.id,
                "runtime_type": "ml",
                "source_task_id": task.id,
                "source_asset_id": f"ml:{task.id}",
                "source_name": task.name,
                "model_type": task.model_type,
                "task_type": detect_task_type(task.model_type),
                "name": dep.name,
                "description": dep.description,
                "status": dep.status,
                "request_count": dep.request_count,
                "created_at": dep.created_at,
                "predict_url": f"{_BASE_URL}/inference/{dep.id}/predict",
                "result_url": f"{_BASE_URL}/inference/{dep.id}/result/{{job_id}}",
                "supports_result_polling": True,
            }
        )

    for dep in dl_deployments:
        task = dl_task_map.get(dep.dl_task_id)
        if task is None:
            continue
        deployments.append(
            {
                "deployment_id": dep.id,
                "runtime_type": "dl",
                "source_task_id": task.id,
                "source_asset_id": f"dl:{task.id}",
                "source_name": task.name,
                "model_type": task.model_type,
                "task_type": task.task_type,
                "name": dep.name,
                "description": dep.description,
                "status": dep.status,
                "request_count": dep.request_count,
                "created_at": dep.created_at,
                "predict_url": f"{_BASE_URL}/api/dl/deployments/{dep.id}/predict",
                "result_url": None,
                "supports_result_polling": False,
            }
        )

    deployments.sort(key=lambda item: item["created_at"], reverse=True)
    total = len(deployments)
    offset = (page - 1) * page_size
    paged_items = deployments[offset: offset + page_size]
    return {
        "deployments": paged_items,
        "total": total,
        "page": page,
        "page_size": page_size,
    }

"""Model management routes — list, detail, compare, delete saved models."""

from typing import Any
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.database import get_db, TrainingTask, Dataset

router = APIRouter(prefix="/models", tags=["Model Management"])


@router.get("/list")
async def list_models(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    model_type: str | None = Query(None, description="Filter by model type"),
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    """List all successfully trained models."""
    stmt = select(TrainingTask).where(TrainingTask.status == "SUCCESS")
    count_stmt = select(func.count(TrainingTask.id)).where(TrainingTask.status == "SUCCESS")

    if model_type:
        stmt = stmt.where(TrainingTask.model_type == model_type)
        count_stmt = count_stmt.where(TrainingTask.model_type == model_type)

    count_result = await db.execute(count_stmt)
    total = count_result.scalar_one()

    offset = (page - 1) * page_size
    stmt = stmt.order_by(TrainingTask.finished_at.desc()).offset(offset).limit(page_size)
    result = await db.execute(stmt)
    tasks = result.scalars().all()

    items = []
    for task in tasks:
        model_size = None
        if task.model_path and Path(task.model_path).exists():
            model_size = Path(task.model_path).stat().st_size

        items.append({
            "task_id": task.id,
            "dataset_id": task.dataset_id,
            "model_type": task.model_type,
            "hyperparameters": task.hyperparameters,
            "target_column": task.target_column,
            "result_metrics": task.result_metrics,
            "model_path": task.model_path,
            "model_size": model_size,
            "started_at": task.started_at.isoformat() if task.started_at else None,
            "finished_at": task.finished_at.isoformat() if task.finished_at else None,
        })

    return {"items": items, "total": total, "page": page, "page_size": page_size}


@router.get("/compare")
async def compare_models(
    task_ids: str = Query(..., description="Comma-separated task IDs to compare"),
    db: AsyncSession = Depends(get_db),
) -> list[dict[str, Any]]:
    """Compare metrics of multiple trained models."""
    ids = [tid.strip() for tid in task_ids.split(",") if tid.strip()]
    if len(ids) < 2:
        raise HTTPException(status_code=400, detail="Provide at least 2 task IDs")

    results = []
    for tid in ids:
        result = await db.execute(select(TrainingTask).where(TrainingTask.id == tid))
        task = result.scalar_one_or_none()
        if task is None:
            continue
        results.append({
            "task_id": task.id,
            "model_type": task.model_type,
            "hyperparameters": task.hyperparameters,
            "result_metrics": {k: v for k, v in (task.result_metrics or {}).items() if not k.startswith("cv_")},
            "status": task.status,
        })

    return results


@router.get("/{task_id}/detail")
async def model_detail(
    task_id: str,
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    """Get detailed info about a saved model."""
    result = await db.execute(select(TrainingTask).where(TrainingTask.id == task_id))
    task = result.scalar_one_or_none()
    if task is None:
        raise HTTPException(status_code=404, detail="Training task not found")

    # Get dataset info
    ds_result = await db.execute(select(Dataset).where(Dataset.id == task.dataset_id))
    dataset = ds_result.scalar_one_or_none()

    model_size = None
    if task.model_path and Path(task.model_path).exists():
        model_size = Path(task.model_path).stat().st_size

    return {
        "task_id": task.id,
        "model_type": task.model_type,
        "hyperparameters": task.hyperparameters,
        "target_column": task.target_column,
        "test_size": task.test_size,
        "eval_metrics": task.eval_metrics,
        "result_metrics": task.result_metrics,
        "model_path": task.model_path,
        "model_size": model_size,
        "status": task.status,
        "error_message": task.error_message,
        "started_at": task.started_at.isoformat() if task.started_at else None,
        "finished_at": task.finished_at.isoformat() if task.finished_at else None,
        "dataset": {
            "id": dataset.id,
            "name": dataset.name,
            "row_count": dataset.row_count,
            "column_count": dataset.column_count,
        } if dataset else None,
    }


@router.delete("/{task_id}")
async def delete_model(
    task_id: str,
    db: AsyncSession = Depends(get_db),
) -> dict:
    """Delete a saved model file (keeps the task record)."""
    result = await db.execute(select(TrainingTask).where(TrainingTask.id == task_id))
    task = result.scalar_one_or_none()
    if task is None:
        raise HTTPException(status_code=404, detail="Training task not found")

    if task.model_path:
        p = Path(task.model_path)
        if p.exists():
            p.unlink()
        task.model_path = None
        await db.flush()

    return {"message": "Model deleted", "task_id": task_id}

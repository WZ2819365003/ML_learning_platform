"""Model management routes — list, detail, compare, delete saved models."""

from typing import Any
from pathlib import Path

from fastapi import APIRouter, Body, Depends, HTTPException, Query
from sqlalchemy import select, func
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.database import get_db, TrainingTask, Dataset, ModelTagLibrary
from app.models.schemas import (
    ModelAssetListResponse,
    PredictionRequest,
    PredictionResponse,
)
from app.services.prediction_service import predict_rows
from app.services.model_asset_service import list_model_assets

router = APIRouter(prefix="/models", tags=["Model Management"])


@router.get("/assets", response_model=ModelAssetListResponse)
async def list_model_assets_route(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    runtime_type: str | None = Query(default=None, pattern="^(ml|dl)$"),
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    """List ML and DL models through a unified asset view."""
    return await list_model_assets(db=db, page=page, page_size=page_size, runtime_type=runtime_type)


@router.get("/list")
async def list_models(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=500),
    model_type: str | None = Query(None, description="Filter by model type"),
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    """List all successfully trained models."""
    stmt = select(TrainingTask).where(
        TrainingTask.status == "SUCCESS",
        TrainingTask.model_path.is_not(None),
    )
    count_stmt = select(func.count(TrainingTask.id)).where(
        TrainingTask.status == "SUCCESS",
        TrainingTask.model_path.is_not(None),
    )

    if model_type:
        stmt = stmt.where(TrainingTask.model_type == model_type)
        count_stmt = count_stmt.where(TrainingTask.model_type == model_type)

    count_result = await db.execute(count_stmt)
    total = count_result.scalar_one()

    offset = (page - 1) * page_size
    stmt = stmt.order_by(TrainingTask.finished_at.desc()).offset(offset).limit(page_size)
    result = await db.execute(stmt)
    tasks = result.scalars().all()

    dataset_map: dict[str, Dataset] = {}
    dataset_ids = {task.dataset_id for task in tasks}
    if dataset_ids:
        dataset_result = await db.execute(select(Dataset).where(Dataset.id.in_(dataset_ids)))
        dataset_map = {dataset.id: dataset for dataset in dataset_result.scalars().all()}

    items = []
    for task in tasks:
        dataset = dataset_map.get(task.dataset_id)
        model_file = Path(task.model_path) if task.model_path else None
        dataset_file = Path(dataset.file_path) if dataset else None

        if dataset is None or dataset_file is None or not dataset_file.exists():
            continue
        if model_file is None or not model_file.exists():
            continue

        model_size = model_file.stat().st_size

        items.append({
            "task_id": task.id,
            "name": task.name,
            "dataset_id": task.dataset_id,
            "dataset_name": dataset.name,
            "model_type": task.model_type,
            "hyperparameters": task.hyperparameters,
            "target_column": task.target_column,
            "result_metrics": task.result_metrics,
            "model_path": task.model_path,
            "model_size": model_size,
            "notes": task.notes,
            "tags": task.tags,
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
        dataset_result = await db.execute(select(Dataset).where(Dataset.id == task.dataset_id))
        dataset = dataset_result.scalar_one_or_none()
        results.append({
            "task_id": task.id,
            "dataset_name": dataset.name if dataset else None,
            "model_type": task.model_type,
            "hyperparameters": task.hyperparameters,
            "result_metrics": {k: v for k, v in (task.result_metrics or {}).items() if not k.startswith("cv_")},
            "status": task.status,
        })

    return results


# ---------------------------------------------------------------------------
# Tag library — MUST be declared before /{task_id} parametric routes
# ---------------------------------------------------------------------------

@router.get("/tags")
async def list_tag_library(
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    """Return all tags in the shared tag library, sorted alphabetically."""
    result = await db.execute(
        select(ModelTagLibrary.name).order_by(ModelTagLibrary.name.asc())
    )
    tags = result.scalars().all()
    return {"tags": list(tags)}


@router.post("/tags/sync")
async def sync_tags_to_library(
    tags: list[str] = Body(..., embed=True),
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    """Upsert tag names into the library (adds new ones, ignores existing)."""
    cleaned = [t.strip() for t in tags if t.strip()]
    for name in cleaned:
        stmt = sqlite_insert(ModelTagLibrary).values(name=name).on_conflict_do_nothing(index_elements=["name"])
        await db.execute(stmt)
    await db.flush()
    result = await db.execute(
        select(ModelTagLibrary.name).order_by(ModelTagLibrary.name.asc())
    )
    return {"tags": list(result.scalars().all())}


@router.delete("/tags/{tag_name}")
async def delete_tag_from_library(
    tag_name: str,
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    """Remove a tag from the shared library."""
    result = await db.execute(
        select(ModelTagLibrary).where(ModelTagLibrary.name == tag_name)
    )
    tag = result.scalar_one_or_none()
    if tag is None:
        raise HTTPException(status_code=404, detail=f"Tag '{tag_name}' not found")
    await db.delete(tag)
    await db.flush()
    all_tags = await db.execute(
        select(ModelTagLibrary.name).order_by(ModelTagLibrary.name.asc())
    )
    return {"tags": list(all_tags.scalars().all()), "deleted": tag_name}


# ---------------------------------------------------------------------------
# Parametric routes — must come AFTER all literal routes above
# ---------------------------------------------------------------------------

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

    ds_result = await db.execute(select(Dataset).where(Dataset.id == task.dataset_id))
    dataset = ds_result.scalar_one_or_none()

    model_size = None
    if task.model_path and Path(task.model_path).exists():
        model_size = Path(task.model_path).stat().st_size

    return {
        "task_id": task.id,
        "name": task.name,
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
        "notes": task.notes,
        "tags": task.tags,
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


@router.post("/{task_id}/predict", response_model=PredictionResponse)
async def predict_model(
    task_id: str,
    request: PredictionRequest,
    db: AsyncSession = Depends(get_db),
) -> PredictionResponse:
    """Run prediction against a saved model using inline JSON rows."""
    result = await predict_rows(
        task_id=task_id,
        rows=request.rows,
        include_probabilities=request.include_probabilities,
        db=db,
    )
    return PredictionResponse(**result)

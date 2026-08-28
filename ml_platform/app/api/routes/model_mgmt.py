"""Model management routes — list, detail, compare, delete saved models."""

from typing import Any

from fastapi import APIRouter, Body, Depends, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sqlalchemy import select, func
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.core.auth import current_username_from_authorization, owner_scope_username
from app.core.ownership import ensure_task_owner
from app.models.database import (
    Dataset,
    DLTrainingTask,
    ModelTagLibrary,
    TrainingTask,
    get_db,
)
from app.models.schemas import (
    ModelAssetListResponse,
    PredictionRequest,
    PredictionResponse,
)
from app.services.prediction_service import predict_rows
from app.services.model_asset_service import list_model_assets
from app.services.object_storage import restore_dataset_file, restore_model_bundle
from app.utils.storage_paths import resolve_runtime_path

router = APIRouter(prefix="/models", tags=["Model Management"])


def _managed_model_file(raw_path: str):
    """Resolve a model artifact without allowing paths outside storage/models."""
    file_path = resolve_runtime_path(raw_path)
    model_root = get_settings().storage_models.resolve()
    try:
        file_path.resolve().relative_to(model_root)
    except ValueError:
        raise HTTPException(status_code=404, detail="Model file not found on disk")
    if not file_path.is_file():
        restore_model_bundle(raw_path)
    if not file_path.is_file():
        raise HTTPException(status_code=404, detail="Model file not found on disk")
    return file_path


@router.get("/assets", response_model=ModelAssetListResponse)
async def list_model_assets_route(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    runtime_type: str | None = Query(default=None, pattern="^(ml|dl)$"),
    db: AsyncSession = Depends(get_db),
    username: str = Depends(current_username_from_authorization),
) -> dict[str, Any]:
    """List ML and DL models through a unified asset view."""
    return await list_model_assets(
        db=db,
        page=page,
        page_size=page_size,
        runtime_type=runtime_type,
        owner_username=owner_scope_username(username),
    )


@router.get("/list")
async def list_models(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=500),
    model_type: str | None = Query(None, description="Filter by model type"),
    db: AsyncSession = Depends(get_db),
    username: str = Depends(current_username_from_authorization),
) -> dict[str, Any]:
    """List all successfully trained models."""
    owner_username = owner_scope_username(username)
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
    if owner_username:
        stmt = stmt.where(TrainingTask.owner_username == owner_username)
        count_stmt = count_stmt.where(TrainingTask.owner_username == owner_username)

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
        model_file = restore_model_bundle(task.model_path) if task.model_path else None
        dataset_file = restore_dataset_file(dataset.id, dataset.file_path) if dataset else None

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
            "model_path": str(model_file),
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
    username: str = Depends(current_username_from_authorization),
) -> list[dict[str, Any]]:
    """Compare metrics of multiple trained models."""
    ids = [tid.strip() for tid in task_ids.split(",") if tid.strip()]
    if len(ids) < 2:
        raise HTTPException(status_code=400, detail="Provide at least 2 task IDs")

    results = []
    owner_username = owner_scope_username(username)
    for tid in ids:
        task_stmt = select(TrainingTask).where(TrainingTask.id == tid)
        if owner_username:
            task_stmt = task_stmt.where(TrainingTask.owner_username == owner_username)
        result = await db.execute(task_stmt)
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
    """Return all tags in the shared tag library with dimension/color, grouped by dimension."""
    result = await db.execute(
        select(ModelTagLibrary).order_by(ModelTagLibrary.dimension.asc(), ModelTagLibrary.name.asc())
    )
    rows = result.scalars().all()
    tags = [{"name": t.name, "dimension": t.dimension, "color": t.color} for t in rows]
    grouped: dict[str, list[dict]] = {}
    for t in tags:
        dim = t["dimension"] or "其他"
        grouped.setdefault(dim, []).append(t)
    return {"tags": tags, "grouped": grouped}


class _TagCreateBody(BaseModel):
    name: str
    dimension: str | None = None
    color: str | None = None


@router.post("/tags/create")
async def create_tag(
    body: _TagCreateBody,
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    """Create a tag with name, dimension, and color (upsert by name)."""
    name = body.name.strip()
    if not name:
        raise HTTPException(status_code=422, detail="name is required")
    stmt = (
        sqlite_insert(ModelTagLibrary)
        .values(name=name, dimension=body.dimension or None, color=body.color or None)
        .on_conflict_do_nothing(index_elements=["name"])
    )
    await db.execute(stmt)
    await db.flush()
    result = await db.execute(select(ModelTagLibrary).where(ModelTagLibrary.name == name))
    tag = result.scalar_one_or_none()
    return {"name": tag.name, "dimension": tag.dimension, "color": tag.color}


@router.post("/tags/sync")
async def sync_tags_to_library(
    tags: list[str] = Body(..., embed=True),
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    """Upsert tag names into the library (adds new ones without dimension, ignores existing)."""
    cleaned = [t.strip() for t in tags if t.strip()]
    for name in cleaned:
        stmt = sqlite_insert(ModelTagLibrary).values(name=name).on_conflict_do_nothing(index_elements=["name"])
        await db.execute(stmt)
    await db.flush()
    result = await db.execute(
        select(ModelTagLibrary).order_by(ModelTagLibrary.dimension.asc(), ModelTagLibrary.name.asc())
    )
    rows = result.scalars().all()
    return {"tags": [{"name": t.name, "dimension": t.dimension, "color": t.color} for t in rows]}


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
    all_result = await db.execute(
        select(ModelTagLibrary).order_by(ModelTagLibrary.dimension.asc(), ModelTagLibrary.name.asc())
    )
    rows = all_result.scalars().all()
    tags = [{"name": t.name, "dimension": t.dimension, "color": t.color} for t in rows]
    return {"tags": tags, "deleted": tag_name}


# ---------------------------------------------------------------------------
# Parametric routes — must come AFTER all literal routes above
# ---------------------------------------------------------------------------

@router.get("/{task_id}/detail")
async def model_detail(
    task_id: str,
    db: AsyncSession = Depends(get_db),
    username: str = Depends(current_username_from_authorization),
) -> dict[str, Any]:
    """Get detailed info about a saved model."""
    owner_username = owner_scope_username(username)
    task_stmt = select(TrainingTask).where(TrainingTask.id == task_id)
    if owner_username:
        task_stmt = task_stmt.where(TrainingTask.owner_username == owner_username)
    result = await db.execute(task_stmt)
    task = result.scalar_one_or_none()
    if task is None:
        # V3 runs and purged legacy rows may still have a valid model artifact
        # and metrics log. Reuse the visualization resolver so result pages can
        # show detail metadata for those recovered tasks as well.
        from app.services.resolver import resolve_task_and_dataset

        await ensure_task_owner(db, task_id, owner_username)
        recovered_task, dataset = await resolve_task_and_dataset(task_id, db)
        model_path = recovered_task.model_path
        model_size = None
        if model_path:
            model_file = resolve_runtime_path(model_path)
            if model_file.exists():
                model_size = model_file.stat().st_size

        is_dl = isinstance(recovered_task, DLTrainingTask)
        train_config = getattr(recovered_task, "train_config", None) or {}

        return {
            "task_id": recovered_task.id,
            "family": "dl" if is_dl else "ml",
            "name": getattr(recovered_task, "name", None),
            "model_type": recovered_task.model_type,
            "hyperparameters": getattr(recovered_task, "hyperparameters", None),
            "target_column": recovered_task.target_column,
            "test_size": getattr(
                recovered_task, "test_size", train_config.get("test_size", 0.2)
            ),
            "eval_metrics": getattr(recovered_task, "eval_metrics", None),
            "result_metrics": recovered_task.result_metrics,
            "model_path": str(resolve_runtime_path(model_path)) if model_path else None,
            "model_size": model_size,
            "status": recovered_task.status,
            "error_message": None,
            "notes": None,
            "tags": None,
            "started_at": None,
            "finished_at": None,
            "dataset": {
                "id": dataset.id,
                "name": dataset.name,
                "row_count": dataset.row_count,
                "column_count": dataset.column_count,
            } if dataset else None,
        }

    ds_result = await db.execute(select(Dataset).where(Dataset.id == task.dataset_id))
    dataset = ds_result.scalar_one_or_none()

    model_size = None
    if task.model_path:
        model_file = resolve_runtime_path(task.model_path)
        if model_file.exists():
            model_size = model_file.stat().st_size

    return {
        "task_id": task.id,
        "name": task.name,
        "model_type": task.model_type,
        "hyperparameters": task.hyperparameters,
        "target_column": task.target_column,
        "test_size": task.test_size,
        "eval_metrics": task.eval_metrics,
        "result_metrics": task.result_metrics,
        "model_path": str(resolve_runtime_path(task.model_path)) if task.model_path else None,
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


@router.get("/{task_id}/download")
async def download_model_file(
    task_id: str,
    db: AsyncSession = Depends(get_db),
    username: str = Depends(current_username_from_authorization),
) -> FileResponse:
    """Download a saved ML (.joblib) or DL (.pt) model file."""
    owner_username = owner_scope_username(username)
    task_stmt = select(TrainingTask).where(TrainingTask.id == task_id)
    if owner_username:
        task_stmt = task_stmt.where(TrainingTask.owner_username == owner_username)
    result = await db.execute(task_stmt)
    task = result.scalar_one_or_none()
    if task is None:
        dl_stmt = select(DLTrainingTask).where(DLTrainingTask.id == task_id)
        if owner_username:
            dl_stmt = dl_stmt.where(DLTrainingTask.owner_username == owner_username)
        result = await db.execute(
            dl_stmt
        )
        task = result.scalar_one_or_none()
    if task is None:
        raise HTTPException(status_code=404, detail="Training task not found")
    if not task.model_path:
        raise HTTPException(status_code=404, detail="Model file not found on disk")

    file_path = _managed_model_file(task.model_path)
    extension = ".pt" if isinstance(task, DLTrainingTask) else ".joblib"
    safe_name = f"{task.model_type or 'model'}_{task_id[:8]}{extension}"
    return FileResponse(
        path=str(file_path),
        filename=safe_name,
        media_type="application/octet-stream",
    )


@router.delete("/{task_id}")
async def delete_model(
    task_id: str,
    db: AsyncSession = Depends(get_db),
    username: str = Depends(current_username_from_authorization),
) -> dict:
    """Delete a saved model file (keeps the task record)."""
    owner_username = owner_scope_username(username)
    task_stmt = select(TrainingTask).where(TrainingTask.id == task_id)
    if owner_username:
        task_stmt = task_stmt.where(TrainingTask.owner_username == owner_username)
    result = await db.execute(task_stmt)
    task = result.scalar_one_or_none()
    if task is None:
        raise HTTPException(status_code=404, detail="Training task not found")

    if task.model_path:
        p = resolve_runtime_path(task.model_path)
        if p.exists():
            p.unlink()
        task.model_path = None
        await db.flush()

    return {"message": "Model deleted", "task_id": task_id}


@router.post(
    "/{task_id}/predict",
    response_model=PredictionResponse,
    response_model_exclude_none=True,
)
async def predict_model(
    task_id: str,
    request: PredictionRequest,
    db: AsyncSession = Depends(get_db),
    username: str = Depends(current_username_from_authorization),
) -> PredictionResponse:
    """Run prediction against a saved model using inline JSON rows."""
    result = await predict_rows(
        task_id=task_id,
        rows=request.rows,
        include_probabilities=request.include_probabilities,
        db=db,
        owner_username=owner_scope_username(username),
    )
    return PredictionResponse(**result)

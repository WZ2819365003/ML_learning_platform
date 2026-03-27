"""Data management routes -- dataset upload, preview, list, and delete."""

from fastapi import APIRouter, Depends, File, Query, UploadFile, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.database import get_db
from app.models.schemas import DatasetListResponse, DatasetPreview, DatasetResponse
from app.services.data_service import (
    delete_dataset,
    get_dataset_preview,
    list_datasets,
    upload_dataset,
)

router = APIRouter(prefix="/data", tags=["Data Management"])


@router.post("/upload", response_model=DatasetResponse, status_code=status.HTTP_201_CREATED)
async def upload_dataset_route(
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
) -> DatasetResponse:
    """Upload a new dataset file."""
    return await upload_dataset(file=file, db=db)


@router.get("/{dataset_id}/preview", response_model=DatasetPreview)
async def preview_dataset(
    dataset_id: str,
    rows: int = Query(default=100, le=1000),
    db: AsyncSession = Depends(get_db),
) -> DatasetPreview:
    """Return a preview (first N rows) of the specified dataset."""
    return await get_dataset_preview(dataset_id=dataset_id, rows=rows, db=db)


@router.get("/list", response_model=DatasetListResponse)
async def list_datasets_route(
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
) -> DatasetListResponse:
    """List all uploaded datasets with pagination."""
    return await list_datasets(page=page, page_size=page_size, db=db)


@router.delete("/{dataset_id}")
async def delete_dataset_route(
    dataset_id: str,
    db: AsyncSession = Depends(get_db),
) -> dict:
    """Delete a dataset by ID."""
    await delete_dataset(dataset_id=dataset_id, db=db)
    return {"message": "Dataset deleted", "id": dataset_id}

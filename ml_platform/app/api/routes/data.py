"""Data management routes -- dataset upload, preview, list, delete, and versioning."""

from typing import Any

from fastapi import APIRouter, Body, Depends, File, Query, UploadFile, status
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.database import get_db
from app.models.schemas import DatasetListResponse, DatasetPreview, DatasetResponse
from app.services.data_service import (
    delete_dataset,
    get_correlation,
    get_dataset_preview,
    get_target_distribution,
    list_datasets,
    upload_dataset,
)


class CreateVersionRequest(BaseModel):
    description: str | None = None

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


@router.get("/{dataset_id}/correlation")
async def correlation_route(
    dataset_id: str,
    method: str = Query(default="pearson", pattern="^(pearson|spearman|kendall)$"),
    db: AsyncSession = Depends(get_db),
) -> dict:
    """Return correlation matrix for all numeric columns."""
    return await get_correlation(dataset_id=dataset_id, method=method, db=db)


@router.get("/{dataset_id}/target_distribution")
async def target_distribution_route(
    dataset_id: str,
    target_column: str = Query(...),
    db: AsyncSession = Depends(get_db),
) -> dict:
    """Return distribution stats and value counts for the target column."""
    return await get_target_distribution(
        dataset_id=dataset_id, target_column=target_column, db=db
    )


@router.delete("/{dataset_id}")
async def delete_dataset_route(
    dataset_id: str,
    db: AsyncSession = Depends(get_db),
) -> dict:
    """Delete a dataset by ID."""
    await delete_dataset(dataset_id=dataset_id, db=db)
    return {"message": "Dataset deleted", "id": dataset_id}


# ---------------------------------------------------------------------------
# V3 Dataset Versioning
# ---------------------------------------------------------------------------

@router.get("/{dataset_id}/versions", summary="List dataset versions")
async def list_dataset_versions(
    dataset_id: str,
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    """Return all versioned snapshots for a dataset, newest first."""
    from app.services.data_version_service import list_versions
    return await list_versions(dataset_id, db)


@router.post("/{dataset_id}/versions", summary="Create a dataset version snapshot", status_code=201)
async def create_dataset_version(
    dataset_id: str,
    body: CreateVersionRequest = Body(default=CreateVersionRequest()),
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    """
    Snapshot the current dataset file as a new version.
    The snapshot is copied to MinIO at datasets/{dataset_id}/{version}/data.<ext>
    """
    from app.services.data_version_service import create_version
    return await create_version(dataset_id, db, description=body.description)


@router.get("/{dataset_id}/versions/{version_id}", summary="Get a specific version")
async def get_dataset_version(
    dataset_id: str,
    version_id: str,
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    from app.services.data_version_service import get_version
    return await get_version(dataset_id, version_id, db)

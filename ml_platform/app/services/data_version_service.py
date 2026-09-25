"""
Dataset Version Service — create and query versioned snapshots of datasets.

Storage layout in MinIO (or local fallback):
  datasets/{dataset_id}/v{N}/{filename}

Each version:
  - captures the current file_path of the Dataset
  - computes a SHA-256 content hash for de-duplication
  - records row_count and schema_hash
  - stores in MinIO if S3 is enabled, otherwise just records the existing path
"""

from __future__ import annotations

import hashlib
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import HTTPException
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.database import Dataset, DatasetVersion

logger = logging.getLogger(__name__)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _sha256_file(path: str | Path) -> str:
    """Compute SHA-256 of a file's contents."""
    h = hashlib.sha256()
    try:
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
    except OSError:
        return ""
    return h.hexdigest()


def _serialize_version(v: DatasetVersion) -> dict[str, Any]:
    return {
        "id": v.id,
        "dataset_id": v.dataset_id,
        "version": v.version,
        "file_uri": v.file_uri,
        "row_count": v.row_count,
        "schema_hash": v.schema_hash,
        "description": v.description,
        "parent_version_id": v.parent_version_id,
        "created_at": v.created_at.isoformat() if v.created_at else None,
    }


async def _get_dataset_or_404(
    dataset_id: str,
    db: AsyncSession,
    owner_username: str | None = None,
) -> Dataset:
    stmt = select(Dataset).where(Dataset.id == dataset_id)
    if owner_username:
        stmt = stmt.where(Dataset.owner_username == owner_username)
    result = await db.execute(stmt)
    ds = result.scalar_one_or_none()
    if ds is None:
        raise HTTPException(status_code=404, detail=f"Dataset {dataset_id!r} not found")
    return ds


async def list_versions(
    dataset_id: str,
    db: AsyncSession,
    owner_username: str | None = None,
) -> dict[str, Any]:
    """List all versions for a dataset, ordered by version number descending."""
    await _get_dataset_or_404(dataset_id, db, owner_username=owner_username)

    rows = await db.execute(
        select(DatasetVersion)
        .where(DatasetVersion.dataset_id == dataset_id)
        .order_by(DatasetVersion.version.desc())
    )
    versions = rows.scalars().all()

    total_result = await db.execute(
        select(func.count(DatasetVersion.id))
        .where(DatasetVersion.dataset_id == dataset_id)
    )
    total = total_result.scalar_one()

    return {
        "dataset_id": dataset_id,
        "total": total,
        "items": [_serialize_version(v) for v in versions],
    }


async def create_version(
    dataset_id: str,
    db: AsyncSession,
    description: str | None = None,
    owner_username: str | None = None,
) -> dict[str, Any]:
    """
    Snapshot the current dataset as a new version.

    - Computes SHA-256 of the current file.
    - If the hash matches the latest version, raises a 409 (no changes).
    - Attempts to upload a copy to MinIO; falls back to recording the local path.
    """
    from app.config import get_settings

    ds = await _get_dataset_or_404(dataset_id, db, owner_username=owner_username)

    # Determine next version number
    max_result = await db.execute(
        select(func.max(DatasetVersion.version))
        .where(DatasetVersion.dataset_id == dataset_id)
    )
    max_ver = max_result.scalar_one() or 0
    next_ver = max_ver + 1

    # Compute hash of current file
    current_path = Path(ds.file_path)
    if not current_path.exists():
        # Try to resolve relative to project root
        settings = get_settings()
        current_path = settings.project_root / ds.file_path
    if not current_path.exists():
        raise HTTPException(status_code=404, detail="Dataset file not found on disk")

    content_hash = _sha256_file(current_path)

    # De-duplicate: if latest version has same hash, reject
    if max_ver > 0:
        last_result = await db.execute(
            select(DatasetVersion)
            .where(DatasetVersion.dataset_id == dataset_id)
            .order_by(DatasetVersion.version.desc())
            .limit(1)
        )
        last_ver = last_result.scalar_one_or_none()
        if last_ver and last_ver.schema_hash == content_hash:
            raise HTTPException(
                status_code=409,
                detail=f"Version v{max_ver} already has the same content (hash: {content_hash[:12]}…). No changes detected.",
            )

    # Compute row count
    row_count = None
    try:
        ext = current_path.suffix.lower()
        if ext == ".csv":
            import pandas as pd
            df = pd.read_csv(current_path, nrows=None)
            row_count = len(df)
        elif ext in (".xlsx", ".xls"):
            import pandas as pd
            df = pd.read_excel(current_path)
            row_count = len(df)
        elif ext == ".parquet":
            import pandas as pd
            df = pd.read_parquet(current_path)
            row_count = len(df)
    except Exception as e:
        logger.warning("Could not compute row_count for version: %s", e)

    # Try to upload snapshot to MinIO
    file_uri = str(current_path)  # fallback: local path
    try:
        from app.services.object_storage import upload_file
        settings = get_settings()
        if settings.s3_enabled:
            dest_key = f"datasets/{dataset_id}/v{next_ver}/{current_path.name}"
            result_uri = upload_file(current_path, dest_key)
            if result_uri:
                file_uri = result_uri
                logger.info("Version snapshot uploaded to %s", file_uri)
    except Exception as e:
        logger.warning("MinIO upload for version failed (%s); using local path", e)

    # Find parent version id
    last_result = await db.execute(
        select(DatasetVersion)
        .where(DatasetVersion.dataset_id == dataset_id)
        .order_by(DatasetVersion.version.desc())
        .limit(1)
    )
    parent = last_result.scalar_one_or_none()

    dv = DatasetVersion(
        dataset_id=dataset_id,
        version=next_ver,
        file_uri=file_uri,
        row_count=row_count,
        schema_hash=content_hash,
        description=description,
        parent_version_id=parent.id if parent else None,
    )
    db.add(dv)
    await db.flush()
    await db.refresh(dv)
    await db.commit()

    logger.info("Dataset %s: created version v%d (hash=%s)", dataset_id, next_ver, content_hash[:12])
    return _serialize_version(dv)


async def get_version(
    dataset_id: str,
    version_id: str,
    db: AsyncSession,
    owner_username: str | None = None,
) -> dict[str, Any]:
    await _get_dataset_or_404(dataset_id, db, owner_username=owner_username)
    result = await db.execute(
        select(DatasetVersion).where(
            DatasetVersion.id == version_id,
            DatasetVersion.dataset_id == dataset_id,
        )
    )
    v = result.scalar_one_or_none()
    if v is None:
        raise HTTPException(status_code=404, detail=f"Version {version_id!r} not found")
    return _serialize_version(v)

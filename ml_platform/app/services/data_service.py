"""Data upload and management service.

Handles dataset uploading, preview, listing, and deletion with
async SQLAlchemy sessions and pandas-based file inspection.
"""

from __future__ import annotations

import hashlib
import logging
import shutil
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import HTTPException, UploadFile
from sqlalchemy import func, select

from app.config import get_settings
from app.models.database import AsyncSession, Dataset
from app.utils.storage_paths import resolve_runtime_path, to_portable_storage_path
from app.utils.file_utils import generate_unique_filename

logger = logging.getLogger(__name__)

_ALLOWED_EXTENSIONS = {".csv", ".parquet", ".xlsx"}


def _compute_content_digest(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


async def _find_existing_dataset_by_content(
    db: AsyncSession,
    content: bytes,
    file_size: int,
) -> Dataset | None:
    digest = _compute_content_digest(content)
    result = await db.execute(select(Dataset).where(Dataset.file_size == file_size))
    for dataset in result.scalars():
        path = resolve_runtime_path(dataset.file_path)
        if not path.exists():
            continue
        try:
            if _compute_content_digest(path.read_bytes()) == digest:
                return dataset
        except OSError:
            logger.warning("Failed to inspect dataset file for dedup: %s", path, exc_info=True)
    return None


def _validate_extension(filename: str) -> str:
    """Return the lowered file extension or raise 400."""
    ext = Path(filename).suffix.lower()
    if ext not in _ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported file type '{ext}'. "
                f"Allowed: {', '.join(sorted(_ALLOWED_EXTENSIONS))}"
            ),
        )
    return ext


def _read_dataframe(path: Path, ext: str, nrows: int | None = None) -> pd.DataFrame:
    """Read a file into a DataFrame based on its extension."""
    if ext == ".csv":
        return pd.read_csv(path, nrows=nrows)
    elif ext == ".parquet":
        df = pd.read_parquet(path)
        if nrows is not None:
            df = df.head(nrows)
        return df
    elif ext == ".xlsx":
        return pd.read_excel(path, nrows=nrows)
    else:
        raise HTTPException(status_code=400, detail=f"Cannot read extension '{ext}'")


def _build_columns_info(df: pd.DataFrame) -> dict[str, dict[str, Any]]:
    """Build per-column metadata used by preview and task target selection."""
    info: dict[str, dict[str, Any]] = {}
    total_rows = len(df)
    for col in df.columns:
        missing = int(df[col].isna().sum())
        unique_count = int(df[col].nunique(dropna=True))
        value_counts = df[col].value_counts(dropna=True)
        info[str(col)] = {
            "dtype": str(df[col].dtype),
            "missing_count": missing,
            "missing_rate": round(missing / total_rows, 4) if total_rows > 0 else 0.0,
            "unique_count": unique_count,
            "unique_rate": round(unique_count / total_rows, 4) if total_rows > 0 else 0.0,
            "min_class_count": int(value_counts.min()) if unique_count > 0 else 0,
        }
    return info


def _merge_columns_info_with_sample(
    stored_info: dict[str, Any] | None,
    sample_info: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Return stored metadata with missing preview-only stats filled from a sample."""
    merged: dict[str, dict[str, Any]] = {}
    for col, sample_meta in sample_info.items():
        stored_meta = stored_info.get(col, {}) if isinstance(stored_info, dict) else {}
        merged[col] = {**sample_meta, **stored_meta}
    return merged


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


async def upload_dataset(file: UploadFile, db: AsyncSession) -> Dataset:
    """Validate, persist, and catalogue an uploaded dataset file.

    Steps:
        1. Validate file extension and size.
        2. Save to the configured upload directory with a UUID-prefixed name.
        3. Read with pandas to extract row/column metadata.
        4. Create a ``Dataset`` record in the database.

    Returns the newly-created ``Dataset`` ORM instance.
    """
    settings = get_settings()

    # -- extension check --
    original_name = file.filename or "unnamed"
    ext = _validate_extension(original_name)

    # -- size check (read content once) --
    content = await file.read()
    file_size = len(content)
    if file_size > settings.max_upload_size:
        raise HTTPException(
            status_code=400,
            detail=(
                f"File size ({file_size:,} bytes) exceeds the maximum "
                f"allowed size ({settings.max_upload_size:,} bytes)."
            ),
        )

    # -- dedup: reuse existing dataset if file content is identical --
    existing_dataset = await _find_existing_dataset_by_content(db, content=content, file_size=file_size)
    if existing_dataset is not None:
        logger.info("Reusing existing dataset %s for '%s' by content digest", existing_dataset.id, original_name)
        return existing_dataset

    # -- save to disk --
    unique_name = generate_unique_filename(original_name)
    dest_path = settings.storage_uploads / unique_name
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    dest_path.write_bytes(content)

    logger.info("Saved uploaded file to %s (%s bytes)", dest_path, file_size)

    # -- read with pandas for metadata --
    try:
        df = _read_dataframe(dest_path, ext)
    except Exception as exc:
        # Clean up the file if pandas cannot parse it
        dest_path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=400,
            detail=f"Failed to parse uploaded file: {exc}",
        ) from exc

    columns_info = _build_columns_info(df)

    # -- persist to DB --
    dataset = Dataset(
        name=original_name,
        file_path=to_portable_storage_path(dest_path),
        file_size=file_size,
        row_count=len(df),
        column_count=len(df.columns),
        columns_info=columns_info,
    )
    db.add(dataset)
    await db.flush()
    await db.refresh(dataset)

    logger.info("Created dataset record %s for '%s'", dataset.id, original_name)
    return dataset


async def get_dataset_preview(
    dataset_id: str,
    db: AsyncSession,
    rows: int = 100,
) -> dict[str, Any]:
    """Return a preview of the dataset including sample rows and statistics.

    Raises ``HTTPException(404)`` when the dataset is not found.
    """
    result = await db.execute(select(Dataset).where(Dataset.id == dataset_id))
    dataset: Dataset | None = result.scalar_one_or_none()
    if dataset is None:
        raise HTTPException(status_code=404, detail="Dataset not found")

    file_path = resolve_runtime_path(dataset.file_path)
    ext = file_path.suffix.lower()

    df = _read_dataframe(file_path, ext, nrows=rows)

    sample_columns_info = _build_columns_info(df)

    # -- statistics --
    statistics: dict[str, dict[str, Any]] = {}
    for col in df.columns:
        col_stats: dict[str, Any] = {
            "missing_count": int(df[col].isna().sum()),
            "unique_count": int(df[col].nunique()),
        }
        if pd.api.types.is_numeric_dtype(df[col]):
            col_stats.update(
                {
                    "mean": round(float(df[col].mean()), 4) if not df[col].isna().all() else None,
                    "std": round(float(df[col].std()), 4) if not df[col].isna().all() else None,
                    "min": float(df[col].min()) if not df[col].isna().all() else None,
                    "max": float(df[col].max()) if not df[col].isna().all() else None,
                }
            )
        statistics[str(col)] = col_stats

    return {
        "id": dataset.id,
        "name": dataset.name,
        "file_size": dataset.file_size,
        "row_count": dataset.row_count,
        "column_count": dataset.column_count,
        "columns_info": _merge_columns_info_with_sample(dataset.columns_info, sample_columns_info),
        "created_at": dataset.created_at.isoformat() if dataset.created_at else None,
        "rows": df.where(df.notna(), None).to_dict(orient="records"),
        "statistics": statistics,
    }


async def list_datasets(
    db: AsyncSession,
    page: int = 1,
    page_size: int = 20,
) -> dict[str, Any]:
    """Return a paginated list of datasets ordered by creation date (newest first)."""
    # total count
    count_result = await db.execute(select(func.count(Dataset.id)))
    total: int = count_result.scalar_one()

    # paginated query
    offset = (page - 1) * page_size
    stmt = (
        select(Dataset)
        .order_by(Dataset.created_at.desc())
        .offset(offset)
        .limit(page_size)
    )
    result = await db.execute(stmt)
    datasets = result.scalars().all()

    return {
        "items": [
            {
                "id": ds.id,
                "name": ds.name,
                "file_size": ds.file_size,
                "row_count": ds.row_count,
                "column_count": ds.column_count,
                "columns_info": ds.columns_info,
                "created_at": ds.created_at.isoformat() if ds.created_at else None,
            }
            for ds in datasets
        ],
        "total": total,
        "page": page,
        "page_size": page_size,
    }


async def get_correlation(
    dataset_id: str,
    db: AsyncSession,
    method: str = "pearson",
) -> dict[str, Any]:
    """Return the correlation matrix for all numeric columns in the dataset."""
    result = await db.execute(select(Dataset).where(Dataset.id == dataset_id))
    dataset: Dataset | None = result.scalar_one_or_none()
    if dataset is None:
        raise HTTPException(status_code=404, detail="Dataset not found")

    file_path = resolve_runtime_path(dataset.file_path)
    df = _read_dataframe(file_path, file_path.suffix.lower())
    numeric_df = df.select_dtypes(include="number")

    if numeric_df.empty:
        return {"columns": [], "matrix": [], "method": method}

    corr = numeric_df.corr(method=method).round(4)
    columns = corr.columns.tolist()
    matrix = corr.where(corr.notna(), None).values.tolist()

    return {"columns": columns, "matrix": matrix, "method": method}


async def get_target_distribution(
    dataset_id: str,
    target_column: str,
    db: AsyncSession,
) -> dict[str, Any]:
    """Return value-count distribution and descriptive stats for target_column."""
    result = await db.execute(select(Dataset).where(Dataset.id == dataset_id))
    dataset: Dataset | None = result.scalar_one_or_none()
    if dataset is None:
        raise HTTPException(status_code=404, detail="Dataset not found")

    file_path = resolve_runtime_path(dataset.file_path)
    df = _read_dataframe(file_path, file_path.suffix.lower())

    if target_column not in df.columns:
        raise HTTPException(
            status_code=400,
            detail=f"Column '{target_column}' not found in dataset",
        )

    col = df[target_column]
    is_numeric = pd.api.types.is_numeric_dtype(col)

    if is_numeric:
        stats = {
            "mean": round(float(col.mean()), 4) if col.notna().any() else None,
            "std": round(float(col.std()), 4) if col.notna().any() else None,
            "min": float(col.min()) if col.notna().any() else None,
            "max": float(col.max()) if col.notna().any() else None,
            "median": round(float(col.median()), 4) if col.notna().any() else None,
        }
        counts = None
        # Histogram bins for continuous numeric
        hist_counts, bin_edges = pd.cut(col.dropna(), bins=20, retbins=True)
        hist = [
            {"bin_start": round(float(bin_edges[i]), 4), "count": int((hist_counts == hist_counts.cat.categories[i]).sum())}
            for i in range(len(hist_counts.cat.categories))
        ]
    else:
        stats = None
        counts = col.value_counts().head(50).to_dict()
        counts = {str(k): int(v) for k, v in counts.items()}
        hist = None

    return {
        "column": target_column,
        "dtype": str(col.dtype),
        "is_numeric": is_numeric,
        "missing_count": int(col.isna().sum()),
        "unique_count": int(col.nunique()),
        "stats": stats,
        "value_counts": counts,
        "histogram": hist,
    }


async def delete_dataset(dataset_id: str, db: AsyncSession) -> bool:
    """Delete a dataset's file from disk and its record from the database.

    Raises ``HTTPException(404)`` when the dataset is not found.
    Returns ``True`` on success.
    """
    result = await db.execute(select(Dataset).where(Dataset.id == dataset_id))
    dataset: Dataset | None = result.scalar_one_or_none()
    if dataset is None:
        raise HTTPException(status_code=404, detail="Dataset not found")

    # -- remove file from disk --
    file_path = resolve_runtime_path(dataset.file_path)
    if file_path.exists():
        file_path.unlink()
        logger.info("Deleted file %s", file_path)
    else:
        logger.warning("File %s not found on disk; removing DB record only", file_path)

    # -- remove DB record --
    await db.delete(dataset)
    await db.flush()

    logger.info("Deleted dataset record %s", dataset_id)
    return True

"""Data upload and management service.

Handles dataset uploading, preview, listing, and deletion with
async SQLAlchemy sessions and pandas-based file inspection.
"""

from __future__ import annotations

import hashlib
import logging
import os
import tempfile
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pandas as pd
from fastapi import HTTPException, UploadFile
from sqlalchemy import func, select

from app.config import get_settings
from app.models.database import AsyncSession, Dataset
from app.services.object_storage import restore_dataset_file, upload_dataset_file
from app.utils.storage_paths import resolve_runtime_path, to_portable_storage_path
from app.utils.file_utils import generate_unique_filename

logger = logging.getLogger(__name__)

_ALLOWED_EXTENSIONS = {".csv", ".parquet", ".xlsx"}
_UPLOAD_CHUNK_SIZE = 4 * 1024 * 1024
_HASH_CHUNK_SIZE = 8 * 1024 * 1024
_DATAFRAME_CHUNK_SIZE = 64_000
_ANALYSIS_SAMPLE_SIZE = 50_000
_XLSX_MAX_SIZE = 50 * 1024 * 1024
_QUANTILE_SAMPLE_SIZE = 4_096
_MAX_EXACT_UNIQUES = 100_000
_HISTOGRAM_BINS = 20


def _require_dataset_path(dataset: Dataset) -> Path:
    path = restore_dataset_file(dataset.id, dataset.file_path)
    if path is None:
        raise HTTPException(status_code=404, detail="Dataset artifact not found")
    return path


def _compute_file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(_HASH_CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


async def _find_existing_dataset_by_content(
    db: AsyncSession,
    content_sha256: str,
    file_size: int,
    owner_username: str | None = None,
) -> Dataset | None:
    stmt = select(Dataset).where(Dataset.file_size == file_size)
    if owner_username:
        stmt = stmt.where(Dataset.owner_username == owner_username)
    result = await db.execute(stmt)
    for dataset in result.scalars():
        stored_digest = getattr(dataset, "content_sha256", None)
        if stored_digest:
            if stored_digest == content_sha256:
                return dataset
            continue

        path = restore_dataset_file(dataset.id, dataset.file_path)
        if path is None:
            continue
        try:
            stored_digest = _compute_file_digest(path)
            dataset.content_sha256 = stored_digest
            if stored_digest == content_sha256:
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


def _validate_xlsx_size(path: Path, file_size: int) -> None:
    if file_size > _XLSX_MAX_SIZE:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Excel 文件 {path.name!r} 超过 50MB，无法以低内存方式解析；"
                "请转换为 CSV 或 Parquet 后重新上传。"
            ),
        )


def _iter_dataframe_chunks(
    path: Path,
    ext: str,
    *,
    chunk_size: int = _DATAFRAME_CHUNK_SIZE,
) -> Iterator[pd.DataFrame]:
    if ext == ".csv":
        yield from pd.read_csv(path, chunksize=chunk_size)
        return
    if ext == ".parquet":
        import pyarrow.parquet as pq

        parquet_file = pq.ParquetFile(path)
        for batch in parquet_file.iter_batches(batch_size=chunk_size):
            yield batch.to_pandas()
        return
    if ext == ".xlsx":
        _validate_xlsx_size(path, path.stat().st_size)
        yield pd.read_excel(path)
        return
    raise HTTPException(status_code=400, detail=f"Cannot read extension '{ext}'")


def _read_dataframe(path: Path, ext: str, nrows: int | None = None) -> pd.DataFrame:
    """Read a bounded preview or a complete small dataframe."""
    if ext == ".csv":
        return pd.read_csv(path, nrows=nrows)
    if ext == ".parquet":
        if nrows is None:
            return pd.read_parquet(path)
        import pyarrow.parquet as pq

        batches = pq.ParquetFile(path).iter_batches(batch_size=max(1, nrows))
        try:
            return next(batches).to_pandas().head(nrows)
        except StopIteration:
            return pd.DataFrame()
    if ext == ".xlsx":
        _validate_xlsx_size(path, path.stat().st_size)
        return pd.read_excel(path, nrows=nrows)
    raise HTTPException(status_code=400, detail=f"Cannot read extension '{ext}'")


def _normalise_counter_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    try:
        hash(value)
    except TypeError:
        return str(value)
    return value


@dataclass
class _ColumnAccumulator:
    dtype: str | None = None
    total_rows: int = 0
    count: int = 0
    null_count: int = 0
    mean: float = 0.0
    m2: float = 0.0
    minimum: float | None = None
    maximum: float | None = None
    numeric: bool | None = None
    exact_counts: Counter = field(default_factory=Counter)
    unique_overflow: bool = False
    quantile_sample: list[float] = field(default_factory=list)
    sampled_numeric_count: int = 0
    rng: np.random.Generator = field(
        default_factory=lambda: np.random.default_rng(42),
        repr=False,
    )

    def update(self, series: pd.Series) -> None:
        self.total_rows += len(series)
        missing = int(series.isna().sum())
        self.null_count += missing
        non_null = series.dropna()
        self.count += len(non_null)

        current_dtype = str(series.dtype)
        if self.dtype is None:
            self.dtype = current_dtype
        elif self.dtype != current_dtype:
            self.dtype = "object"

        is_numeric = pd.api.types.is_numeric_dtype(series)
        if self.numeric is None:
            self.numeric = bool(is_numeric)
        elif not is_numeric:
            self.numeric = False

        if not self.unique_overflow:
            counts = non_null.value_counts(dropna=True)
            for value, count in counts.items():
                key = _normalise_counter_value(value)
                if key not in self.exact_counts and len(self.exact_counts) >= _MAX_EXACT_UNIQUES:
                    self.unique_overflow = True
                    break
                self.exact_counts[key] += int(count)

        if not is_numeric or non_null.empty:
            return

        values = pd.to_numeric(non_null, errors="coerce").dropna().to_numpy(dtype=float)
        if values.size == 0:
            return
        chunk_count = int(values.size)
        chunk_mean = float(values.mean())
        chunk_m2 = float(((values - chunk_mean) ** 2).sum())
        previous_count = self.sampled_numeric_count
        combined_count = previous_count + chunk_count
        if previous_count == 0:
            self.mean = chunk_mean
            self.m2 = chunk_m2
        else:
            delta = chunk_mean - self.mean
            self.mean += delta * chunk_count / combined_count
            self.m2 += chunk_m2 + delta * delta * previous_count * chunk_count / combined_count
        self.sampled_numeric_count = combined_count
        chunk_min = float(values.min())
        chunk_max = float(values.max())
        self.minimum = chunk_min if self.minimum is None else min(self.minimum, chunk_min)
        self.maximum = chunk_max if self.maximum is None else max(self.maximum, chunk_max)

        for value in values:
            seen = previous_count
            previous_count += 1
            if len(self.quantile_sample) < _QUANTILE_SAMPLE_SIZE:
                self.quantile_sample.append(float(value))
                continue
            replacement = int(self.rng.integers(0, seen + 1))
            if replacement < _QUANTILE_SAMPLE_SIZE:
                self.quantile_sample[replacement] = float(value)

    def to_metadata(self) -> dict[str, Any]:
        if self.unique_overflow:
            sample_unique = len(self.exact_counts)
            sample_count = max(1, sum(self.exact_counts.values()))
            unique_count = min(
                self.count,
                max(sample_unique, round(sample_unique / sample_count * self.count)),
            )
            min_class_count = 1 if self.count else 0
        else:
            unique_count = len(self.exact_counts)
            min_class_count = min(self.exact_counts.values(), default=0)

        quantiles: dict[str, float | None] = {
            "p25": None,
            "p50": None,
            "p75": None,
        }
        if self.quantile_sample:
            p25, p50, p75 = np.quantile(self.quantile_sample, [0.25, 0.5, 0.75])
            quantiles = {"p25": float(p25), "p50": float(p50), "p75": float(p75)}

        return {
            "dtype": self.dtype or "object",
            "missing_count": self.null_count,
            "missing_rate": round(self.null_count / self.total_rows, 4) if self.total_rows else 0.0,
            "unique_count": int(unique_count),
            "unique_rate": round(unique_count / self.total_rows, 4) if self.total_rows else 0.0,
            "min_class_count": int(min_class_count),
            "count": self.count,
            "null_count": self.null_count,
            "mean": self.mean if self.numeric and self.sampled_numeric_count else None,
            "m2": self.m2 if self.numeric and self.sampled_numeric_count else None,
            "min": self.minimum if self.numeric else None,
            "max": self.maximum if self.numeric else None,
            "quantiles": {
                "approx": True,
                "sample_size": len(self.quantile_sample),
                **quantiles,
            },
        }


@dataclass(frozen=True)
class _DatasetScan:
    row_count: int
    column_count: int
    columns_info: dict[str, dict[str, Any]]


def _scan_dataset(
    path: Path,
    ext: str,
    *,
    chunk_size: int = _DATAFRAME_CHUNK_SIZE,
) -> _DatasetScan:
    if ext == ".xlsx":
        _validate_xlsx_size(path, path.stat().st_size)

    accumulators: dict[str, _ColumnAccumulator] = {}
    row_count = 0
    for chunk in _iter_dataframe_chunks(path, ext, chunk_size=chunk_size):
        row_count += len(chunk)
        for column in chunk.columns:
            name = str(column)
            accumulators.setdefault(name, _ColumnAccumulator()).update(chunk[column])

    if ext == ".parquet":
        import pyarrow.parquet as pq

        row_count = int(pq.ParquetFile(path).metadata.num_rows)

    columns_info = {
        name: accumulator.to_metadata()
        for name, accumulator in accumulators.items()
    }

    histogram_specs: dict[str, tuple[np.ndarray, list[int]]] = {}
    for name, metadata in columns_info.items():
        minimum = metadata["min"]
        maximum = metadata["max"]
        if minimum is None or maximum is None:
            continue
        if minimum == maximum:
            edges = np.linspace(minimum - 0.5, maximum + 0.5, _HISTOGRAM_BINS + 1)
        else:
            edges = np.linspace(minimum, maximum, _HISTOGRAM_BINS + 1)
        histogram_specs[name] = (edges, [0] * _HISTOGRAM_BINS)

    if histogram_specs:
        underflow = {name: 0 for name in histogram_specs}
        overflow = {name: 0 for name in histogram_specs}
        for chunk in _iter_dataframe_chunks(path, ext, chunk_size=chunk_size):
            for name, (edges, counts) in histogram_specs.items():
                values = pd.to_numeric(chunk[name], errors="coerce").dropna().to_numpy(dtype=float)
                if not values.size:
                    continue
                underflow[name] += int((values < edges[0]).sum())
                overflow[name] += int((values > edges[-1]).sum())
                in_range = values[(values >= edges[0]) & (values <= edges[-1])]
                batch_counts, _ = np.histogram(in_range, bins=edges)
                for index, value in enumerate(batch_counts):
                    counts[index] += int(value)

        for name, (edges, counts) in histogram_specs.items():
            columns_info[name]["histogram"] = {
                "bin_edges": [float(value) for value in edges],
                "counts": counts,
                "underflow": underflow[name],
                "overflow": overflow[name],
                "missing": columns_info[name]["null_count"],
            }

    for metadata in columns_info.values():
        metadata.setdefault("histogram", None)

    return _DatasetScan(
        row_count=row_count,
        column_count=len(columns_info),
        columns_info=columns_info,
    )


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


async def upload_dataset(
    file: UploadFile,
    db: AsyncSession,
    *,
    content_length: int | None = None,
    owner_username: str | None = None,
) -> Dataset:
    """Validate, persist, and catalogue an uploaded dataset file.

    Steps:
        1. Validate file extension and size.
        2. Save to the configured upload directory with a UUID-prefixed name.
        3. Read with pandas to extract row/column metadata.
        4. Create a ``Dataset`` record in the database.

    Returns the newly-created ``Dataset`` ORM instance.
    """
    settings = get_settings()

    original_name = file.filename or "unnamed"
    ext = _validate_extension(original_name)

    if content_length is not None and content_length > settings.max_upload_size:
        raise HTTPException(
            status_code=413,
            detail=(
                f"Request size ({content_length:,} bytes) exceeds the maximum "
                f"allowed size ({settings.max_upload_size:,} bytes)."
            ),
        )

    unique_name = generate_unique_filename(original_name)
    dest_path = settings.storage_uploads / unique_name
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    temp_name: str | None = None
    file_size = 0
    digest = hashlib.sha256()
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{dest_path.name}.",
            suffix=".tmp",
            dir=dest_path.parent,
            delete=False,
        ) as tmp:
            temp_name = tmp.name
            while chunk := await file.read(_UPLOAD_CHUNK_SIZE):
                file_size += len(chunk)
                if file_size > settings.max_upload_size:
                    raise HTTPException(
                        status_code=413,
                        detail=(
                            f"File size exceeds the maximum allowed size "
                            f"({settings.max_upload_size:,} bytes)."
                        ),
                    )
                tmp.write(chunk)
                digest.update(chunk)
            tmp.flush()
            os.fsync(tmp.fileno())
        os.replace(temp_name, dest_path)
        temp_name = None
    except Exception:
        if temp_name:
            Path(temp_name).unlink(missing_ok=True)
        dest_path.unlink(missing_ok=True)
        raise

    content_sha256 = digest.hexdigest()
    if ext == ".xlsx":
        try:
            _validate_xlsx_size(dest_path, file_size)
        except HTTPException:
            dest_path.unlink(missing_ok=True)
            raise

    existing_dataset = await _find_existing_dataset_by_content(
        db,
        content_sha256=content_sha256,
        file_size=file_size,
        owner_username=owner_username,
    )
    if existing_dataset is not None:
        dest_path.unlink(missing_ok=True)
        logger.info("Reusing existing dataset %s for '%s' by content digest", existing_dataset.id, original_name)
        return existing_dataset

    logger.info("Saved uploaded file to %s (%s bytes)", dest_path, file_size)

    try:
        scan = _scan_dataset(dest_path, ext)
    except HTTPException:
        dest_path.unlink(missing_ok=True)
        raise
    except Exception as exc:
        dest_path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=400,
            detail=f"Failed to parse uploaded file: {exc}",
        ) from exc

    dataset = Dataset(
        owner_username=owner_username,
        name=original_name,
        file_path=to_portable_storage_path(dest_path),
        file_size=file_size,
        row_count=scan.row_count,
        column_count=scan.column_count,
        columns_info=scan.columns_info,
        content_sha256=content_sha256,
    )
    db.add(dataset)
    await db.flush()
    await db.refresh(dataset)

    upload_dataset_file(dataset.id, dest_path)

    logger.info("Created dataset record %s for '%s'", dataset.id, original_name)
    return dataset


def _sample_dataframe(
    path: Path,
    ext: str,
    *,
    sample_size: int = _ANALYSIS_SAMPLE_SIZE,
    seed: int = 42,
) -> tuple[pd.DataFrame, int]:
    """Return a deterministic uniform sample without materialising the full file."""
    rng = np.random.default_rng(seed)
    sampled: pd.DataFrame | None = None
    total_rows = 0
    priority_column = "__m1_sample_priority__"
    for chunk in _iter_dataframe_chunks(path, ext):
        total_rows += len(chunk)
        candidate = chunk.copy()
        candidate[priority_column] = rng.random(len(candidate))
        if sampled is not None:
            candidate = pd.concat([sampled, candidate], ignore_index=True)
        if len(candidate) > sample_size:
            candidate = candidate.nsmallest(sample_size, priority_column)
        sampled = candidate

    if sampled is None:
        return pd.DataFrame(), 0
    return sampled.drop(columns=[priority_column]).reset_index(drop=True), total_rows


async def get_dataset_preview(
    dataset_id: str,
    db: AsyncSession,
    rows: int = 100,
    owner_username: str | None = None,
) -> dict[str, Any]:
    """Return a preview of the dataset including sample rows and statistics.

    Raises ``HTTPException(404)`` when the dataset is not found.
    """
    stmt = select(Dataset).where(Dataset.id == dataset_id)
    if owner_username:
        stmt = stmt.where(Dataset.owner_username == owner_username)
    result = await db.execute(stmt)
    dataset: Dataset | None = result.scalar_one_or_none()
    if dataset is None:
        raise HTTPException(status_code=404, detail="Dataset not found")

    file_path = _require_dataset_path(dataset)
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
    owner_username: str | None = None,
) -> dict[str, Any]:
    """Return a paginated list of datasets ordered by creation date (newest first)."""
    # total count
    count_stmt = select(func.count(Dataset.id))
    if owner_username:
        count_stmt = count_stmt.where(Dataset.owner_username == owner_username)
    count_result = await db.execute(count_stmt)
    total: int = count_result.scalar_one()

    # paginated query
    offset = (page - 1) * page_size
    stmt = select(Dataset)
    if owner_username:
        stmt = stmt.where(Dataset.owner_username == owner_username)
    stmt = stmt.order_by(Dataset.created_at.desc()).offset(offset).limit(page_size)
    result = await db.execute(stmt)
    datasets = result.scalars().all()

    return {
        "items": [
            {
                "id": ds.id,
                "owner_username": ds.owner_username,
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
    owner_username: str | None = None,
) -> dict[str, Any]:
    """Return the correlation matrix for all numeric columns in the dataset."""
    stmt = select(Dataset).where(Dataset.id == dataset_id)
    if owner_username:
        stmt = stmt.where(Dataset.owner_username == owner_username)
    result = await db.execute(stmt)
    dataset: Dataset | None = result.scalar_one_or_none()
    if dataset is None:
        raise HTTPException(status_code=404, detail="Dataset not found")

    file_path = _require_dataset_path(dataset)
    df, total_rows = _sample_dataframe(file_path, file_path.suffix.lower())
    sampled = total_rows > _ANALYSIS_SAMPLE_SIZE
    numeric_df = df.select_dtypes(include="number")

    if numeric_df.empty:
        return {
            "columns": [],
            "matrix": [],
            "method": method,
            "sampled": sampled,
            "sample_size": len(df),
        }

    corr = numeric_df.corr(method=method).round(4)
    columns = corr.columns.tolist()
    matrix = corr.where(corr.notna(), None).values.tolist()

    return {
        "columns": columns,
        "matrix": matrix,
        "method": method,
        "sampled": sampled,
        "sample_size": len(df),
    }


async def get_target_distribution(
    dataset_id: str,
    target_column: str,
    db: AsyncSession,
    owner_username: str | None = None,
) -> dict[str, Any]:
    """Return value-count distribution and descriptive stats for target_column."""
    stmt = select(Dataset).where(Dataset.id == dataset_id)
    if owner_username:
        stmt = stmt.where(Dataset.owner_username == owner_username)
    result = await db.execute(stmt)
    dataset: Dataset | None = result.scalar_one_or_none()
    if dataset is None:
        raise HTTPException(status_code=404, detail="Dataset not found")

    file_path = _require_dataset_path(dataset)
    df, total_rows = _sample_dataframe(file_path, file_path.suffix.lower())
    sampled = total_rows > _ANALYSIS_SAMPLE_SIZE

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
        "sampled": sampled,
        "sample_size": len(df),
    }


async def delete_dataset(
    dataset_id: str,
    db: AsyncSession,
    owner_username: str | None = None,
) -> bool:
    """Delete a dataset's file from disk and its record from the database.

    Raises ``HTTPException(404)`` when the dataset is not found.
    Returns ``True`` on success.
    """
    stmt = select(Dataset).where(Dataset.id == dataset_id)
    if owner_username:
        stmt = stmt.where(Dataset.owner_username == owner_username)
    result = await db.execute(stmt)
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

from __future__ import annotations

import hashlib
import io
from datetime import datetime, timezone
from types import SimpleNamespace

import pandas as pd
import pytest
from fastapi import HTTPException

from app.services import data_service


class _ScalarResult:
    def __init__(self, value):
        self.value = value

    def scalar_one_or_none(self):
        return self.value

    def scalars(self):
        return iter(self.value)


class _Db:
    def __init__(self, result=None):
        self.result = result
        self.added = []

    async def execute(self, _statement):
        return _ScalarResult(self.result)

    def add(self, value):
        self.added.append(value)

    async def flush(self):
        for value in self.added:
            value.id = value.id or "dataset-new"
            value.created_at = value.created_at or datetime.now(timezone.utc)

    async def refresh(self, _value):
        return None


class _TrackingUpload:
    filename = "stream.csv"

    def __init__(self, payload: bytes):
        self._stream = io.BytesIO(payload)
        self.read_sizes = []

    async def read(self, size=-1):
        self.read_sizes.append(size)
        return self._stream.read(size)


@pytest.mark.asyncio
async def test_upload_rejects_content_length_before_reading(monkeypatch, tmp_path):
    upload = _TrackingUpload(b"a,b\n1,2\n")
    settings = SimpleNamespace(storage_uploads=tmp_path, max_upload_size=5)
    monkeypatch.setattr(data_service, "get_settings", lambda: settings)

    with pytest.raises(HTTPException) as exc:
        await data_service.upload_dataset(upload, _Db(), content_length=10)

    assert exc.value.status_code == 413
    assert upload.read_sizes == []
    assert list(tmp_path.iterdir()) == []


@pytest.mark.asyncio
async def test_upload_streams_to_temp_file_and_records_digest(monkeypatch, tmp_path):
    payload = b"feature,target\n1,0\n2,1\n"
    upload = _TrackingUpload(payload)
    settings = SimpleNamespace(storage_uploads=tmp_path, max_upload_size=1024)
    db = _Db()

    async def no_duplicate(_db, content_sha256, file_size, **_kwargs):
        assert content_sha256 == hashlib.sha256(payload).hexdigest()
        assert file_size == len(payload)
        return None

    monkeypatch.setattr(data_service, "get_settings", lambda: settings)
    monkeypatch.setattr(data_service, "_find_existing_dataset_by_content", no_duplicate)
    monkeypatch.setattr(data_service, "to_portable_storage_path", lambda path: str(path))
    monkeypatch.setattr(data_service, "upload_dataset_file", lambda *_args: None)

    dataset = await data_service.upload_dataset(upload, db, content_length=len(payload))

    assert upload.read_sizes and set(upload.read_sizes) == {data_service._UPLOAD_CHUNK_SIZE}
    assert dataset.content_sha256 == hashlib.sha256(payload).hexdigest()
    assert dataset.file_size == len(payload)
    assert dataset.row_count == 2
    assert (tmp_path / dataset.file_path.split("/")[-1]).read_bytes() == payload
    assert list(tmp_path.glob("*.tmp")) == []


def test_csv_metadata_scan_is_incremental_and_mergeable(tmp_path):
    path = tmp_path / "stats.csv"
    frame = pd.DataFrame(
        {
            "value": [*range(100), None],
            "label": ["a"] * 60 + ["b"] * 40 + [None],
        }
    )
    frame.to_csv(path, index=False)

    scan = data_service._scan_dataset(path, ".csv", chunk_size=17)
    value = scan.columns_info["value"]

    assert scan.row_count == 101
    assert scan.column_count == 2
    assert value["count"] == 100
    assert value["null_count"] == 1
    assert value["mean"] == pytest.approx(49.5)
    assert value["m2"] == pytest.approx(83325.0)
    assert value["histogram"]["bin_edges"] and len(value["histogram"]["bin_edges"]) == 21
    assert sum(value["histogram"]["counts"]) == 100
    assert value["histogram"]["missing"] == 1
    assert value["quantiles"]["approx"] is True
    assert scan.columns_info["label"]["unique_count"] == 2
    assert scan.columns_info["label"]["min_class_count"] == 40


def test_parquet_metadata_scan_uses_batches(tmp_path):
    path = tmp_path / "stats.parquet"
    pd.DataFrame({"value": range(123), "label": ["x"] * 123}).to_parquet(path)

    scan = data_service._scan_dataset(path, ".parquet", chunk_size=19)

    assert scan.row_count == 123
    assert scan.column_count == 2
    assert scan.columns_info["value"]["count"] == 123
    assert scan.columns_info["value"]["mean"] == pytest.approx(61.0)


def test_xlsx_over_50mb_is_rejected_with_csv_guidance(tmp_path):
    path = tmp_path / "large.xlsx"

    with pytest.raises(HTTPException) as exc:
        data_service._validate_xlsx_size(path, 50 * 1024 * 1024 + 1)

    assert exc.value.status_code == 400
    assert "CSV" in exc.value.detail


@pytest.mark.asyncio
async def test_dedup_lazily_hashes_and_backfills_legacy_dataset(monkeypatch, tmp_path):
    payload = b"feature,target\n1,0\n"
    path = tmp_path / "legacy.csv"
    path.write_bytes(payload)
    legacy = SimpleNamespace(
        id="legacy",
        file_path="storage/uploads/legacy.csv",
        file_size=len(payload),
        content_sha256=None,
    )
    monkeypatch.setattr(data_service, "restore_dataset_file", lambda *_args: path)

    found = await data_service._find_existing_dataset_by_content(
        _Db([legacy]),
        hashlib.sha256(payload).hexdigest(),
        len(payload),
    )

    assert found is legacy
    assert legacy.content_sha256 == hashlib.sha256(payload).hexdigest()


@pytest.mark.asyncio
async def test_correlation_samples_datasets_over_50000_rows(tmp_path):
    path = tmp_path / "large.csv"
    pd.DataFrame({"a": range(50_010), "b": range(50_010)}).to_csv(path, index=False)
    dataset = SimpleNamespace(
        id="dataset-1",
        file_path=str(path),
        row_count=50_010,
    )

    result = await data_service.get_correlation("dataset-1", _Db(dataset))

    assert result["sampled"] is True
    assert result["sample_size"] == 50_000
    assert result["columns"] == ["a", "b"]


@pytest.mark.asyncio
async def test_target_distribution_samples_datasets_over_50000_rows(tmp_path):
    path = tmp_path / "large.csv"
    pd.DataFrame({"target": [0, 1] * 25_005}).to_csv(path, index=False)
    dataset = SimpleNamespace(
        id="dataset-1",
        file_path=str(path),
        row_count=50_010,
    )

    result = await data_service.get_target_distribution(
        "dataset-1", "target", _Db(dataset)
    )

    assert result["sampled"] is True
    assert result["sample_size"] == 50_000
    assert result["is_numeric"] is True

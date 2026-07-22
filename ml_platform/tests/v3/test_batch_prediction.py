"""M3-2 — asynchronous CSV batch prediction.

The properties worth pinning are the ones that separate this from the existing
synchronous route: the request must not do the work, results must live in a
file rather than a JSON column, and a partial run must leave honest progress.
"""
from __future__ import annotations

import csv
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi import HTTPException

from app.models.database import (
    Dataset,
    InferenceJob,
    ModelDeployment,
    PlatformTask,
    TrainingTask,
)
from app.services import batch_prediction_service as bps
from sqlalchemy import select


@pytest.fixture(autouse=True)
def use_test_sessions(session_factory):
    with patch("app.models.database.async_session_factory", session_factory), \
            patch.object(bps, "async_session_factory", session_factory):
        yield


@pytest.fixture(autouse=True)
def scratch_predictions(tmp_path, monkeypatch):
    monkeypatch.setattr(bps, "_predictions_dir", lambda: tmp_path)
    return tmp_path


@pytest.fixture(autouse=True)
def dont_auto_execute(monkeypatch):
    """Stop submit() from running the job in the background.

    The in-process scheduler dispatches immediately, so without this a test
    that calls ``run_batch_prediction`` explicitly races a copy the scheduler
    already started — the two runs interleave and share fixture state. Tests
    here drive the executor themselves; scheduling is covered by asserting the
    PlatformTask was created.
    """
    submitted: list[str] = []

    class _NoopScheduler:
        async def submit(self, task_id):
            submitted.append(task_id)
            return None

    monkeypatch.setattr("app.scheduler.scheduler.get_scheduler", lambda kind: _NoopScheduler())
    return submitted


async def _seed(db, *, status="active"):
    ds = Dataset(name="d.csv", file_path="/tmp/d.csv", file_size=10, row_count=3)
    db.add(ds)
    await db.flush()
    task = TrainingTask(
        name="t", dataset_id=ds.id, model_type="logistic_regression",
        target_column="y", status="completed", model_path="/tmp/m.joblib",
    )
    db.add(task)
    await db.flush()
    dep = ModelDeployment(task_id=task.id, name="dep", status=status)
    db.add(dep)
    await db.flush()
    await db.commit()
    return dep.id


CSV_IN = b"a,b\n1,2\n3,4\n5,6\n"


async def test_submit_returns_immediately_without_predicting(db, session_factory, dont_auto_execute):
    """The request handler must not run the model — that is the whole point of
    making this asynchronous."""
    dep_id = await _seed(db)
    predicted = []

    with patch.object(bps, "_load_predictor", lambda *a, **k: predicted.append(1)):
        result = await bps.create_batch_job(
            db, deployment_id=dep_id, filename="in.csv", content=CSV_IN
        )

    assert result["status"] == "pending"
    assert result["job_id"] and result["platform_task_id"]
    assert predicted == [], "the model was loaded inside the request"
    assert dont_auto_execute == [result["platform_task_id"]], "task was not dispatched"

    async with session_factory() as s:
        job = (await s.execute(select(InferenceJob).where(InferenceJob.id == result["job_id"]))).scalar_one()
        task = (await s.execute(select(PlatformTask).where(PlatformTask.id == result["platform_task_id"]))).scalar_one()
    assert job.status == "pending"
    assert Path(job.input_path).exists(), "upload was not persisted"
    assert task.kind == "predict"
    assert task.payload_ref == f"predict:{job.id}"


async def test_paused_deployment_is_refused(db):
    dep_id = await _seed(db, status="paused")
    with pytest.raises(HTTPException) as exc:
        await bps.create_batch_job(db, deployment_id=dep_id, filename="in.csv", content=CSV_IN)
    assert exc.value.status_code == 400


async def test_non_csv_is_refused(db):
    dep_id = await _seed(db)
    with pytest.raises(HTTPException) as exc:
        await bps.create_batch_job(db, deployment_id=dep_id, filename="in.xlsx", content=CSV_IN)
    assert exc.value.status_code == 422
    assert "CSV" in str(exc.value.detail)


async def test_empty_upload_is_refused(db):
    dep_id = await _seed(db)
    with pytest.raises(HTTPException) as exc:
        await bps.create_batch_job(db, deployment_id=dep_id, filename="in.csv", content=b"   \n")
    assert exc.value.status_code == 422


async def test_executor_writes_a_result_file_not_a_json_column(db, session_factory, scratch_predictions):
    """Results belong in a file. A JSON column cannot hold a large result and
    cannot be streamed back to the caller."""
    dep_id = await _seed(db)
    submitted = await bps.create_batch_job(
        db, deployment_id=dep_id, filename="in.csv", content=CSV_IN
    )
    job_id = submitted["job_id"]

    def fake_predict(model, training_df, rows, target, include_probabilities=False):
        return {"predictions": [f"p{i}" for i in range(len(rows))], "probabilities": None}

    async def fake_loader(db_, deployment_id):
        return object(), None, "y"

    with patch.object(bps, "_load_predictor", fake_loader), \
            patch("app.services.prediction_service.predict_with_model", fake_predict), \
            patch.object(bps, "upload_file", lambda *a, **k: None):
        out = await bps.run_batch_prediction(job_id, submitted["platform_task_id"])

    assert out["metrics"]["processed_rows"] == 3

    async with session_factory() as s:
        job = (await s.execute(select(InferenceJob).where(InferenceJob.id == job_id))).scalar_one()
    assert job.status == "completed"
    assert job.processed_rows == 3
    assert job.predictions is None, "batch results must not land in the JSON column"

    rows = list(csv.DictReader(Path(job.result_path).open(encoding="utf-8-sig")))
    assert len(rows) == 3
    assert rows[0]["prediction"] == "p0"
    assert rows[0]["a"] == "1", "input columns must be preserved alongside the prediction"


async def test_missing_input_file_fails_loudly(db, session_factory):
    """A vanished upload must raise so the scheduler records a failure, rather
    than quietly producing an empty result file."""
    dep_id = await _seed(db)
    submitted = await bps.create_batch_job(
        db, deployment_id=dep_id, filename="in.csv", content=CSV_IN
    )
    async with session_factory() as s:
        job = (await s.execute(select(InferenceJob).where(InferenceJob.id == submitted["job_id"]))).scalar_one()
        Path(job.input_path).unlink()

    with pytest.raises(FileNotFoundError):
        await bps.run_batch_prediction(submitted["job_id"], submitted["platform_task_id"])


async def test_partial_run_leaves_honest_progress(db, session_factory, monkeypatch, scratch_predictions):
    """Progress is committed per chunk, so a job that dies partway reports how
    far it actually got. Writing progress only at the end would make a crashed
    job indistinguishable from one that never started."""
    monkeypatch.setattr(bps, "CHUNK_SIZE", 1)
    dep_id = await _seed(db)
    submitted = await bps.create_batch_job(
        db, deployment_id=dep_id, filename="in.csv", content=CSV_IN
    )
    job_id = submitted["job_id"]
    calls = {"n": 0}

    def failing_predict(model, training_df, rows, target, include_probabilities=False):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("model blew up on chunk 2")
        return {"predictions": ["p"] * len(rows), "probabilities": None}

    async def fake_loader(db_, deployment_id):
        return object(), None, "y"

    with patch.object(bps, "_load_predictor", fake_loader), \
            patch("app.services.prediction_service.predict_with_model", failing_predict), \
            patch.object(bps, "upload_file", lambda *a, **k: None):
        with pytest.raises(RuntimeError, match="chunk 2"):
            await bps.run_batch_prediction(job_id, submitted["platform_task_id"])

    async with session_factory() as s:
        job = (await s.execute(select(InferenceJob).where(InferenceJob.id == job_id))).scalar_one()

    assert job.processed_rows == 1, (
        f"progress was not committed per chunk (got {job.processed_rows}); "
        "a crashed job cannot report how far it got"
    )
    assert job.status != "completed", "a failed job must not look completed"


async def test_serializer_exposes_progress_without_leaking_paths(db, session_factory):
    dep_id = await _seed(db)
    submitted = await bps.create_batch_job(
        db, deployment_id=dep_id, filename="in.csv", content=CSV_IN
    )
    async with session_factory() as s:
        job = (await s.execute(select(InferenceJob).where(InferenceJob.id == submitted["job_id"]))).scalar_one()
    payload = bps.serialize_batch_job(job)

    assert payload["status"] == "pending"
    assert payload["processed_rows"] == 0
    assert payload["has_result"] is False
    # Filesystem paths are server internals — the client gets a download route.
    assert "result_path" not in payload
    assert "input_path" not in payload

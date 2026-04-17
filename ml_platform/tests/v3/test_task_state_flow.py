"""
V3 Stage 2 tests — PlatformTask state-flow and write-back contract.

Tests use an in-memory SQLite DB (fixtures defined in conftest.py).
pytest.ini sets asyncio_mode=auto.
"""

import pytest
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.models.database import PlatformTask


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

async def _make_task(
    session_factory,
    kind: str = "train",
    status: str = "QUEUED",
    retry_count: int = 0,
    max_retries: int = 3,
) -> str:
    """Create a PlatformTask and return its id."""
    async with session_factory() as db:
        task = PlatformTask(
            kind=kind,
            status=status,
            payload_ref=f"{kind}:fake-domain-id",
            retry_count=retry_count,
            max_retries=max_retries,
        )
        db.add(task)
        await db.commit()
        await db.refresh(task)
        return task.id


# ---------------------------------------------------------------------------
# register_domain_task
# ---------------------------------------------------------------------------

async def test_register_domain_task_creates_queued_record(db):
    from app.scheduler.task_runner import register_domain_task

    task = await register_domain_task(db, kind="train", payload_ref="train:abc123")
    await db.commit()

    assert task.id is not None
    assert task.status == "QUEUED"
    assert task.payload_ref == "train:abc123"
    assert task.kind == "train"
    assert task.queued_at is not None


# ---------------------------------------------------------------------------
# update_platform_task_status
# ---------------------------------------------------------------------------

async def test_update_status_running(session_factory):
    task_id = await _make_task(session_factory)

    from app.scheduler.task_runner import update_platform_task_status
    from unittest.mock import patch

    # async_session_factory is resolved via local import inside the function;
    # patch at the source module so the local import sees the stub.
    with patch("app.models.database.async_session_factory", session_factory):
        await update_platform_task_status(task_id, "RUNNING")

    async with session_factory() as fresh_db:
        result = await fresh_db.execute(select(PlatformTask).where(PlatformTask.id == task_id))
        refreshed = result.scalar_one()

    assert refreshed.status == "RUNNING"
    assert refreshed.started_at is not None


async def test_update_status_success_writes_metrics(session_factory):
    task_id = await _make_task(session_factory)

    from app.scheduler.task_runner import update_platform_task_status
    from unittest.mock import patch

    with patch("app.models.database.async_session_factory", session_factory):
        await update_platform_task_status(
            task_id, "SUCCESS",
            metrics={"accuracy": 0.93, "f1": 0.91},
        )

    async with session_factory() as fresh_db:
        result = await fresh_db.execute(select(PlatformTask).where(PlatformTask.id == task_id))
        refreshed = result.scalar_one()

    assert refreshed.status == "SUCCESS"
    assert refreshed.finished_at is not None
    assert refreshed.progress == 1.0
    assert refreshed.metrics_snapshot == {"accuracy": 0.93, "f1": 0.91}


async def test_update_status_failed_writes_error(session_factory):
    task_id = await _make_task(session_factory)

    from app.scheduler.task_runner import update_platform_task_status
    from unittest.mock import patch

    with patch("app.models.database.async_session_factory", session_factory):
        await update_platform_task_status(task_id, "FAILED", error="Division by zero")

    async with session_factory() as fresh_db:
        result = await fresh_db.execute(select(PlatformTask).where(PlatformTask.id == task_id))
        refreshed = result.scalar_one()

    assert refreshed.status == "FAILED"
    assert "Division by zero" in refreshed.error_message
    assert refreshed.finished_at is not None


async def test_update_status_missing_id_does_not_raise():
    """Missing platform_task_id should log a warning but not crash."""
    from app.scheduler.task_runner import update_platform_task_status
    await update_platform_task_status("nonexistent-id-xyz", "SUCCESS")


# ---------------------------------------------------------------------------
# retry_task
# ---------------------------------------------------------------------------

async def test_retry_failed_task(session_factory):
    task_id = await _make_task(session_factory, status="FAILED", retry_count=1, max_retries=3)

    from app.scheduler.task_runner import retry_task
    async with session_factory() as retry_db:
        updated = await retry_task(retry_db, task_id)
        await retry_db.commit()

    assert updated.status == "QUEUED"
    assert updated.error_message is None


async def test_retry_exhausted_raises(session_factory):
    task_id = await _make_task(session_factory, status="FAILED", retry_count=3, max_retries=3)

    from app.scheduler.task_runner import retry_task
    with pytest.raises(ValueError, match="exhausted retries"):
        async with session_factory() as retry_db:
            await retry_task(retry_db, task_id)


# ---------------------------------------------------------------------------
# cancel_task
# ---------------------------------------------------------------------------

async def test_cancel_queued_task(session_factory):
    task_id = await _make_task(session_factory, status="QUEUED")

    from app.scheduler.task_runner import cancel_task
    async with session_factory() as cancel_db:
        updated = await cancel_task(cancel_db, task_id)
        await cancel_db.commit()

    assert updated.status == "CANCELLED"
    assert updated.finished_at is not None


async def test_cancel_already_done_raises(session_factory):
    task_id = await _make_task(session_factory, status="SUCCESS")

    from app.scheduler.task_runner import cancel_task
    with pytest.raises(ValueError, match="already"):
        async with session_factory() as cancel_db:
            await cancel_task(cancel_db, task_id)

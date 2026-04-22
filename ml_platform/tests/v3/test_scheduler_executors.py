"""
Phase 2 + 5 — Executor Registry + Scheduler tests.

Validates that:
  * register/get/list_registered_kinds round-trips work
  * ``get_executor`` raises KeyError for unknown kinds with a helpful message
  * ``run_with_status`` writes SUCCESS + metrics on normal return
  * ``run_with_status`` writes FAILED and re-raises on exception
  * ``InProcessScheduler.submit`` routes to the registered executor
  * DAG gate (Phase 5): tasks with unfinished upstreams park in PENDING
    rather than dispatching
  * DAG gate: tasks whose upstream has SUCCEEDED are released
  * DAG gate: tasks whose upstream has FAILED cascade-cancel
  * ``on_task_done`` re-evaluates dependents (resubmits blocked ones)

These tests follow the same pattern as ``test_task_state_flow.py``: the
in-memory SQLite ``session_factory`` is patched onto
``app.models.database.async_session_factory`` so lazy imports inside the
scheduler / task_runner use the test DB instead of the real MySQL one.
"""
from __future__ import annotations

import asyncio
from unittest.mock import patch

import pytest
from sqlalchemy import select

from app.models.database import PlatformTask
from app.scheduler import executors
from app.scheduler.scheduler import InProcessScheduler


# ---------------------------------------------------------------------------
# Registry basics (no DB needed)
# ---------------------------------------------------------------------------

def test_register_and_get_roundtrip():
    async def fake(domain_id, platform_task_id):  # pragma: no cover - dummy body
        return {"metrics": {}}

    executors.register_executor("test_kind_xyz", fake)
    assert executors.has_executor("test_kind_xyz")
    assert "test_kind_xyz" in executors.list_registered_kinds()
    assert executors.get_executor("test_kind_xyz") is fake


def test_get_unknown_kind_raises_keyerror_with_hint():
    with pytest.raises(KeyError) as exc:
        executors.get_executor("definitely_not_registered_kind")
    msg = str(exc.value)
    assert "definitely_not_registered_kind" in msg
    assert "Known kinds" in msg


# ---------------------------------------------------------------------------
# run_with_status — status write-back contract
# ---------------------------------------------------------------------------

async def test_run_with_status_success_writes_metrics(db, session_factory):
    task = PlatformTask(kind="train_ok", status="QUEUED", payload_ref="train:x")
    db.add(task)
    await db.flush()
    await db.commit()
    task_id = task.id

    async def fake(domain_id, platform_task_id):
        return {"metrics": {"accuracy": 0.87}, "extra": "kept"}

    executors.register_executor("train_ok", fake)

    with patch("app.models.database.async_session_factory", session_factory):
        result = await executors.run_with_status("train_ok", "x", task_id)

    assert result == {"metrics": {"accuracy": 0.87}, "extra": "kept"}

    async with session_factory() as db2:
        row = (
            await db2.execute(select(PlatformTask).where(PlatformTask.id == task_id))
        ).scalar_one()
        assert row.status == "SUCCESS"
        assert row.metrics_snapshot == {"accuracy": 0.87}


async def test_run_with_status_failure_marks_failed_and_reraises(db, session_factory):
    task = PlatformTask(kind="train_bad", status="QUEUED", payload_ref="train:y")
    db.add(task)
    await db.flush()
    await db.commit()
    task_id = task.id

    async def boom(domain_id, platform_task_id):
        raise RuntimeError("engine exploded")

    executors.register_executor("train_bad", boom)

    with patch("app.models.database.async_session_factory", session_factory):
        with pytest.raises(RuntimeError, match="engine exploded"):
            await executors.run_with_status("train_bad", "y", task_id)

    async with session_factory() as db2:
        row = (
            await db2.execute(select(PlatformTask).where(PlatformTask.id == task_id))
        ).scalar_one()
        assert row.status == "FAILED"
        assert "engine exploded" in (row.error_message or "")


# ---------------------------------------------------------------------------
# InProcessScheduler — dispatch routing
# ---------------------------------------------------------------------------

async def test_scheduler_dispatches_to_registered_executor(db, session_factory):
    task = PlatformTask(kind="sched_kind", status="QUEUED", payload_ref="sched_kind:abc")
    db.add(task)
    await db.flush()
    await db.commit()
    task_id = task.id

    called: dict[str, tuple] = {}

    async def fake(domain_id, platform_task_id):
        called["args"] = (domain_id, platform_task_id)
        return {"metrics": {"score": 1.0}}

    executors.register_executor("sched_kind", fake)

    with patch("app.models.database.async_session_factory", session_factory):
        sched = InProcessScheduler()
        await sched.submit(task_id)
        # create_task runs on the current loop; yield until our spy fires.
        for _ in range(50):
            if called:
                break
            await asyncio.sleep(0.01)

    assert called.get("args") == ("abc", task_id)


async def test_scheduler_dag_gate_parks_when_upstream_unknown(db, session_factory, caplog):
    """Phase 5: a task with an unresolved upstream must stay PENDING, not dispatch."""
    task = PlatformTask(
        kind="sched_kind_dag",
        status="PENDING",
        payload_ref="sched_kind_dag:zz",
        depends_on=["upstream-123"],
    )
    db.add(task)
    await db.flush()
    await db.commit()
    task_id = task.id

    called: list[str] = []

    async def fake(domain_id, platform_task_id):
        called.append(platform_task_id)
        return {"metrics": {}}

    executors.register_executor("sched_kind_dag", fake)

    with patch("app.models.database.async_session_factory", session_factory):
        with caplog.at_level("INFO", logger="app.scheduler.scheduler"):
            sched = InProcessScheduler()
            await sched.submit(task_id)
            # Give the create_task scheduler a chance to run — it shouldn't.
            for _ in range(20):
                await asyncio.sleep(0.01)

    assert called == []  # blocked — executor not called
    # The park path writes an error_message explaining the wait.
    async with session_factory() as db2:
        row = (await db2.execute(select(PlatformTask).where(PlatformTask.id == task_id))).scalar_one()
        assert row.status == "PENDING"
        assert row.error_message and "upstream" in row.error_message.lower()
    assert any("blocked" in rec.getMessage().lower() for rec in caplog.records)


async def test_scheduler_dag_gate_releases_after_upstream_success(db, session_factory):
    """When the upstream task is SUCCESS, the gate lets the dependent dispatch."""
    upstream = PlatformTask(
        kind="sched_kind_up",
        status="SUCCESS",
        payload_ref="sched_kind_up:u1",
    )
    db.add(upstream)
    await db.flush()
    await db.commit()
    upstream_id = upstream.id

    downstream = PlatformTask(
        kind="sched_kind_down",
        status="QUEUED",
        payload_ref="sched_kind_down:d1",
        depends_on=[upstream_id],
    )
    db.add(downstream)
    await db.flush()
    await db.commit()
    downstream_id = downstream.id

    called: list[str] = []

    async def fake(domain_id, platform_task_id):
        called.append(platform_task_id)
        return {"metrics": {}}

    executors.register_executor("sched_kind_down", fake)

    with patch("app.models.database.async_session_factory", session_factory):
        sched = InProcessScheduler()
        await sched.submit(downstream_id)
        for _ in range(50):
            if called:
                break
            await asyncio.sleep(0.01)

    assert called == [downstream_id]


async def test_scheduler_dag_gate_cascade_cancels_on_upstream_failure(db, session_factory):
    """Upstream FAILED → dependent is CANCELLED, not dispatched."""
    upstream = PlatformTask(
        kind="sched_kind_up2",
        status="FAILED",
        payload_ref="sched_kind_up2:u2",
    )
    db.add(upstream)
    await db.flush()
    await db.commit()
    upstream_id = upstream.id

    downstream = PlatformTask(
        kind="sched_kind_down2",
        status="PENDING",
        payload_ref="sched_kind_down2:d2",
        depends_on=[upstream_id],
    )
    db.add(downstream)
    await db.flush()
    await db.commit()
    downstream_id = downstream.id

    called: list[str] = []

    async def fake(domain_id, platform_task_id):
        called.append(platform_task_id)
        return {"metrics": {}}

    executors.register_executor("sched_kind_down2", fake)

    with patch("app.models.database.async_session_factory", session_factory):
        sched = InProcessScheduler()
        await sched.submit(downstream_id)
        # No executor should fire and the dependent should be marked CANCELLED.
        for _ in range(20):
            await asyncio.sleep(0.01)

    assert called == []
    async with session_factory() as db2:
        row = (await db2.execute(select(PlatformTask).where(PlatformTask.id == downstream_id))).scalar_one()
        assert row.status == "CANCELLED"
        assert row.error_message and upstream_id in row.error_message


async def test_scheduler_on_task_done_resubmits_dependents(db, session_factory):
    """on_task_done should release tasks that were waiting on the completed one."""
    upstream = PlatformTask(
        kind="sched_kind_up3",
        status="SUCCESS",
        payload_ref="sched_kind_up3:u3",
    )
    db.add(upstream)
    await db.flush()
    await db.commit()
    upstream_id = upstream.id

    downstream = PlatformTask(
        kind="sched_kind_down3",
        status="PENDING",
        payload_ref="sched_kind_down3:d3",
        depends_on=[upstream_id],
    )
    db.add(downstream)
    await db.flush()
    await db.commit()
    downstream_id = downstream.id

    called: list[str] = []

    async def fake(domain_id, platform_task_id):
        called.append(platform_task_id)
        return {"metrics": {}}

    executors.register_executor("sched_kind_down3", fake)

    with patch("app.models.database.async_session_factory", session_factory):
        sched = InProcessScheduler()
        await sched.on_task_done(upstream_id)
        for _ in range(50):
            if called:
                break
            await asyncio.sleep(0.01)

    assert called == [downstream_id]


async def test_scheduler_missing_task_raises(session_factory):
    with patch("app.models.database.async_session_factory", session_factory):
        sched = InProcessScheduler()
        with pytest.raises(ValueError, match="not found"):
            await sched.submit("no-such-id")

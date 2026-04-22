"""
Scheduler abstraction — one entry point for dispatching queued PlatformTasks.

Why this exists
===============
Before Phase 2 the Scheduler layer was implicit: callers invoked
``task_runner.dispatch_platform_task`` directly and that function branched on
``kind`` to pick an ML/DL/SHAP coroutine.  Every new task kind required
touching one very crowded if/elif chain.

Phase 2 replaced the branching with two decoupled pieces:

  1. **Executor Registry** (``app.scheduler.executors``) — services register
     ``(kind → async fn)``.  The registry knows *how* to run work.
  2. **Scheduler** (this module) — knows *when* to run work: consults the DAG
     gate (``PlatformTask.depends_on``) and delegates to the executor via
     ``run_with_status`` so status/metrics write-back is standardized.

Phase 5 additions
=================
* ``CeleryScheduler`` — routes submissions through ``apply_async`` so
  executors run in a separate worker process (essential for GPU-heavy DL
  training that must not block the FastAPI event loop).
* **DAG gate is live** — submissions check ``depends_on``; blocked tasks
  stay in ``PENDING`` and are resubmitted by ``on_task_done`` once every
  upstream reports SUCCESS.  The gate treats FAILED/CANCELLED upstream as
  "blocked forever" — a dependent task is marked CANCELLED rather than
  orphaned.
* ``get_scheduler()`` chooses Celery vs InProcess based on
  ``settings.scheduler_mode``.  Tests always run InProcess.
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any, Iterable, Protocol, runtime_checkable

from sqlalchemy import select

from app.scheduler.executors import has_executor, run_with_status

logger = logging.getLogger(__name__)


@runtime_checkable
class Scheduler(Protocol):
    """
    Scheduler contract — two public methods.

    ``submit`` is called when a task has just been created (or re-queued) and
    is ready to run *subject to DAG gating*.  The Scheduler is responsible for
    checking ``depends_on`` and either starting the work or parking the task.

    ``on_task_done`` is called after any task transitions to a terminal state;
    DAG-aware schedulers use it as a hook to re-evaluate downstream tasks.
    """

    async def submit(self, platform_task_id: str) -> None: ...
    async def on_task_done(self, platform_task_id: str) -> None: ...


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _split_payload_ref(payload_ref: str | None) -> tuple[str | None, str | None]:
    if not payload_ref:
        return None, None
    payload_kind, _, domain_id = payload_ref.partition(":")
    return payload_kind or None, domain_id or None


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


# DAG gate constants
_TERMINAL_SUCCESS = {"SUCCESS", "COMPLETED"}
_TERMINAL_FAILURE = {"FAILED", "CANCELLED"}
_ACTIVE_STATUSES  = {"PENDING", "QUEUED", "RUNNING", "RETRY"}


async def _gate_upstream(depends_on: Iterable[str] | None) -> tuple[str, list[str]]:
    """
    Inspect upstream tasks and return a gate decision.

    Returns ``(decision, failures)`` where ``decision`` is one of:
      * ``"ready"``     — every upstream is SUCCESS; proceed.
      * ``"blocked"``   — at least one upstream is still active; park.
      * ``"cascaded"``  — at least one upstream is in a terminal failure
                         state; the dependent should be cancelled with the
                         failing upstream IDs surfaced in ``failures``.
    """
    if not depends_on:
        return "ready", []
    from app.models.database import PlatformTask, async_session_factory

    async with async_session_factory() as db:
        rows = await db.execute(
            select(PlatformTask.id, PlatformTask.status)
            .where(PlatformTask.id.in_(list(depends_on)))
        )
        statuses = {tid: status for tid, status in rows.all()}

    failures = [tid for tid, st in statuses.items() if (st or "").upper() in _TERMINAL_FAILURE]
    if failures:
        return "cascaded", failures

    missing = [tid for tid in depends_on if tid not in statuses]
    if missing:
        # Unknown upstream → treat as blocked rather than silently dropping;
        # the submit caller will log and leave the task in PENDING.
        return "blocked", missing

    if all((statuses.get(tid) or "").upper() in _TERMINAL_SUCCESS for tid in depends_on):
        return "ready", []

    return "blocked", []


async def _park_task_pending(platform_task_id: str, reason: str) -> None:
    """Leave a task in PENDING with a note explaining the wait."""
    from app.models.database import PlatformTask, async_session_factory
    async with async_session_factory() as db:
        row = await db.execute(
            select(PlatformTask).where(PlatformTask.id == platform_task_id)
        )
        task = row.scalar_one_or_none()
        if task is None:
            return
        # Don't overwrite an already-running task — only the PENDING path matters.
        if (task.status or "").upper() == "PENDING":
            task.error_message = reason[:2000]
            await db.commit()


async def _cascade_cancel(platform_task_id: str, failing_upstream: list[str]) -> None:
    """Mark a blocked task as CANCELLED when any upstream has failed."""
    from app.models.database import PlatformTask, async_session_factory
    async with async_session_factory() as db:
        row = await db.execute(
            select(PlatformTask).where(PlatformTask.id == platform_task_id)
        )
        task = row.scalar_one_or_none()
        if task is None:
            return
        task.status = "CANCELLED"
        task.finished_at = _utcnow()
        task.error_message = f"Upstream task(s) failed: {', '.join(failing_upstream)}"
        await db.commit()


async def _find_dependents(platform_task_id: str) -> list[str]:
    """
    Return the ids of PlatformTasks whose ``depends_on`` list includes
    ``platform_task_id`` and that are still in an active status.

    We do the containment check in Python rather than in SQL because
    ``depends_on`` is a JSON column and cross-backend JSON contains SQL is
    messy — the cardinality of active tasks at any moment is small, so the
    linear scan is cheap and portable.
    """
    from app.models.database import PlatformTask, async_session_factory
    async with async_session_factory() as db:
        rows = await db.execute(
            select(PlatformTask.id, PlatformTask.depends_on, PlatformTask.status)
            .where(PlatformTask.status.in_(["PENDING", "QUEUED"]))
        )
        dependents: list[str] = []
        for tid, deps, _st in rows.all():
            if deps and platform_task_id in deps:
                dependents.append(tid)
    return dependents


# ---------------------------------------------------------------------------
# InProcessScheduler (dev / tests / single-process deployments)
# ---------------------------------------------------------------------------

class InProcessScheduler:
    """
    Run executors on the current asyncio loop (via ``asyncio.create_task``).

    Phase 5 behaviour:
      * DAG gate live — blocked tasks stay PENDING; ``on_task_done`` resubmits.
      * Unknown ``kind`` falls back to the legacy ``task_runner.dispatch_platform_task``
        for forward-compat with services that haven't migrated to the registry.
    """

    async def submit(self, platform_task_id: str) -> None:
        kind, payload_ref, depends_on = await _load_task_fields(platform_task_id)
        if kind is None:
            raise ValueError(f"PlatformTask {platform_task_id!r} not found")

        decision, upstream = await _gate_upstream(depends_on)
        if decision == "blocked":
            logger.info(
                "DAG gate: task %s blocked waiting on %s", platform_task_id, upstream or depends_on,
            )
            await _park_task_pending(
                platform_task_id, f"Waiting on upstream: {list(depends_on or [])}"
            )
            return
        if decision == "cascaded":
            logger.warning(
                "DAG gate: task %s cancelled — upstream %s failed",
                platform_task_id, upstream,
            )
            await _cascade_cancel(platform_task_id, upstream)
            return

        payload_kind, domain_id = _split_payload_ref(payload_ref)
        if not payload_kind or not domain_id:
            raise ValueError(
                f"PlatformTask {platform_task_id!r} has invalid payload_ref {payload_ref!r}"
            )

        if has_executor(kind):
            asyncio.create_task(
                _run_and_propagate(self, kind, domain_id, platform_task_id)
            )
            return

        # Legacy fallback — kept for services that haven't migrated yet.
        logger.debug(
            "No executor registered for kind=%r; falling back to legacy dispatch_platform_task",
            kind,
        )
        from app.scheduler.task_runner import dispatch_platform_task
        await dispatch_platform_task(
            platform_task_id=platform_task_id,
            kind=kind,
            payload_ref=payload_ref,
            priority=5,
        )

    async def on_task_done(self, platform_task_id: str) -> None:
        """
        Re-submit every blocked task whose upstream set is now satisfied.

        Called by ``run_with_status`` after a task reaches a terminal state,
        so this runs in the worker path — exactly one submission per
        transition.  We deliberately iterate sequentially rather than using
        gather() so a single failing submit() doesn't short-circuit the rest.
        """
        dependents = await _find_dependents(platform_task_id)
        for dep_id in dependents:
            try:
                await self.submit(dep_id)
            except Exception as exc:  # pragma: no cover — defensive
                logger.exception("Re-submit failed for dependent %s: %s", dep_id, exc)


# ---------------------------------------------------------------------------
# CeleryScheduler (Phase 5 production default when SCHEDULER_MODE=celery)
# ---------------------------------------------------------------------------

class CeleryScheduler:
    """
    Route PlatformTask submissions through Celery workers.

    Per-kind queue routing:
      * ``train``, ``dl_train`` → queue ``train``
      * ``explain``             → queue ``explain``
      * ``ts_forecast``         → queue ``forecast``
      * anything else          → default queue, generic task

    The underlying Celery task is ``run_platform_task_generic`` — it dispatches
    to the executor registry, so we don't need a per-kind Celery task body.
    """

    _KIND_TO_QUEUE = {
        "train":       "train",
        "dl_train":    "train",
        "explain":     "explain",
        "ts_forecast": "forecast",
        "forecast":    "forecast",
    }

    async def submit(self, platform_task_id: str) -> None:
        kind, payload_ref, depends_on = await _load_task_fields(platform_task_id)
        if kind is None:
            raise ValueError(f"PlatformTask {platform_task_id!r} not found")

        decision, upstream = await _gate_upstream(depends_on)
        if decision == "blocked":
            logger.info(
                "DAG gate (celery): task %s blocked waiting on %s",
                platform_task_id, upstream or depends_on,
            )
            await _park_task_pending(
                platform_task_id, f"Waiting on upstream: {list(depends_on or [])}"
            )
            return
        if decision == "cascaded":
            await _cascade_cancel(platform_task_id, upstream)
            return

        payload_kind, domain_id = _split_payload_ref(payload_ref)
        if not payload_kind or not domain_id:
            raise ValueError(
                f"PlatformTask {platform_task_id!r} has invalid payload_ref {payload_ref!r}"
            )

        queue = self._KIND_TO_QUEUE.get(kind, "celery")
        # Lazy import — Celery app construction hits Redis eagerly on some
        # brokers, and tests run with SCHEDULER_MODE=inprocess.
        from app.scheduler.celery_tasks import run_platform_task_generic
        run_platform_task_generic.apply_async(
            args=[platform_task_id],
            queue=queue,
        )

    async def on_task_done(self, platform_task_id: str) -> None:
        """Re-submit dependents whose upstream constraint is now satisfied."""
        dependents = await _find_dependents(platform_task_id)
        for dep_id in dependents:
            try:
                await self.submit(dep_id)
            except Exception as exc:  # pragma: no cover — defensive
                logger.exception("Re-submit failed for dependent %s: %s", dep_id, exc)


# ---------------------------------------------------------------------------
# Run wrapper — calls executor and notifies scheduler of completion
# ---------------------------------------------------------------------------

async def _run_and_propagate(
    scheduler: Scheduler, kind: str, domain_id: str, platform_task_id: str
) -> None:
    """
    Execute a task and, regardless of outcome, fire ``on_task_done`` so
    dependent tasks get a chance to run.  ``run_with_status`` swallows
    executor exceptions and writes FAILED — we still want to unblock
    dependents (which will cascade-cancel when they see the failure).
    """
    try:
        await run_with_status(kind, domain_id, platform_task_id)
    finally:
        try:
            await scheduler.on_task_done(platform_task_id)
        except Exception:  # pragma: no cover
            logger.exception("on_task_done raised after %s", platform_task_id)


# ---------------------------------------------------------------------------
# Module-level singleton accessor
# ---------------------------------------------------------------------------

_scheduler: Scheduler | None = None


def get_scheduler() -> Scheduler:
    global _scheduler
    if _scheduler is None:
        _scheduler = _build_scheduler()
    return _scheduler


def set_scheduler(scheduler: Scheduler) -> None:
    """Dependency-injection hook for tests."""
    global _scheduler
    _scheduler = scheduler


def _build_scheduler() -> Scheduler:
    """
    Factory — reads ``settings.scheduler_mode`` to pick an implementation.

    Defaults to ``InProcessScheduler`` so unit tests, single-process dev,
    and legacy deployments keep working.  Ops flip to Celery via
    ``SCHEDULER_MODE=celery`` in the env file once a Redis + worker pair
    is part of the deployment.
    """
    from app.config import get_settings
    mode = (get_settings().scheduler_mode or "inprocess").lower()
    if mode == "celery":
        logger.info("Scheduler: using CeleryScheduler (SCHEDULER_MODE=celery)")
        return CeleryScheduler()
    if mode not in ("inprocess", "celery"):
        logger.warning(
            "Unknown SCHEDULER_MODE=%r; falling back to inprocess", mode,
        )
    return InProcessScheduler()


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------

async def _load_task_fields(
    platform_task_id: str,
) -> tuple[str | None, str | None, Any]:
    """Return ``(kind, payload_ref, depends_on)`` or ``(None, None, None)``."""
    # Lazy import so tests can swap out ``async_session_factory`` via
    # ``unittest.mock.patch("app.models.database.async_session_factory", ...)``.
    from app.models.database import PlatformTask, async_session_factory

    async with async_session_factory() as db:
        result = await db.execute(
            select(PlatformTask.kind, PlatformTask.payload_ref, PlatformTask.depends_on)
            .where(PlatformTask.id == platform_task_id)
        )
        row = result.first()
    if row is None:
        return None, None, None
    return row[0], row[1], row[2]

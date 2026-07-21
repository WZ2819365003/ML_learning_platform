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

M2 scheduling contract
======================
* ``CeleryScheduler`` — routes submissions through ``apply_async`` so
  executors run in a separate worker process (essential for GPU-heavy DL
  training that must not block the FastAPI event loop).
* **DAG gate is live** — submissions check ``depends_on``; blocked tasks
  stay in ``PENDING`` and are resubmitted by ``on_task_done`` once every
  upstream reports SUCCESS.  The gate treats FAILED/CANCELLED upstream as
  "blocked forever" — a dependent task is marked CANCELLED rather than
  orphaned.
* ``get_scheduler(kind)`` selects Celery per kind through ``CELERY_KINDS``;
  ``SCHEDULER_MODE=celery`` remains the all-kinds compatibility override.
* Every migrated submission happens after the domain and PlatformTask rows
  commit. Celery acknowledgements are stored in ``celery_task_id`` and stale
  rows without an acknowledgement are repaired by the reconciler.
* V3 in-process trials deliberately retain ``_execute_single_trial`` as their
  writeback wrapper. Cross-process run writeback and distributed Optuna state
  remain later M2c/M2d work, so deployments should gray Celery by safe kinds
  (``explain`` first) rather than enabling all training kinds immediately.
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Iterable, Protocol, runtime_checkable

from sqlalchemy import select, update

from app.scheduler.executors import has_executor, run_with_status
from app.scheduler.queues import DEFAULT, KIND_TO_QUEUE

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

    async def submit(self, platform_task_id: str) -> asyncio.Task | None: ...
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

    async def submit(self, platform_task_id: str) -> asyncio.Task | None:
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

        v3_trial = await _load_v3_trial_context(platform_task_id)
        if v3_trial is not None:
            run_id, experiment_id, modeling_task_id, strategy_type = v3_trial
            return asyncio.create_task(
                _run_v3_trial_and_propagate(
                    self,
                    kind,
                    domain_id,
                    platform_task_id,
                    run_id,
                    experiment_id,
                    modeling_task_id,
                    strategy_type,
                )
            )

        if has_executor(kind):
            return asyncio.create_task(
                _run_and_propagate(self, kind, domain_id, platform_task_id)
            )

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
        return None

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

        queue = KIND_TO_QUEUE.get(kind, DEFAULT)
        # Lazy import — Celery app construction hits Redis eagerly on some
        # brokers, and tests run with SCHEDULER_MODE=inprocess.
        from app.scheduler.celery_tasks import run_platform_task_generic
        try:
            async_result = run_platform_task_generic.apply_async(
                args=[platform_task_id],
                queue=queue,
            )
        except Exception as exc:  # broker outage: leave row for reconciler
            logger.error(
                "Celery submission failed for PlatformTask %s: %s",
                platform_task_id,
                exc,
            )
            return

        celery_task_id = getattr(async_result, "id", None)
        if celery_task_id:
            try:
                await _store_celery_task_id(platform_task_id, str(celery_task_id))
            except Exception:
                logger.exception(
                    "Celery task %s was published but its id could not be "
                    "persisted for PlatformTask %s; reconciler will retry",
                    celery_task_id,
                    platform_task_id,
                )
        else:
            logger.warning(
                "Celery submission for PlatformTask %s returned no task id; "
                "reconciler will retry",
                platform_task_id,
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


async def _run_v3_trial_and_propagate(
    scheduler: Scheduler,
    kind: str,
    domain_id: str,
    platform_task_id: str,
    run_id: str,
    experiment_id: str,
    modeling_task_id: str | None,
    strategy_type: str,
) -> dict[str, Any]:
    """Keep the existing V3 trial writeback sequence on the in-process path."""
    from app.services.tuning_service import _execute_single_trial

    try:
        # M2c: batch finalisation AND DAG propagation are owned by
        # ``run_writeback.complete_platform_task``. Propagating here as well is
        # not a harmless no-op — ``on_task_done`` has no atomic claim, so two
        # callers can both observe a dependent as PENDING and submit it twice.
        # ``_execute_single_trial`` funnels every exit (including unexpected
        # exceptions) through the write-back, so propagation is guaranteed.
        return await _execute_single_trial(
            domain_id,
            platform_task_id,
            run_id,
            experiment_id,
            kind=kind,
        )
    except Exception:
        logger.exception("V3 trial wrapper failed for %s", platform_task_id)
        raise


# ---------------------------------------------------------------------------
# Module-level singleton accessor
# ---------------------------------------------------------------------------

_scheduler_override: Scheduler | None = None
_inprocess_scheduler = InProcessScheduler()
_celery_scheduler = CeleryScheduler()


def get_scheduler(kind: str) -> Scheduler:
    """Return the scheduler selected for one PlatformTask kind."""
    if _scheduler_override is not None:
        return _scheduler_override
    return _build_scheduler(kind)


def set_scheduler(scheduler: Scheduler | None) -> None:
    """Dependency-injection hook for tests; pass ``None`` to clear it."""
    global _scheduler_override
    _scheduler_override = scheduler


def _build_scheduler(kind: str) -> Scheduler:
    """
    Factory — reads the global mode and per-kind allowlist.

    Defaults to ``InProcessScheduler`` so unit tests, single-process dev,
    and legacy deployments keep working. ``CELERY_KINDS`` enables a safe
    per-kind rollout; the legacy global mode routes all kinds to Celery.
    """
    from app.config import get_settings
    settings = get_settings()
    mode = (settings.scheduler_mode or "inprocess").lower()
    if mode == "celery":
        logger.info("Scheduler: using CeleryScheduler (SCHEDULER_MODE=celery)")
        return _celery_scheduler
    if mode not in ("inprocess", "celery"):
        logger.warning(
            "Unknown SCHEDULER_MODE=%r; falling back to inprocess", mode,
        )
    celery_kinds = {
        item.strip()
        for item in (getattr(settings, "celery_kinds", "") or "").split(",")
        if item.strip()
    }
    if kind in celery_kinds:
        return _celery_scheduler
    return _inprocess_scheduler


async def reconcile_queued_tasks(
    *,
    older_than: timedelta = timedelta(minutes=2),
    now: datetime | None = None,
    limit: int = 100,
) -> list[str]:
    """Re-submit stale Celery-routed rows left without a broker task id.

    This is an at-least-once outbox repair. In-process kinds are skipped so
    the default ``CELERY_KINDS=''`` startup remains behavior-neutral.
    """
    from app.models.database import PlatformTask, async_session_factory

    cutoff = (now or _utcnow()) - older_than
    async with async_session_factory() as db:
        rows = await db.execute(
            select(PlatformTask.id, PlatformTask.kind)
            .where(
                PlatformTask.status == "QUEUED",
                PlatformTask.celery_task_id.is_(None),
                PlatformTask.queued_at.is_not(None),
                PlatformTask.queued_at <= cutoff,
            )
            .order_by(PlatformTask.queued_at, PlatformTask.id)
            .limit(limit)
        )
        candidates = rows.all()

    submitted: list[str] = []
    for task_id, kind in candidates:
        scheduler = get_scheduler(kind)
        if not isinstance(scheduler, CeleryScheduler):
            continue
        await scheduler.submit(task_id)
        submitted.append(task_id)
    return submitted


async def recover_stalled_tasks(
    *,
    older_than: timedelta = timedelta(hours=6),
    now: datetime | None = None,
    limit: int = 100,
) -> list[str]:
    """Re-arm Celery tasks stuck non-terminal with no one left to finish them.

    ``reconcile_queued_tasks`` only repairs rows that never reached the broker.
    It cannot help the other hole: a worker that *executed* the task but died
    before the terminal commit landed. Those rows sit at RUNNING/RETRY with a
    ``celery_task_id`` already set, so nothing re-drives them and the trial
    hangs forever. That is exactly the tail a ``WritebackError`` leaves behind
    when every write-back attempt fails.

    We cannot recover the lost *result* — it was never persisted anywhere
    durable — so the only honest recovery is re-execution. The row is reset to
    QUEUED and resubmitted, and the normal claim/complete path takes over.

    Consequences, stated precisely:

    * This is at-least-once. A genuine trial that runs longer than
      ``older_than`` will be executed a second time. The threshold must sit
      above the training hard time limit; the default is deliberately generous.
    * Only the **terminal write-back is idempotent**. Whichever attempt commits
      first wins and the loser's ``complete_platform_task`` is a no-op. That is
      NOT the same as the duplicate being harmless: the executors are not
      attempt-fenced, so two attempts write the same domain task, the same log
      files and the same ``storage/models/{task_id}.joblib``. A loser that
      finishes after the winner can overwrite the winner's artifact, leaving
      the recorded metrics describing a model that is no longer on disk.
      Artifact fencing is required before Celery ``train`` is enabled in
      production — see the known-limits list in the M2c commit.
    * The re-arm is a compare-and-set on ``status`` plus the ``attempt_token``
      we observed. A task that reached a terminal state, or that was re-claimed
      into a *fresh* attempt between the scan and the write, no longer matches
      and is left alone. The token exists precisely because the obvious
      candidates cannot do this job: Celery reuses one task id across retries,
      and ``started_at`` is stamped only on the first claim.
    """
    from app.models.database import ExperimentRun, PlatformTask, async_session_factory

    stalled = ["RUNNING", "RETRY"]
    cutoff = (now or _utcnow()) - older_than
    async with async_session_factory() as db:
        rows = await db.execute(
            select(
                PlatformTask.id,
                PlatformTask.kind,
                PlatformTask.attempt_token,
            )
            .where(
                PlatformTask.status.in_(stalled),
                PlatformTask.started_at.is_not(None),
                PlatformTask.started_at <= cutoff,
            )
            .order_by(PlatformTask.started_at, PlatformTask.id)
            .limit(limit)
        )
        candidates = rows.all()

    recovered: list[str] = []
    for task_id, kind, seen_attempt in candidates:
        scheduler = get_scheduler(kind)
        if not isinstance(scheduler, CeleryScheduler):
            continue

        # Compare-and-set on the attempt we observed, not just on status.
        # Between the scan and here the task may have been re-claimed; the
        # claim mints a new attempt_token, so a fresh attempt no longer
        # matches and is left running.
        identity = [
            PlatformTask.attempt_token.is_(None)
            if seen_attempt is None
            else PlatformTask.attempt_token == seen_attempt
        ]

        async with async_session_factory() as db:
            async with db.begin():
                result = await db.execute(
                    update(PlatformTask)
                    .where(
                        PlatformTask.id == task_id,
                        PlatformTask.status.in_(stalled),
                        *identity,
                    )
                    .values(
                        status="QUEUED",
                        attempt_token=None,
                        celery_task_id=None,
                        queued_at=_utcnow(),
                        started_at=None,
                        finished_at=None,
                    )
                )
                if not result.rowcount:
                    # Finished, or re-claimed into a fresh attempt, since the
                    # scan. Either way there is nothing stalled to recover.
                    continue
                await db.execute(
                    update(ExperimentRun)
                    .where(
                        ExperimentRun.task_id == task_id,
                        ExperimentRun.status.in_(stalled),
                    )
                    .values(status="PENDING", started_at=None)
                )

        await scheduler.submit(task_id)
        recovered.append(task_id)
        logger.warning("Recovered stalled task %s (kind=%s) by re-queueing", task_id, kind)
    return recovered


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


async def _store_celery_task_id(platform_task_id: str, celery_task_id: str) -> None:
    """Persist the broker acknowledgement after ``apply_async`` returns."""
    from app.models.database import PlatformTask, async_session_factory

    async with async_session_factory() as db:
        task = await db.get(PlatformTask, platform_task_id)
        if task is None:
            logger.error(
                "Cannot persist Celery id %s: PlatformTask %s disappeared",
                celery_task_id,
                platform_task_id,
            )
            return
        task.celery_task_id = celery_task_id
        await db.commit()


async def _load_v3_trial_context(
    platform_task_id: str,
) -> tuple[str, str, str | None, str] | None:
    """Return V3 workbench run and batch context for in-process writeback."""
    from app.models.database import (
        ExperimentRun,
        PlatformExperiment,
        async_session_factory,
    )

    async with async_session_factory() as db:
        row = (
            await db.execute(
                select(
                    ExperimentRun.id,
                    ExperimentRun.experiment_id,
                    PlatformExperiment.config,
                    PlatformExperiment.modeling_task_id,
                    PlatformExperiment.strategy_type,
                )
                .join(
                    PlatformExperiment,
                    PlatformExperiment.id == ExperimentRun.experiment_id,
                )
                .where(ExperimentRun.task_id == platform_task_id)
            )
        ).first()
    if row is None or (row[2] or {}).get("submitted_from") != "v3_workbench":
        return None
    return row[0], row[1], row[3], row[4]

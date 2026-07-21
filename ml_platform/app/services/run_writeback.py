"""Single, idempotent completion path for PlatformTask + ExperimentRun (M2c).

Before M2c the write-back lived inline in ``tuning_service._execute_single_trial``
(RUNNING marks → await executor → normalise metrics → update Run → update
PlatformTask → mirror logs). That only works when the executor is awaited in
the same coroutine, so a Celery worker finishing a trial left the
``ExperimentRun`` stuck in RUNNING forever and ``_finalise_batch`` never fired.

Both paths now converge here:

    in-process : _execute_single_trial     → complete_platform_task(...)
    celery     : _execute_generic          → complete_platform_task(...)

Guarantees (and their limits)
-----------------------------
* **Terminal state is committed atomically.** The Run and its PlatformTask are
  locked and written inside ONE transaction, so a duplicate delivery can never
  leave Run=SUCCESS / Task=FAILED. Whoever loses the race sees the row already
  terminal and returns the canonical state without touching it.
* **Authoritative writes raise.** If the terminal commit fails we propagate, so
  Celery retries instead of reporting a success that never landed. Only the
  observability tail (log mirroring) is best-effort.
* **Sole DAG owner.** Downstream propagation happens here and nowhere else.
* **Retry-aware.** A failure with retries remaining parks the run in RETRY and
  writes no terminal state or terminal error.

Known limits (tracked, not solved here): log mirroring is check-then-insert and
is not safe against two *simultaneous* completions of the same run, and
finalise/SHAP dispatch is not crash-atomic with the terminal commit. Both need
an outbox and are out of M2c's scope.
"""
from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import func, select, update

from app.models import database as _db_module
from app.models.database import ExperimentRun, PlatformExperiment, PlatformTask

logger = logging.getLogger(__name__)

# CANCELLED is terminal too: a cancelled task must never be re-claimed and
# re-executed by a late/duplicate delivery.
_TERMINAL_STATES = {"SUCCESS", "FAILED", "CANCELLED"}


class WritebackError(RuntimeError):
    """The work finished but its result could not be recorded.

    Distinct from an execution failure on purpose: Celery's failure hooks must
    not convert "training succeeded, the commit hiccuped" into a FAILED run.
    """


def _sessions():
    """Resolve the session factory at call time.

    Tests patch ``app.models.database.async_session_factory`` with a per-test
    in-memory engine; a module-level ``from ... import async_session_factory``
    would bind the global one at import and silently write to the wrong
    database (and across event loops).
    """
    return _db_module.async_session_factory


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class CompletionOutcome:
    """What actually got committed (not what the caller asked for)."""

    platform_task_id: str
    status: str
    run_id: str | None = None
    experiment_id: str | None = None
    modeling_task_id: str | None = None
    strategy_type: str | None = None
    metrics: dict[str, Any] | None = None
    already_terminal: bool = False


async def complete_platform_task(
    platform_task_id: str,
    *,
    status: str,
    metrics: dict[str, Any] | None = None,
    error: str | None = None,
    evaluation_mode: str = "standard",
    domain_task_id: str | None = None,
    final_attempt: bool = True,
) -> CompletionOutcome:
    """Finish a PlatformTask and, when it backs a V3 trial, its ExperimentRun.

    Raises if the authoritative terminal write fails — the caller (Celery) must
    be able to retry rather than report a completion that never committed.
    """
    from app.services.tuning_service import (
        _finalise_batch,
        _mirror_logs_to_v3,
        _normalise_run_metrics,
    )

    succeeded = status == "SUCCESS"
    normalised = (
        _normalise_run_metrics(metrics or {}, evaluation_mode=evaluation_mode)
        if succeeded
        else {}
    )

    # Retryable failure: park, write no terminal state and no terminal error.
    if not succeeded and not final_attempt:
        await _park_for_retry(platform_task_id, error)
        return CompletionOutcome(platform_task_id, "RETRY")

    outcome = await _commit_terminal_state(
        platform_task_id, status=status, metrics=normalised, error=error
    )

    # A duplicate delivery loses the race: the tails already ran for the winner.
    if outcome.already_terminal:
        logger.info(
            "complete_platform_task: %s already terminal (%s); skipping tails",
            platform_task_id,
            outcome.status,
        )
        return outcome

    # ---- Post-commit tails -------------------------------------------------
    # Observability only: a failed mirror must not fail a committed run.
    if domain_task_id and outcome.run_id:
        try:
            await _mirror_logs_to_v3(domain_task_id=domain_task_id, run_id=outcome.run_id)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Log mirror failed for run %s: %s", outcome.run_id, exc)

    # bayesian_search is excluded: it creates its ExperimentRuns *incrementally*
    # (one per ask/tell iteration), so after trial 1 the batch would look like
    # done=1/total=1 and close while the study is still running. Its orchestrator
    # finalises the whole study instead — see tuning_service._run_bayesian_search
    # (and the Celery guard that refuses this combination until M2d).
    if (
        outcome.experiment_id
        and outcome.modeling_task_id
        and outcome.strategy_type != "bayesian_search"
    ):
        try:
            await _finalise_batch(outcome.experiment_id, outcome.modeling_task_id)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Batch finalisation failed for %s: %s", outcome.experiment_id, exc
            )

    # Sole DAG propagation owner — callers must not propagate again.
    await _propagate_dag(platform_task_id)
    return outcome


async def _commit_terminal_state(
    platform_task_id: str,
    *,
    status: str,
    metrics: dict[str, Any],
    error: str | None,
) -> CompletionOutcome:
    """Lock + write the Run and its PlatformTask inside one transaction.

    Either both records reach the terminal state or neither does. Exceptions
    propagate so the caller can retry.

    The terminal transition is claimed with a **conditional UPDATE whose
    rowcount decides the winner**, not with read-then-write. On MySQL the row
    locks below already serialise concurrent completions, but SQLite has no
    row locks — "single writer" serialises *writes*, not the read that a
    decision was based on. A concurrent cancel could therefore observe
    non-terminal, and both paths would write, splitting Task from Run. The CAS
    holds on every backend because the database, not a Python snapshot,
    resolves the race. SQLite is the development default, so this matters.
    """
    succeeded = status == "SUCCESS"
    now = _utcnow()

    async with _sessions()() as db:
        async with db.begin():
            is_sqlite = db.bind is not None and db.bind.dialect.name == "sqlite"

            task_stmt = select(PlatformTask).where(PlatformTask.id == platform_task_id)
            run_id_stmt = select(ExperimentRun.id).where(
                ExperimentRun.task_id == platform_task_id
            )
            if not is_sqlite:
                task_stmt = task_stmt.with_for_update()
                run_id_stmt = run_id_stmt.with_for_update()

            task = (await db.execute(task_stmt)).scalar_one_or_none()
            if task is None:
                raise ValueError(f"PlatformTask {platform_task_id} not found")
            run_id = (await db.execute(run_id_stmt)).scalar_one_or_none()

            # --- claim the terminal transition ---------------------------------
            # The Run is authoritative for V3 trials; without one the
            # PlatformTask gates on its own.
            if run_id is not None:
                run_values: dict[str, Any] = {
                    "status": status,
                    "finished_at": now,
                    # Stamp a start only if the row never got one.
                    "started_at": func.coalesce(ExperimentRun.started_at, now),
                }
                if succeeded:
                    run_values["metrics"] = metrics
                    run_values["error_message"] = None
                elif error:
                    run_values["error_message"] = str(error)[:2000]
                gate = (
                    update(ExperimentRun)
                    .where(
                        ExperimentRun.id == run_id,
                        ExperimentRun.status.notin_(list(_TERMINAL_STATES)),
                    )
                    .values(**run_values)
                    .execution_options(synchronize_session=False)
                )
            else:
                gate = (
                    update(PlatformTask)
                    .where(
                        PlatformTask.id == platform_task_id,
                        PlatformTask.status.notin_(list(_TERMINAL_STATES)),
                    )
                    .values(status=status, finished_at=now)
                    .execution_options(synchronize_session=False)
                )

            claimed = bool((await db.execute(gate)).rowcount)

            run = None
            if run_id is not None:
                run = (
                    await db.execute(
                        select(ExperimentRun).where(ExperimentRun.id == run_id)
                    )
                ).scalar_one_or_none()

            if not claimed:
                # Someone else terminalised first. Report their state verbatim.
                await db.refresh(task)
                reference = run if run is not None else task
                return await _describe(
                    db, task, run, already_terminal=True, status=reference.status
                )

            # We own the transition; finish the pair in the same transaction.
            task.status = status
            task.finished_at = now
            if succeeded:
                task.progress = 1.0
                task.metrics_snapshot = metrics
                # Clear a stale error left by an earlier failed attempt.
                task.error_message = None
            elif error:
                task.error_message = str(error)[:2000]

            return await _describe(db, task, run, already_terminal=False, status=status)


async def _describe(
    db,
    task: PlatformTask,
    run: ExperimentRun | None,
    *,
    already_terminal: bool,
    status: str,
) -> CompletionOutcome:
    """Build the outcome, resolving experiment context for the tails."""
    experiment_id = run.experiment_id if run is not None else None
    modeling_task_id = None
    strategy_type = None
    if experiment_id:
        exp = (
            await db.execute(
                select(PlatformExperiment).where(PlatformExperiment.id == experiment_id)
            )
        ).scalar_one_or_none()
        if exp is not None:
            modeling_task_id = exp.modeling_task_id
            strategy_type = exp.strategy_type
    return CompletionOutcome(
        platform_task_id=task.id,
        status=status,
        run_id=run.id if run is not None else None,
        experiment_id=experiment_id,
        modeling_task_id=modeling_task_id,
        strategy_type=strategy_type,
        metrics=dict(run.metrics or {}) if run is not None else None,
        already_terminal=already_terminal,
    )


async def _park_for_retry(platform_task_id: str, error: str | None) -> None:
    """Between Celery attempts: mark RETRY without any terminal write.

    Uses conditional UPDATEs (``WHERE status NOT IN (terminal)``) rather than
    read-then-write: a late retry hook must never drag a task that another
    delivery already committed as SUCCESS back to RETRY. The database decides,
    not a stale Python snapshot.

    The transient reason lives on the PlatformTask (per-attempt bookkeeping);
    ``ExperimentRun.error_message`` stays reserved for the final failure so a
    run that eventually succeeds carries no scary leftovers.
    """
    task_values: dict[str, Any] = {"status": "RETRY"}
    if error:
        task_values["error_message"] = str(error)[:2000]

    async with _sessions()() as db:
        async with db.begin():
            await db.execute(
                update(PlatformTask)
                .where(
                    PlatformTask.id == platform_task_id,
                    PlatformTask.status.notin_(list(_TERMINAL_STATES)),
                )
                .values(**task_values)
            )
            await db.execute(
                update(ExperimentRun)
                .where(
                    ExperimentRun.task_id == platform_task_id,
                    ExperimentRun.status.notin_(list(_TERMINAL_STATES)),
                )
                .values(status="RETRY")
            )


async def _propagate_dag(platform_task_id: str) -> None:
    """Release downstream dependents. Sole owner of this call."""
    try:
        from app.scheduler.scheduler import get_scheduler

        async with _sessions()() as db:
            kind = (
                await db.execute(
                    select(PlatformTask.kind).where(PlatformTask.id == platform_task_id)
                )
            ).scalar_one_or_none()
        if kind is None:
            return
        await get_scheduler(kind).on_task_done(platform_task_id)
    except Exception as exc:  # noqa: BLE001
        logger.warning("DAG propagation failed for %s: %s", platform_task_id, exc)


async def claim_for_execution(platform_task_id: str) -> bool:
    """Atomically claim a task for execution; False means "already finished".

    Both the PlatformTask and its ExperimentRun move to RUNNING inside ONE
    transaction, and only if neither is terminal. Callers must NOT write RUNNING
    themselves beforehand: doing so on a duplicate delivery would drag an
    already-finished task back to RUNNING and strand it there (the completion
    that follows sees ``already_terminal`` and correctly refuses to touch it).

    ``started_at`` is stamped once; ``status`` is refreshed on every attempt so
    a retried trial doesn't sit visibly in RETRY while it actually runs.
    """
    async with _sessions()() as db:
        async with db.begin():
            is_sqlite = db.bind is not None and db.bind.dialect.name == "sqlite"
            task_stmt = select(PlatformTask).where(PlatformTask.id == platform_task_id)
            run_stmt = select(ExperimentRun).where(
                ExperimentRun.task_id == platform_task_id
            )
            if not is_sqlite:
                task_stmt = task_stmt.with_for_update()
                run_stmt = run_stmt.with_for_update()

            task = (await db.execute(task_stmt)).scalar_one_or_none()
            run = (await db.execute(run_stmt)).scalar_one_or_none()

            # EITHER record being terminal blocks the claim, not just the Run.
            # The real cancel path (task_runner.cancel_task) writes CANCELLED to
            # the PlatformTask; a Run-only check would let a late delivery
            # resurrect both rows into RUNNING and re-execute cancelled work.
            for record in (task, run):
                if record is not None and record.status in _TERMINAL_STATES:
                    return False
            if task is None and run is None:
                return False

            now = _utcnow()
            if task is not None:
                task.status = "RUNNING"
                # A fresh token on EVERY claim — this is what makes an attempt
                # identifiable. started_at is deliberately stamped once (it is
                # the user-visible start), and Celery reuses one task id across
                # retries, so neither can tell attempt N from attempt N+1.
                task.attempt_token = uuid.uuid4().hex
                if task.started_at is None:
                    task.started_at = now
            if run is not None:
                run.status = "RUNNING"
                if run.started_at is None:
                    run.started_at = now
            return True


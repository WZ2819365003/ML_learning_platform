"""
V3 Unified Platform Task API — /api/platform/tasks

Provides CRUD + status polling + retry/cancel for PlatformTask records.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import case, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.database import (
    ExperimentRun,
    ModelingTask,
    PlatformExperiment,
    PlatformTask,
    get_db,
)
from app.scheduler.task_runner import cancel_task, dispatch_platform_task, retry_task
from app.services.platform_task_detail_service import get_platform_task_detail

router = APIRouter(prefix="/platform/tasks", tags=["Platform Tasks V3"])


# ---------------------------------------------------------------------------
# Serializer
# ---------------------------------------------------------------------------

def _serialize(task: PlatformTask) -> dict[str, Any]:
    return {
        "id": task.id,
        "kind": task.kind,
        "status": task.status,
        "priority": task.priority,
        "celery_task_id": task.celery_task_id,
        "worker_id": task.worker_id,
        "retry_count": task.retry_count,
        "max_retries": task.max_retries,
        "payload_ref": task.payload_ref,
        "progress": task.progress,
        "logs_uri": task.logs_uri,
        "metrics_uri": task.metrics_uri,
        "artifacts_uri": task.artifacts_uri,
        "metrics_snapshot": task.metrics_snapshot,
        "error_message": task.error_message,
        "queued_at": task.queued_at.isoformat() if task.queued_at else None,
        "started_at": task.started_at.isoformat() if task.started_at else None,
        "finished_at": task.finished_at.isoformat() if task.finished_at else None,
        "created_at": task.created_at.isoformat() if task.created_at else None,
        "updated_at": task.updated_at.isoformat() if task.updated_at else None,
    }


# ---------------------------------------------------------------------------
# Stats  (must be registered before /{task_id} to avoid path-parameter capture)
# ---------------------------------------------------------------------------

@router.get("/stats", summary="Global task counts grouped by status")
async def platform_task_stats(
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    """Return total task counts for every status in a single query."""
    rows = await db.execute(
        select(PlatformTask.status, func.count(PlatformTask.id))
        .group_by(PlatformTask.status)
    )
    counts: dict[str, int] = {status: cnt for status, cnt in rows.all()}
    return {
        "total": sum(counts.values()),
        "by_status": counts,
        # Convenience roll-ups used by the frontend summary cards
        "running":   counts.get("RUNNING", 0),
        "queued":    counts.get("QUEUED", 0) + counts.get("PENDING", 0),
        "succeeded": counts.get("SUCCESS", 0),
        "failed":    counts.get("FAILED", 0) + counts.get("RETRY", 0),
    }


# ---------------------------------------------------------------------------
# Tree  (hierarchical view: ModelingTask → Experiment → Run/Task)
# ---------------------------------------------------------------------------

def _iso(dt) -> str | None:
    return dt.isoformat() if dt else None


@router.get(
    "/tree",
    summary="Hierarchical task view — ModelingTask ▸ Experiment ▸ Run(+PlatformTask)",
)
async def platform_task_tree(
    page: int = Query(1, ge=1),
    page_size: int = Query(10, ge=1, le=50),
    status: str | None = Query(None, description="Filter modeling tasks by status"),
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    """
    Return a paginated list of modeling tasks with nested experiments and runs.

    Structure:
        {
          items: [{
            id, name, status, dataset_name, created_at, objective_metric,
            stats: { total_runs, success, failed, running, queued },
            experiments: [{
              id, name, strategy_type, status, created_at,
              run_count, success_count, failed_count,
              runs: [{
                id, trial_no, rank, status, metrics, params,
                task: { id, kind, status, progress, retry_count, error_message }
              }]
            }]
          }],
          total, page, page_size
        }

    This powers the hierarchical Task Center: top-level rows are modeling tasks
    (the user's mental "job"), expand to see parallel experiment batches, and
    expand further to see individual runs + their scheduler state.  Bare
    PlatformTasks not associated with an ExperimentRun (e.g. standalone
    forecast/predict tasks) are surfaced separately by the flat /list endpoint.
    """
    # --- 1. Paginate modeling tasks
    stmt = select(ModelingTask)
    count_stmt = select(func.count(ModelingTask.id))
    if status:
        stmt = stmt.where(ModelingTask.status == status.upper())
        count_stmt = count_stmt.where(ModelingTask.status == status.upper())

    total = (await db.execute(count_stmt)).scalar_one()
    rows = await db.execute(
        stmt.order_by(ModelingTask.created_at.desc())
        .offset((page - 1) * page_size)
        .limit(page_size)
    )
    modeling_tasks = rows.scalars().all()
    if not modeling_tasks:
        return {"items": [], "total": total, "page": page, "page_size": page_size}

    mt_ids = [mt.id for mt in modeling_tasks]

    # --- 2. Load all experiments for these modeling tasks in one query
    exp_rows = await db.execute(
        select(PlatformExperiment)
        .where(PlatformExperiment.modeling_task_id.in_(mt_ids))
        .order_by(PlatformExperiment.created_at.desc())
    )
    experiments = exp_rows.scalars().all()
    exp_ids = [e.id for e in experiments]

    # --- 3. Load all runs in one query
    run_rows: list[ExperimentRun] = []
    if exp_ids:
        run_rows = list((
            await db.execute(
                select(ExperimentRun)
                .where(ExperimentRun.experiment_id.in_(exp_ids))
                .order_by(
                    ExperimentRun.rank.is_(None),
                    ExperimentRun.rank.asc(),
                    ExperimentRun.created_at.asc(),
                )
            )
        ).scalars().all())

    # --- 4. Load platform tasks referenced by runs
    task_ids = [r.task_id for r in run_rows if r.task_id]
    platform_tasks_by_id: dict[str, PlatformTask] = {}
    if task_ids:
        ptasks = (
            await db.execute(
                select(PlatformTask).where(PlatformTask.id.in_(task_ids))
            )
        ).scalars().all()
        platform_tasks_by_id = {p.id: p for p in ptasks}

    # --- 5. Index runs by experiment
    runs_by_exp: dict[str, list[ExperimentRun]] = {}
    for r in run_rows:
        runs_by_exp.setdefault(r.experiment_id, []).append(r)

    # --- 6. Assemble the tree
    def _serialize_run(run: ExperimentRun) -> dict[str, Any]:
        task = platform_tasks_by_id.get(run.task_id) if run.task_id else None
        return {
            "id": run.id,
            "trial_no": run.trial_no,
            "rank": run.rank,
            "status": run.status,
            "metrics": run.metrics or {},
            "params": run.params or {},
            "source_experiment_type": run.source_experiment_type,
            "created_at": _iso(run.created_at),
            "started_at": _iso(run.started_at),
            "finished_at": _iso(run.finished_at),
            "task": {
                "id": task.id,
                "kind": task.kind,
                "status": task.status,
                "progress": task.progress,
                "retry_count": task.retry_count,
                "max_retries": task.max_retries,
                "worker_id": task.worker_id,
                "error_message": task.error_message,
                "queued_at": _iso(task.queued_at),
                "started_at": _iso(task.started_at),
                "finished_at": _iso(task.finished_at),
            } if task else None,
        }

    def _experiment_stats(runs: list[ExperimentRun]) -> dict[str, int]:
        acc = {"total": 0, "success": 0, "failed": 0, "running": 0, "queued": 0}
        for r in runs:
            acc["total"] += 1
            st = (r.status or "").upper()
            if st == "SUCCESS":
                acc["success"] += 1
            elif st == "FAILED":
                acc["failed"] += 1
            elif st == "RUNNING":
                acc["running"] += 1
            elif st in ("PENDING", "QUEUED"):
                acc["queued"] += 1
        return acc

    items: list[dict[str, Any]] = []
    for mt in modeling_tasks:
        mt_exps = [e for e in experiments if e.modeling_task_id == mt.id]
        mt_total = {"total_runs": 0, "success": 0, "failed": 0, "running": 0, "queued": 0}

        exp_payloads: list[dict[str, Any]] = []
        for exp in mt_exps:
            exp_runs = runs_by_exp.get(exp.id, [])
            exp_stats = _experiment_stats(exp_runs)
            mt_total["total_runs"] += exp_stats["total"]
            mt_total["success"] += exp_stats["success"]
            mt_total["failed"] += exp_stats["failed"]
            mt_total["running"] += exp_stats["running"]
            mt_total["queued"] += exp_stats["queued"]

            exp_payloads.append({
                "id": exp.id,
                "name": exp.name,
                "strategy_type": getattr(exp, "strategy_type", None),
                "kind": exp.kind,
                "status": exp.status,
                "objective_metric": exp.objective_metric,
                "objective_direction": exp.objective_direction,
                "selected_models": getattr(exp, "selected_models", None) or [],
                "created_at": _iso(exp.created_at),
                "finished_at": _iso(exp.finished_at),
                "stats": exp_stats,
                "runs": [_serialize_run(r) for r in exp_runs],
            })

        items.append({
            "id": mt.id,
            "name": mt.name,
            "description": mt.description,
            "status": mt.status,
            "dataset_id": mt.dataset_id,
            "dataset_name": mt.dataset_name,
            "task_type": mt.task_type,
            "objective_metric": mt.objective_metric,
            "objective_direction": mt.objective_direction,
            "best_run_id": mt.best_run_id,
            "created_at": _iso(mt.created_at),
            "finished_at": _iso(mt.finished_at),
            "stats": mt_total,
            "experiments": exp_payloads,
        })

    return {
        "items": items,
        "total": total,
        "page": page,
        "page_size": page_size,
    }


# ---------------------------------------------------------------------------
# List
# ---------------------------------------------------------------------------

@router.get("/", summary="List platform tasks")
async def list_platform_tasks(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=200),
    kind: str | None = Query(None),
    status: str | None = Query(None),
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    stmt = select(PlatformTask)
    count_stmt = select(func.count(PlatformTask.id))

    if kind:
        stmt = stmt.where(PlatformTask.kind == kind)
        count_stmt = count_stmt.where(PlatformTask.kind == kind)
    if status:
        stmt = stmt.where(PlatformTask.status == status.upper())
        count_stmt = count_stmt.where(PlatformTask.status == status.upper())

    total = (await db.execute(count_stmt)).scalar_one()
    rows = await db.execute(
        stmt.order_by(PlatformTask.created_at.desc())
        .offset((page - 1) * page_size)
        .limit(page_size)
    )
    tasks = rows.scalars().all()
    return {
        "items": [_serialize(t) for t in tasks],
        "total": total,
        "page": page,
        "page_size": page_size,
    }


# ---------------------------------------------------------------------------
# Detail (orphan-task drawer)  ── must precede /{task_id} to avoid capture
# ---------------------------------------------------------------------------

@router.get(
    "/{task_id}/detail",
    summary="Enriched PlatformTask detail — resolves payload_ref and tails logs",
)
async def platform_task_detail(
    task_id: str,
    log_limit: int = Query(200, ge=1, le=2000),
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    """
    Backs the ``OrphanTaskDetailDrawer`` in TaskCenter's 孤立任务 tab.

    The response wraps the base PlatformTask serialization with:
      * ``source_label`` — human-readable tag (e.g. "时序预测") derived from payload_ref
      * ``domain_kind`` / ``domain_id`` — parsed prefix of payload_ref
      * ``domain`` — per-kind summary dict (TrainingTask, DLTrainingTask,
        ExperimentRun, TimeSeriesForecastTask) or ``None`` if the row is
        gone.  Individual resolver failures are swallowed so the drawer
        still opens.
      * ``recent_logs`` — tail of ``storage/logs/{domain_id}.log``
    """
    return await get_platform_task_detail(db, task_id, log_limit=log_limit)


# ---------------------------------------------------------------------------
# Get
# ---------------------------------------------------------------------------

@router.get("/{task_id}", summary="Get a platform task")
async def get_platform_task(
    task_id: str,
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    result = await db.execute(select(PlatformTask).where(PlatformTask.id == task_id))
    task = result.scalar_one_or_none()
    if task is None:
        raise HTTPException(status_code=404, detail=f"PlatformTask {task_id!r} not found")
    return _serialize(task)


# ---------------------------------------------------------------------------
# Retry
# ---------------------------------------------------------------------------

@router.post("/{task_id}/retry", summary="Retry a failed task")
async def retry_platform_task(
    task_id: str,
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    try:
        task = await retry_task(db, task_id)
    except ValueError as exc:
        detail = str(exc)
        status_code = 404 if "not found" in detail else 400
        raise HTTPException(status_code=status_code, detail=detail) from exc
    await db.commit()
    await dispatch_platform_task(task.id, task.kind, task.payload_ref, task.priority)
    return _serialize(task)


# ---------------------------------------------------------------------------
# Cancel
# ---------------------------------------------------------------------------

@router.post("/{task_id}/cancel", summary="Cancel a pending/queued task")
async def cancel_platform_task(
    task_id: str,
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    try:
        task = await cancel_task(db, task_id)
    except ValueError as exc:
        detail = str(exc)
        status_code = 404 if "not found" in detail else 400
        raise HTTPException(status_code=status_code, detail=detail) from exc
    return _serialize(task)


# ---------------------------------------------------------------------------
# Delete
# ---------------------------------------------------------------------------

@router.delete("/{task_id}", summary="Delete a finished task record")
async def delete_platform_task(
    task_id: str,
    db: AsyncSession = Depends(get_db),
) -> dict[str, str]:
    result = await db.execute(select(PlatformTask).where(PlatformTask.id == task_id))
    task = result.scalar_one_or_none()
    if task is None:
        raise HTTPException(status_code=404, detail=f"PlatformTask {task_id!r} not found")
    if task.status in ("QUEUED", "RUNNING"):
        raise HTTPException(status_code=400, detail="Cannot delete an active task; cancel it first")
    await db.delete(task)
    return {"message": "deleted"}

"""
V3 Experiment Service — manages PlatformExperiment, ExperimentRun lifecycle.

Responsibilities:
- CRUD for experiments and runs
- Leaderboard generation (rank runs by objective metric)
- AutoML run dispatch (submit N runs via task_runner)
- best_run_id update after all runs complete
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any

from fastapi import HTTPException
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.database import (
    Dataset,
    DatasetVersion,
    ExperimentRun,
    PlatformExperiment,
    PlatformTask,
)

logger = logging.getLogger(__name__)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# Serializers
# ---------------------------------------------------------------------------

def _serialize_experiment(exp: PlatformExperiment) -> dict[str, Any]:
    return {
        "id": exp.id,
        "name": exp.name,
        "description": exp.description,
        "modeling_task_id": getattr(exp, "modeling_task_id", None),
        "dataset_id": exp.dataset_id,
        "dataset_name": exp.dataset_name,
        "dataset_version_id": exp.dataset_version_id,
        "objective_metric": exp.objective_metric,
        "objective_direction": exp.objective_direction,
        "status": exp.status,
        "kind": exp.kind,
        "strategy_type": getattr(exp, "strategy_type", None),
        "best_run_id": exp.best_run_id,
        "config": exp.config or {},
        "created_at": exp.created_at.isoformat() if exp.created_at else None,
        "updated_at": exp.updated_at.isoformat() if exp.updated_at else None,
        "finished_at": exp.finished_at.isoformat() if exp.finished_at else None,
    }


def _serialize_run(run: ExperimentRun) -> dict[str, Any]:
    return {
        "id": run.id,
        "experiment_id": run.experiment_id,
        "task_id": run.task_id,
        "parent_run_id": run.parent_run_id,
        "params": run.params or {},
        "metrics": run.metrics or {},
        "status": run.status,
        "rank": run.rank,
        "artifacts_uri": run.artifacts_uri,
        "notes": run.notes,
        "created_at": run.created_at.isoformat() if run.created_at else None,
        "started_at": run.started_at.isoformat() if run.started_at else None,
        "finished_at": run.finished_at.isoformat() if run.finished_at else None,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def _get_experiment_or_404(db: AsyncSession, experiment_id: str) -> PlatformExperiment:
    result = await db.execute(
        select(PlatformExperiment).where(PlatformExperiment.id == experiment_id)
    )
    exp = result.scalar_one_or_none()
    if exp is None:
        raise HTTPException(status_code=404, detail=f"Experiment {experiment_id!r} not found")
    return exp


async def _get_run_or_404(db: AsyncSession, run_id: str) -> ExperimentRun:
    result = await db.execute(
        select(ExperimentRun).where(ExperimentRun.id == run_id)
    )
    run = result.scalar_one_or_none()
    if run is None:
        raise HTTPException(status_code=404, detail=f"ExperimentRun {run_id!r} not found")
    return run


def _pick_metric(metrics: dict, metric_key: str) -> float | None:
    """Extract a numeric metric value from a metrics dict."""
    v = metrics.get(metric_key)
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Experiment CRUD
# ---------------------------------------------------------------------------

async def list_experiments(
    db: AsyncSession,
    page: int = 1,
    page_size: int = 20,
    status: str | None = None,
) -> dict[str, Any]:
    stmt = select(PlatformExperiment)
    count_stmt = select(func.count(PlatformExperiment.id))

    if status:
        stmt = stmt.where(PlatformExperiment.status == status.upper())
        count_stmt = count_stmt.where(PlatformExperiment.status == status.upper())

    total = (await db.execute(count_stmt)).scalar_one()
    rows = await db.execute(
        stmt.order_by(PlatformExperiment.created_at.desc())
        .offset((page - 1) * page_size)
        .limit(page_size)
    )
    experiments = rows.scalars().all()
    return {
        "items": [_serialize_experiment(e) for e in experiments],
        "total": total,
        "page": page,
        "page_size": page_size,
    }


async def create_experiment(
    db: AsyncSession,
    name: str,
    dataset_id: str | None = None,
    description: str | None = None,
    objective_metric: str = "accuracy",
    objective_direction: str = "max",
    kind: str = "single",
    config: dict | None = None,
) -> dict[str, Any]:
    # Resolve dataset name if dataset_id provided
    dataset_name = None
    if dataset_id:
        ds_result = await db.execute(select(Dataset).where(Dataset.id == dataset_id))
        ds = ds_result.scalar_one_or_none()
        if ds is None:
            raise HTTPException(status_code=404, detail=f"Dataset {dataset_id!r} not found")
        dataset_name = ds.name

    exp = PlatformExperiment(
        name=name,
        description=description,
        dataset_id=dataset_id,
        dataset_name=dataset_name,
        objective_metric=objective_metric,
        objective_direction=objective_direction,
        kind=kind,
        config=config,
        status="CREATED",
    )
    db.add(exp)
    await db.flush()
    await db.refresh(exp)
    return _serialize_experiment(exp)


async def get_experiment(db: AsyncSession, experiment_id: str) -> dict[str, Any]:
    exp = await _get_experiment_or_404(db, experiment_id)
    return _serialize_experiment(exp)


async def delete_experiment(db: AsyncSession, experiment_id: str) -> None:
    exp = await _get_experiment_or_404(db, experiment_id)
    if exp.status == "RUNNING":
        raise HTTPException(status_code=400, detail="Cannot delete a running experiment")
    await db.delete(exp)
    await db.flush()


# ---------------------------------------------------------------------------
# Run CRUD
# ---------------------------------------------------------------------------

async def list_runs(
    db: AsyncSession,
    experiment_id: str,
    page: int = 1,
    page_size: int = 50,
) -> dict[str, Any]:
    await _get_experiment_or_404(db, experiment_id)

    count_stmt = select(func.count(ExperimentRun.id)).where(
        ExperimentRun.experiment_id == experiment_id
    )
    total = (await db.execute(count_stmt)).scalar_one()

    rows = await db.execute(
        select(ExperimentRun)
        .where(ExperimentRun.experiment_id == experiment_id)
        .order_by(ExperimentRun.created_at.desc())
        .offset((page - 1) * page_size)
        .limit(page_size)
    )
    runs = rows.scalars().all()
    return {
        "items": [_serialize_run(r) for r in runs],
        "total": total,
        "page": page,
        "page_size": page_size,
    }


async def create_run(
    db: AsyncSession,
    experiment_id: str,
    params: dict | None = None,
    parent_run_id: str | None = None,
    notes: str | None = None,
) -> dict[str, Any]:
    await _get_experiment_or_404(db, experiment_id)

    run = ExperimentRun(
        experiment_id=experiment_id,
        params=params,
        parent_run_id=parent_run_id,
        notes=notes,
        status="PENDING",
    )
    db.add(run)
    await db.flush()
    await db.refresh(run)
    return _serialize_run(run)


async def get_run(db: AsyncSession, run_id: str) -> dict[str, Any]:
    run = await _get_run_or_404(db, run_id)
    return _serialize_run(run)


async def update_run_metrics(
    db: AsyncSession,
    run_id: str,
    metrics: dict,
    status: str = "SUCCESS",
    artifacts_uri: str | None = None,
) -> dict[str, Any]:
    """Called by the training worker to write back metrics after completion."""
    run = await _get_run_or_404(db, run_id)
    run.metrics = metrics
    run.status = status
    run.artifacts_uri = artifacts_uri
    if status in ("SUCCESS", "FAILED"):
        run.finished_at = _utcnow()
    await db.flush()

    # Refresh leaderboard for this experiment
    await _refresh_leaderboard(db, run.experiment_id)
    return _serialize_run(run)


# ---------------------------------------------------------------------------
# Leaderboard
# ---------------------------------------------------------------------------

async def get_leaderboard(
    db: AsyncSession,
    experiment_id: str,
) -> list[dict[str, Any]]:
    """Return runs sorted by objective metric (best first)."""
    exp = await _get_experiment_or_404(db, experiment_id)

    rows = await db.execute(
        select(ExperimentRun)
        .where(
            ExperimentRun.experiment_id == experiment_id,
            ExperimentRun.status == "SUCCESS",
        )
        .order_by(ExperimentRun.rank.asc())
    )
    runs = rows.scalars().all()
    return [_serialize_run(r) for r in runs]


async def _refresh_leaderboard(db: AsyncSession, experiment_id: str) -> None:
    """Sort all successful runs and assign rank + update best_run_id on experiment."""
    exp_result = await db.execute(
        select(PlatformExperiment).where(PlatformExperiment.id == experiment_id)
    )
    exp = exp_result.scalar_one_or_none()
    if exp is None:
        return

    rows = await db.execute(
        select(ExperimentRun).where(
            ExperimentRun.experiment_id == experiment_id,
            ExperimentRun.status == "SUCCESS",
        )
    )
    runs = rows.scalars().all()

    if not runs:
        return

    direction = exp.objective_direction or "max"
    metric_key = exp.objective_metric or "accuracy"
    reverse = direction == "max"

    scored = []
    for run in runs:
        v = _pick_metric(run.metrics or {}, metric_key)
        if v is not None:
            scored.append((run, v))

    scored.sort(key=lambda x: x[1], reverse=reverse)

    for rank, (run, _) in enumerate(scored, start=1):
        run.rank = rank

    if scored:
        exp.best_run_id = scored[0][0].id

    await db.flush()


# ---------------------------------------------------------------------------
# AutoML: submit multiple runs in parallel via Celery
# ---------------------------------------------------------------------------

async def submit_automl_experiment(
    db: AsyncSession,
    experiment_id: str,
    candidates: list[dict],
) -> list[dict[str, Any]]:
    """
    Submit N AutoML candidate runs.

    `candidates` is a list of dicts like:
      {"model_type": "RandomForestClassifier", "hyperparameters": {...}, "dataset_id": "...", ...}

    Each candidate creates:
      1. An ExperimentRun
      2. A domain TrainingTask (training_tasks table)
      3. A PlatformTask (platform_tasks table) dispatched to Celery
    """
    from app.scheduler.task_runner import submit_task
    from app.services.training_service import create_training_task_record

    exp = await _get_experiment_or_404(db, experiment_id)
    if exp.status not in ("CREATED", "FAILED"):
        raise HTTPException(status_code=400, detail="Experiment is already running or completed")

    exp.status = "RUNNING"
    await db.flush()

    submitted_runs = []
    asyncio_tasks = []

    for candidate in candidates:
        # 1. Create domain training task record
        domain_task = await create_training_task_record(db, candidate)

        # 2. Create ExperimentRun
        run = ExperimentRun(
            experiment_id=experiment_id,
            params=candidate,
            status="PENDING",
        )
        db.add(run)
        await db.flush()
        await db.refresh(run)

        # 3. Register PlatformTask (asyncio mode — no Celery dispatch)
        from app.scheduler.task_runner import register_domain_task
        ptask = await register_domain_task(
            db=db,
            kind="train",
            payload_ref=f"train:{domain_task.id}",
        )

        # Link run → task
        run.task_id = ptask.id
        await db.flush()

        submitted_runs.append(_serialize_run(run))
        asyncio_tasks.append((domain_task.id, ptask.id, run.id))

    await db.commit()

    # 4. Fire asyncio training coroutines concurrently (Celery substitute for dev mode)
    from app.services.automl_service import _run_automl_candidate
    for domain_task_id, platform_task_id, run_id in asyncio_tasks:
        asyncio.create_task(
            _run_automl_candidate(
                domain_task_id=domain_task_id,
                platform_task_id=platform_task_id,
                run_id=run_id,
                experiment_id=experiment_id,
            )
        )

    return submitted_runs

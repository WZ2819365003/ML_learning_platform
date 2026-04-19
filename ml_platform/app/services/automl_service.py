"""
AutoML Service — batch-submit training candidates from the registry YAML.

Flow (asyncio / dev mode, no Celery worker required):
  1. Load candidate list from registry/automl_candidates.yaml
  2. For each candidate, create:
       - TrainingTask (domain record)
       - ExperimentRun (for leaderboard)
       - PlatformTask (unified visibility)
  3. Fire asyncio tasks concurrently — each runs _run_training_sync_by_id
     and on completion writes back to ExperimentRun + refreshes leaderboard.

When Celery workers are available (Stage 5), replace the asyncio launch
with submit_task() and drop the fire_and_forget_automl_run calls.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from fastapi import HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.database import ExperimentRun, PlatformExperiment, async_session_factory
from app.services.training_service import create_training_task_record
from app.scheduler.task_runner import register_domain_task, update_platform_task_status

logger = logging.getLogger(__name__)

_REGISTRY_PATH = Path(__file__).parent.parent.parent / "registry" / "automl_candidates.yaml"

# Models whose training direction can be inferred from the objective metric
_REGRESSION_METRICS = {"rmse", "mae", "mse", "mape", "r2"}


def load_candidates(task_type: str = "classification") -> list[dict]:
    """Load and return AutoML candidates for the given task type."""
    from app.core.trainer import detect_task_type, list_available_models

    if not _REGISTRY_PATH.exists():
        raise FileNotFoundError(f"AutoML registry not found: {_REGISTRY_PATH}")
    with open(_REGISTRY_PATH) as f:
        data = yaml.safe_load(f)
    candidates = data.get(task_type, [])
    valid_models = set(list_available_models())
    filtered_candidates: list[dict[str, Any]] = []
    for candidate in candidates:
        model_type = candidate.get("model_type")
        if model_type not in valid_models:
            logger.warning("Skipping AutoML candidate %r: model is not registered", model_type)
            continue
        if detect_task_type(model_type) != task_type:
            logger.warning(
                "Skipping AutoML candidate %r: expected %s model, got %s",
                model_type, task_type, detect_task_type(model_type),
            )
            continue
        filtered_candidates.append(candidate)
    if not filtered_candidates:
        raise ValueError(f"No candidates defined for task_type={task_type!r}")
    return filtered_candidates


def warm_start_capable() -> list[str]:
    """Return model types that support incremental training via warm_start."""
    if not _REGISTRY_PATH.exists():
        return []
    with open(_REGISTRY_PATH) as f:
        data = yaml.safe_load(f)
    return data.get("warm_start_capable", [])


async def submit_automl_from_registry(
    db: AsyncSession,
    experiment_id: str,
    dataset_id: str,
    target_column: str,
    task_type: str = "classification",
    eval_metrics: list[str] | None = None,
    test_size: float = 0.2,
    max_candidates: int | None = None,
) -> dict[str, Any]:
    """
    Load candidates from registry YAML and submit them as ExperimentRuns.

    Returns a summary: {"submitted": N, "experiment_id": ..., "run_ids": [...]}
    """
    from sqlalchemy import select

    # Validate experiment exists and is in a submittable state
    result = await db.execute(
        select(PlatformExperiment).where(PlatformExperiment.id == experiment_id)
    )
    exp = result.scalar_one_or_none()
    if exp is None:
        raise HTTPException(status_code=404, detail=f"Experiment {experiment_id!r} not found")
    if exp.status not in ("CREATED", "FAILED"):
        raise HTTPException(
            status_code=400,
            detail=f"Experiment is {exp.status!r} — can only submit from CREATED or FAILED state",
        )

    # Infer task_type from objective_metric if not explicitly given
    if exp.objective_metric in _REGRESSION_METRICS:
        task_type = "regression"

    candidates = load_candidates(task_type)
    if max_candidates:
        candidates = candidates[:max_candidates]

    if eval_metrics is None:
        if task_type == "regression":
            eval_metrics = ["rmse", "mae", "r2"]
        else:
            eval_metrics = ["accuracy", "f1", "roc_auc"]

    # Mark experiment RUNNING
    exp.status = "RUNNING"
    await db.flush()

    submitted_run_ids = []
    asyncio_tasks = []

    for candidate in candidates:
        model_type = candidate["model_type"]
        hyperparameters = candidate.get("hyperparameters", {})

        # 1. Domain task
        try:
            domain_task = await create_training_task_record(db, {
                "dataset_id": dataset_id,
                "model_type": model_type,
                "target_column": target_column,
                "hyperparameters": hyperparameters,
                "test_size": test_size,
                "eval_metrics": eval_metrics,
            })
        except HTTPException as e:
            logger.warning("Skipping candidate %r: %s", model_type, e.detail)
            continue

        # 2. ExperimentRun
        run = ExperimentRun(
            experiment_id=experiment_id,
            params={
                "model_type": model_type,
                "hyperparameters": hyperparameters,
                "description": candidate.get("description", model_type),
                "dataset_id": dataset_id,
                "target_column": target_column,
                "task_type": task_type,
                "eval_metrics": eval_metrics,
                "test_size": test_size,
            },
            status="PENDING",
        )
        db.add(run)
        await db.flush()
        await db.refresh(run)

        # 3. PlatformTask
        platform_task = await register_domain_task(
            db,
            kind="train",
            payload_ref=f"train:{domain_task.id}",
        )
        run.task_id = platform_task.id
        await db.flush()

        submitted_run_ids.append(run.id)
        asyncio_tasks.append((domain_task.id, platform_task.id, run.id))

    await db.commit()

    # 4. Fire asyncio training tasks concurrently (non-blocking, dev-mode Celery substitute)
    for domain_task_id, platform_task_id, run_id in asyncio_tasks:
        asyncio.create_task(
            _run_automl_candidate(
                domain_task_id=domain_task_id,
                platform_task_id=platform_task_id,
                run_id=run_id,
                experiment_id=experiment_id,
            )
        )

    logger.info(
        "AutoML experiment %s: submitted %d candidates for %s",
        experiment_id, len(asyncio_tasks), task_type,
    )
    return {
        "submitted": len(asyncio_tasks),
        "experiment_id": experiment_id,
        "task_type": task_type,
        "run_ids": submitted_run_ids,
    }


async def _run_automl_candidate(
    domain_task_id: str,
    platform_task_id: str,
    run_id: str,
    experiment_id: str,
) -> None:
    """
    Background coroutine for a single AutoML candidate.

    Runs training via _run_training_sync_by_id, then writes back to
    ExperimentRun and refreshes the leaderboard.
    """
    from app.services.training_service import _run_training_sync_by_id
    from app.services.experiment_service import update_run_metrics, _refresh_leaderboard
    from sqlalchemy import select

    await update_platform_task_status(platform_task_id, "RUNNING")

    try:
        async with async_session_factory() as db:
            run_result = await db.execute(
                select(ExperimentRun).where(ExperimentRun.id == run_id)
            )
            run = run_result.scalar_one_or_none()
            if run and run.started_at is None:
                run.started_at = datetime.now(timezone.utc)
                run.status = "RUNNING"
                await db.commit()
    except Exception as exc:
        logger.warning("Could not mark AutoML run %s as RUNNING: %s", run_id, exc)

    try:
        result = await _run_training_sync_by_id(domain_task_id, platform_task_id)
        metrics = result.get("metrics", {})

        # Write metrics back to ExperimentRun and refresh leaderboard
        async with async_session_factory() as db:
            await update_run_metrics(db, run_id, metrics, status="SUCCESS")
            await db.commit()

        await update_platform_task_status(platform_task_id, "SUCCESS", metrics=metrics)
        logger.info("AutoML run %s (platform_task=%s) completed: %s", run_id, platform_task_id, metrics)

    except Exception as exc:
        logger.error("AutoML run %s failed: %s", run_id, exc, exc_info=True)
        try:
            async with async_session_factory() as db:
                await update_run_metrics(db, run_id, {}, status="FAILED")
                await db.commit()
        except Exception:
            pass
        await update_platform_task_status(platform_task_id, "FAILED", error=str(exc))

    # Check if all runs are done and update experiment status
    await _maybe_complete_experiment(experiment_id)


async def _maybe_complete_experiment(experiment_id: str) -> None:
    """Mark experiment COMPLETED when all its runs have finished."""
    from sqlalchemy import select, func
    try:
        async with async_session_factory() as db:
            total_q = await db.execute(
                select(func.count(ExperimentRun.id))
                .where(ExperimentRun.experiment_id == experiment_id)
            )
            done_q = await db.execute(
                select(func.count(ExperimentRun.id))
                .where(
                    ExperimentRun.experiment_id == experiment_id,
                    ExperimentRun.status.in_(["SUCCESS", "FAILED"]),
                )
            )
            total = total_q.scalar_one()
            done  = done_q.scalar_one()

            if total > 0 and done >= total:
                result = await db.execute(
                    select(PlatformExperiment).where(PlatformExperiment.id == experiment_id)
                )
                exp = result.scalar_one_or_none()
                if exp and exp.status == "RUNNING":
                    exp.status = "COMPLETED"
                    await db.commit()
                    logger.info("Experiment %s marked COMPLETED (%d/%d runs done)", experiment_id, done, total)
    except Exception as exc:
        logger.warning("_maybe_complete_experiment(%s) failed: %s", experiment_id, exc)

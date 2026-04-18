"""
V3 Run Inspector API — /api/platform/runs/{run_id}/inspector

Aggregates everything the frontend needs to render a single ExperimentRun
detail drawer in one round-trip:

  - the run itself (params, metrics, status, rank, search_meta)
  - the linked PlatformTask (progress, worker, retry count, error)
  - the underlying TrainingTask + dataset summary
  - step metrics / logs (latest N entries)
  - sibling runs in the same experiment (for prev/next navigation)
  - SHAP importances if already computed

One aggregated endpoint beats 5 round-trips on a drawer open, and keeps the
payload inspectable/audit-friendly on the backend side.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.database import (
    Dataset,
    ExperimentRun,
    PlatformExperiment,
    PlatformTask,
    TrainingLog,
    TrainingTask,
    get_db,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/platform/runs", tags=["V3 Run Inspector"])


# ---------------------------------------------------------------------------
# Serializers (kept local to avoid circular imports with other routes)
# ---------------------------------------------------------------------------

def _iso(dt) -> str | None:
    return dt.isoformat() if dt else None


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
        "trial_no": run.trial_no,
        "search_meta": run.search_meta or {},
        "source_experiment_type": run.source_experiment_type,
        "artifacts_uri": run.artifacts_uri,
        "notes": run.notes,
        "created_at": _iso(run.created_at),
        "started_at": _iso(run.started_at),
        "finished_at": _iso(run.finished_at),
    }


def _serialize_platform_task(task: PlatformTask) -> dict[str, Any]:
    return {
        "id": task.id,
        "kind": task.kind,
        "status": task.status,
        "priority": task.priority,
        "worker_id": task.worker_id,
        "retry_count": task.retry_count,
        "max_retries": task.max_retries,
        "payload_ref": task.payload_ref,
        "progress": task.progress,
        "metrics_snapshot": task.metrics_snapshot,
        "error_message": task.error_message,
        "queued_at": _iso(task.queued_at),
        "started_at": _iso(task.started_at),
        "finished_at": _iso(task.finished_at),
    }


def _serialize_training_task(tt: TrainingTask, dataset: Dataset | None) -> dict[str, Any]:
    return {
        "id": tt.id,
        "name": tt.name,
        "model_type": tt.model_type,
        "hyperparameters": tt.hyperparameters or {},
        "target_column": tt.target_column,
        "test_size": tt.test_size,
        "eval_metrics": tt.eval_metrics or [],
        "status": tt.status,
        "progress": tt.progress,
        "model_path": tt.model_path,
        "dataset": {
            "id": dataset.id if dataset else tt.dataset_id,
            "name": dataset.name if dataset else None,
            "row_count": dataset.row_count if dataset else None,
            "column_count": dataset.column_count if dataset else None,
        } if (dataset or tt.dataset_id) else None,
    }


def _serialize_log(log: TrainingLog) -> dict[str, Any]:
    return {
        "level": log.level,
        "message": log.message,
        "extra": log.extra or {},
        "created_at": _iso(log.created_at),
    }


# ---------------------------------------------------------------------------
# Inspector endpoint
# ---------------------------------------------------------------------------

@router.get("/{run_id}/inspector", summary="Aggregated run detail for the Run Inspector drawer")
async def inspect_run(
    run_id: str,
    log_limit: int = Query(100, ge=1, le=500),
    include_siblings: bool = Query(True),
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    # --- 1. Load run
    run = (
        await db.execute(select(ExperimentRun).where(ExperimentRun.id == run_id))
    ).scalar_one_or_none()
    if run is None:
        raise HTTPException(status_code=404, detail=f"Run {run_id!r} not found")

    # --- 2. Experiment context (for direction / metric / strategy)
    exp = (
        await db.execute(
            select(PlatformExperiment).where(PlatformExperiment.id == run.experiment_id)
        )
    ).scalar_one_or_none()

    # --- 3. Platform task (unified scheduler view)
    platform_task: PlatformTask | None = None
    domain_task_id: str | None = None
    if run.task_id:
        platform_task = (
            await db.execute(select(PlatformTask).where(PlatformTask.id == run.task_id))
        ).scalar_one_or_none()
        if platform_task and platform_task.payload_ref and ":" in platform_task.payload_ref:
            _, _, domain_task_id = platform_task.payload_ref.partition(":")

    # --- 4. Domain training task + dataset
    training_task_payload: dict[str, Any] | None = None
    logs_payload: list[dict[str, Any]] = []
    if domain_task_id:
        tt = (
            await db.execute(select(TrainingTask).where(TrainingTask.id == domain_task_id))
        ).scalar_one_or_none()
        if tt:
            ds = (
                await db.execute(select(Dataset).where(Dataset.id == tt.dataset_id))
            ).scalar_one_or_none()
            training_task_payload = _serialize_training_task(tt, ds)

            log_rows = await db.execute(
                select(TrainingLog)
                .where(TrainingLog.task_id == tt.id)
                .order_by(TrainingLog.created_at.desc())
                .limit(log_limit)
            )
            logs = list(log_rows.scalars().all())
            logs.reverse()  # UI wants oldest-first
            logs_payload = [_serialize_log(lg) for lg in logs]

    # --- 5. Sibling runs (prev/next + rank context)
    siblings: list[dict[str, Any]] = []
    if include_siblings and exp is not None:
        sib_rows = await db.execute(
            select(ExperimentRun)
            .where(ExperimentRun.experiment_id == exp.id)
            .order_by(
                ExperimentRun.rank.is_(None),
                ExperimentRun.rank.asc(),
                ExperimentRun.created_at.asc(),
            )
        )
        siblings = [
            {
                "id": r.id,
                "trial_no": r.trial_no,
                "rank": r.rank,
                "status": r.status,
                "metrics": r.metrics or {},
                "source_experiment_type": r.source_experiment_type,
            }
            for r in sib_rows.scalars().all()
        ]

    # --- 6. SHAP importances if already computed (inline)
    shap_importances = (run.metrics or {}).get("shap_importances")
    shap_summary: dict[str, Any] | None = None
    if shap_importances:
        # Top-10 by default — front-end can ask for more via the full endpoint.
        top_items = list(shap_importances.items())[:10]
        shap_summary = {
            "has_explanation": True,
            "feature_count": len(shap_importances),
            "top_features": [
                {"feature": k, "importance": float(v)} for k, v in top_items
            ],
            "method": (run.metrics or {}).get("shap_method", "shap"),
        }
    else:
        shap_summary = {"has_explanation": False}

    return {
        "run": _serialize_run(run),
        "experiment": {
            "id": exp.id,
            "name": exp.name,
            "strategy_type": exp.strategy_type,
            "objective_metric": exp.objective_metric,
            "objective_direction": exp.objective_direction,
            "modeling_task_id": exp.modeling_task_id,
            "status": exp.status,
        } if exp else None,
        "platform_task": _serialize_platform_task(platform_task) if platform_task else None,
        "training_task": training_task_payload,
        "logs": logs_payload,
        "siblings": siblings,
        "shap": shap_summary,
    }

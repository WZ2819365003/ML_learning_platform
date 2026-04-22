"""
V3 Phase 3 — orphan PlatformTask detail resolver.

``GET /api/platform/tasks/{id}/detail`` delegates here.  Given a PlatformTask
id, we:

  * Load the task row and serialise the base fields (status, progress,
    timing, payload_ref, etc.).
  * Parse ``payload_ref`` into ``(kind, domain_id)`` and resolve the
    domain-side entity so the frontend can render a useful "source data"
    panel even when the task is not wired into a ModelingTask tree.
  * Tail the per-task log file at ``storage/logs/{domain_id}.log`` so the
    drawer can show stack traces without a second API round-trip.

Resolvers are intentionally defensive — if the domain row has been purged
or the payload_ref uses an unknown prefix we still return a usable
skeleton rather than 500.  The drawer can render "源数据已不可用" based on
``domain is None``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from fastapi import HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.models.database import (
    DLTrainingTask,
    ExperimentRun,
    PlatformTask,
    TimeSeriesForecastTask,
    TrainingTask,
)

logger = logging.getLogger(__name__)


# Human-readable labels for the drawer header.  When a new kind is added
# downstream, add it here and the drawer picks it up automatically.
SOURCE_LABELS: dict[str, str] = {
    "train":       "独立训练",
    "dl_train":    "独立 DL 训练",
    "explain":     "独立 SHAP 解释",
    "ts_forecast": "时序预测",
    "eval":        "独立评估",
    "predict":     "独立预测",
    "preprocess":  "数据预处理",
}


def _iso(dt) -> str | None:
    return dt.isoformat() if dt else None


def _serialise_task(task: PlatformTask) -> dict[str, Any]:
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
        "queued_at":   _iso(task.queued_at),
        "started_at":  _iso(task.started_at),
        "finished_at": _iso(task.finished_at),
        "created_at":  _iso(task.created_at),
        "updated_at":  _iso(task.updated_at),
        "depends_on":  task.depends_on or [],
    }


def _split_payload_ref(payload_ref: str | None) -> tuple[str | None, str | None]:
    if not payload_ref:
        return None, None
    kind, sep, domain_id = payload_ref.partition(":")
    if not sep:
        return None, None
    return (kind or None), (domain_id or None)


# ---------------------------------------------------------------------------
# Per-kind domain resolvers — each returns a plain dict or None.
# ---------------------------------------------------------------------------

async def _resolve_train(db: AsyncSession, domain_id: str) -> dict[str, Any] | None:
    row = await db.execute(select(TrainingTask).where(TrainingTask.id == domain_id))
    tt = row.scalar_one_or_none()
    if tt is None:
        return None
    return {
        "id": tt.id,
        "name": tt.name,
        "model_type": tt.model_type,
        "dataset_id": tt.dataset_id,
        "target_column": tt.target_column,
        "task_type": None,
        "status": tt.status,
        "progress": tt.progress,
        "result_metrics": tt.result_metrics or {},
        "model_path": tt.model_path,
        "error_message": tt.error_message,
        "started_at":  _iso(tt.started_at),
        "finished_at": _iso(tt.finished_at),
        "created_at":  _iso(tt.created_at),
    }


async def _resolve_dl_train(db: AsyncSession, domain_id: str) -> dict[str, Any] | None:
    row = await db.execute(select(DLTrainingTask).where(DLTrainingTask.id == domain_id))
    tt = row.scalar_one_or_none()
    if tt is None:
        return None
    return {
        "id": tt.id,
        "name": tt.name,
        "model_type": tt.model_type,
        "dataset_id": tt.dataset_id,
        "target_column": tt.target_column,
        "task_type": tt.task_type,
        "status": tt.status,
        "progress": tt.progress,
        "current_epoch": tt.current_epoch,
        "total_epochs":  tt.total_epochs,
        "arch_config":  tt.arch_config or {},
        "opt_config":   tt.opt_config or {},
        "train_config": tt.train_config or {},
        "result_metrics": tt.result_metrics or {},
        "model_path": tt.model_path,
        "error_message": tt.error_message,
        "started_at":  _iso(tt.started_at),
        "finished_at": _iso(tt.finished_at),
        "created_at":  _iso(tt.created_at),
    }


async def _resolve_explain(db: AsyncSession, domain_id: str) -> dict[str, Any] | None:
    # explain's domain is the ExperimentRun being explained — the training
    # task behind it is linked through run.task_id → platform_task.payload_ref.
    row = await db.execute(select(ExperimentRun).where(ExperimentRun.id == domain_id))
    run = row.scalar_one_or_none()
    if run is None:
        return None

    params = run.params or {}
    linked_training_task_id: str | None = None
    if run.task_id:
        ptask_row = await db.execute(
            select(PlatformTask).where(PlatformTask.id == run.task_id)
        )
        ptask = ptask_row.scalar_one_or_none()
        if ptask and ptask.payload_ref:
            _, _, linked_training_task_id = ptask.payload_ref.partition(":")

    return {
        "run_id": run.id,
        "experiment_id": run.experiment_id,
        "trial_no": run.trial_no,
        "status": run.status,
        "rank": run.rank,
        "metrics": run.metrics or {},
        "model_type": params.get("model_type"),
        "dataset_id": params.get("dataset_id"),
        "target_column": params.get("target_column"),
        "linked_training_task_id": linked_training_task_id,
        "artifacts_uri": run.artifacts_uri,
        "source_experiment_type": run.source_experiment_type,
        "started_at":  _iso(run.started_at),
        "finished_at": _iso(run.finished_at),
        "created_at":  _iso(run.created_at),
    }


async def _resolve_ts_forecast(db: AsyncSession, domain_id: str) -> dict[str, Any] | None:
    row = await db.execute(
        select(TimeSeriesForecastTask).where(TimeSeriesForecastTask.id == domain_id)
    )
    tt = row.scalar_one_or_none()
    if tt is None:
        return None
    # Trim the `result` payload — forecasts can be large arrays and we only
    # need a preview in the drawer.
    result = tt.result or {}
    preview: dict[str, Any] = {}
    if isinstance(result, dict):
        for key in ("summary", "metrics", "horizon", "median"):
            if key in result:
                preview[key] = result[key]
        for key in ("low", "high", "point"):
            values = result.get(key)
            if isinstance(values, list) and values:
                preview[f"{key}_head"] = values[:10]
                preview[f"{key}_count"] = len(values)
    return {
        "id": tt.id,
        "dataset_id": tt.dataset_id,
        "dataset_name": tt.dataset_name,
        "value_column": tt.value_column,
        "time_column": tt.time_column,
        "horizon": tt.horizon,
        "frequency": tt.frequency,
        "model_name": tt.model_name,
        "status": tt.status,
        "error_message": tt.error_message,
        "result_preview": preview,
        "started_at":  _iso(tt.started_at),
        "finished_at": _iso(tt.finished_at),
        "created_at":  _iso(tt.created_at),
    }


_RESOLVERS = {
    "train":       _resolve_train,
    "dl_train":    _resolve_dl_train,
    "explain":     _resolve_explain,
    "ts_forecast": _resolve_ts_forecast,
}


# ---------------------------------------------------------------------------
# Log tail — read the per-task log file written by TrainingLogger.
# ---------------------------------------------------------------------------

def _tail_log_file(domain_id: str | None, limit: int = 200) -> list[str]:
    """Return the last ``limit`` lines of the per-task log file, if any."""
    if not domain_id:
        return []
    settings = get_settings()
    log_path: Path = settings.storage_logs / f"{domain_id}.log"
    if not log_path.exists():
        return []
    try:
        # Full-file read is fine — these files top out at a few MB even for
        # long DL runs, and slicing keeps the response payload small.
        with log_path.open("r", encoding="utf-8", errors="replace") as fp:
            lines = fp.readlines()
    except OSError as exc:
        logger.warning("Failed to tail log file %s: %s", log_path, exc)
        return []
    tail = lines[-limit:]
    return [ln.rstrip("\n") for ln in tail]


# ---------------------------------------------------------------------------
# Public entrypoint
# ---------------------------------------------------------------------------

async def get_platform_task_detail(
    db: AsyncSession,
    task_id: str,
    log_limit: int = 200,
) -> dict[str, Any]:
    """Return enriched detail for a PlatformTask.  Raises 404 if missing."""
    row = await db.execute(select(PlatformTask).where(PlatformTask.id == task_id))
    task = row.scalar_one_or_none()
    if task is None:
        raise HTTPException(
            status_code=404, detail=f"PlatformTask {task_id!r} not found"
        )

    payload_kind, domain_id = _split_payload_ref(task.payload_ref)
    source_label = SOURCE_LABELS.get(payload_kind or "", payload_kind or "未知来源")

    domain: dict[str, Any] | None = None
    if payload_kind and domain_id:
        resolver = _RESOLVERS.get(payload_kind)
        if resolver is not None:
            try:
                domain = await resolver(db, domain_id)
            except Exception as exc:  # defensive — never let a resolver 500 the drawer
                logger.exception(
                    "Detail resolver for kind=%s id=%s failed: %s",
                    payload_kind, domain_id, exc,
                )
                domain = None

    return {
        "task": _serialise_task(task),
        "source_label": source_label,
        "domain_kind": payload_kind,
        "domain_id": domain_id,
        "domain": domain,
        "recent_logs": _tail_log_file(domain_id, limit=log_limit),
    }

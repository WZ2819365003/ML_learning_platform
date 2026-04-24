"""
V3 Modeling Task Service — top-level workbench entity.

A ``ModelingTask`` wraps a modeling goal (dataset + target + task_type) and
owns one or more ``PlatformExperiment``s — each being a distinct *strategy*
(baseline, grid_search, bayesian_search).

This service provides CRUD + aggregated views (experiments, leaderboard
across all experiments, best run).  Tuning-engine dispatch lives in
``app/services/tuning_service.py`` (Commit C) and is called by routes.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from fastapi import HTTPException
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.database import (
    Dataset,
    DatasetVersion,
    ExperimentRun,
    ModelingTask,
    PlatformTask,
    PlatformExperiment,
)
from app.services.prediction_service import load_dataframe

logger = logging.getLogger(__name__)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# Serializers
# ---------------------------------------------------------------------------

def serialize_modeling_task(task: ModelingTask) -> dict[str, Any]:
    return {
        "id": task.id,
        "name": task.name,
        "description": task.description,
        "dataset_id": task.dataset_id,
        "dataset_name": task.dataset_name,
        "dataset_version_id": task.dataset_version_id,
        "target_column": task.target_column,
        "task_type": task.task_type,
        "objective_metric": task.objective_metric,
        "objective_direction": task.objective_direction,
        "status": task.status,
        "best_experiment_id": task.best_experiment_id,
        "best_run_id": task.best_run_id,
        "config": task.config or {},
        "summary_snapshot": task.summary_snapshot or {},
        # V3 Phase 2 — plan binding
        "training_plan_id": task.training_plan_id,
        "training_plan_snapshot": task.training_plan_snapshot,
        "created_at": task.created_at.isoformat() if task.created_at else None,
        "updated_at": task.updated_at.isoformat() if task.updated_at else None,
        "finished_at": task.finished_at.isoformat() if task.finished_at else None,
    }


def serialize_experiment(exp: PlatformExperiment) -> dict[str, Any]:
    """Serializer that includes the new V3 strategy fields."""
    return {
        "id": exp.id,
        "modeling_task_id": exp.modeling_task_id,
        "name": exp.name,
        "description": exp.description,
        "dataset_id": exp.dataset_id,
        "dataset_name": exp.dataset_name,
        "dataset_version_id": exp.dataset_version_id,
        "objective_metric": exp.objective_metric,
        "objective_direction": exp.objective_direction,
        "status": exp.status,
        "kind": exp.kind,                       # legacy
        "strategy_type": exp.strategy_type,      # V3
        "selected_models": exp.selected_models or [],
        "search_space": exp.search_space or {},
        "budget_config": exp.budget_config or {},
        "best_run_id": exp.best_run_id,
        "config": exp.config or {},
        "created_at": exp.created_at.isoformat() if exp.created_at else None,
        "updated_at": exp.updated_at.isoformat() if exp.updated_at else None,
        "finished_at": exp.finished_at.isoformat() if exp.finished_at else None,
    }


# ---------------------------------------------------------------------------
# Lookups
# ---------------------------------------------------------------------------

async def _get_task_or_404(db: AsyncSession, task_id: str) -> ModelingTask:
    result = await db.execute(select(ModelingTask).where(ModelingTask.id == task_id))
    task = result.scalar_one_or_none()
    if task is None:
        raise HTTPException(status_code=404, detail=f"ModelingTask {task_id!r} not found")
    return task


# ---------------------------------------------------------------------------
# CRUD
# ---------------------------------------------------------------------------

async def list_modeling_tasks(
    db: AsyncSession,
    *,
    page: int = 1,
    page_size: int = 20,
    status: str | None = None,
    dataset_id: str | None = None,
) -> dict[str, Any]:
    stmt = select(ModelingTask)
    count_stmt = select(func.count(ModelingTask.id))
    if status:
        stmt = stmt.where(ModelingTask.status == status.upper())
        count_stmt = count_stmt.where(ModelingTask.status == status.upper())
    if dataset_id:
        stmt = stmt.where(ModelingTask.dataset_id == dataset_id)
        count_stmt = count_stmt.where(ModelingTask.dataset_id == dataset_id)

    total = (await db.execute(count_stmt)).scalar_one()
    rows = await db.execute(
        stmt.order_by(ModelingTask.created_at.desc())
        .offset((page - 1) * page_size)
        .limit(page_size)
    )
    tasks = rows.scalars().all()

    items = []
    for t in tasks:
        payload = serialize_modeling_task(t)
        # Attach a cheap count of experiments + successful runs for list cards.
        exp_rows = await db.execute(
            select(PlatformExperiment.id).where(PlatformExperiment.modeling_task_id == t.id)
        )
        exp_ids = [r for (r,) in exp_rows.all()]
        payload["experiment_count"] = len(exp_ids)
        if exp_ids:
            run_count = (
                await db.execute(
                    select(func.count(ExperimentRun.id)).where(
                        ExperimentRun.experiment_id.in_(exp_ids),
                        ExperimentRun.status == "SUCCESS",
                    )
                )
            ).scalar_one()
        else:
            run_count = 0
        payload["successful_run_count"] = int(run_count)
        items.append(payload)

    return {"items": items, "total": total, "page": page, "page_size": page_size}


async def create_modeling_task(
    db: AsyncSession,
    *,
    name: str,
    dataset_id: str | None,
    target_column: str | None,
    task_type: str = "classification",
    objective_metric: str = "accuracy",
    objective_direction: str = "max",
    description: str | None = None,
    dataset_version_id: str | None = None,
    config: dict | None = None,
    training_plan_id: str | None = None,
) -> dict[str, Any]:
    if task_type not in ("classification", "regression"):
        raise HTTPException(status_code=422, detail="task_type must be classification|regression")
    if objective_direction not in ("max", "min"):
        raise HTTPException(status_code=422, detail="objective_direction must be max|min")

    dataset_name: str | None = None
    ds: Dataset | None = None
    if dataset_id:
        ds = (await db.execute(select(Dataset).where(Dataset.id == dataset_id))).scalar_one_or_none()
        if ds is None:
            raise HTTPException(status_code=404, detail=f"Dataset {dataset_id!r} not found")
        dataset_name = ds.name
        _validate_target_column(ds, target_column, task_type)

    if dataset_version_id:
        dv = (
            await db.execute(
                select(DatasetVersion).where(DatasetVersion.id == dataset_version_id)
            )
        ).scalar_one_or_none()
        if dv is None:
            raise HTTPException(
                status_code=404, detail=f"DatasetVersion {dataset_version_id!r} not found"
            )

    # -----------------------------------------------------------------------
    # V3 Phase 2 — bind to a TrainingPlan and freeze a snapshot.
    # We validate the plan's task_type agrees so the user never picks a
    # regression plan for a classification task (would produce nonsense runs).
    # The snapshot is frozen HERE (at bind time) — any subsequent plan edits
    # don't retroactively rewrite this task's recipe.
    # -----------------------------------------------------------------------
    training_plan_snapshot: dict | None = None
    if training_plan_id:
        from app.models.database import TrainingPlan
        from app.services.training_plan_service import capture_snapshot, mark_used

        plan = (
            await db.execute(
                select(TrainingPlan).where(TrainingPlan.id == training_plan_id)
            )
        ).scalar_one_or_none()
        if plan is None:
            raise HTTPException(
                status_code=404,
                detail=f"TrainingPlan {training_plan_id!r} not found",
            )
        if plan.task_type and plan.task_type != task_type:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"TrainingPlan task_type={plan.task_type!r} does not match "
                    f"ModelingTask task_type={task_type!r}"
                ),
            )
        training_plan_snapshot = capture_snapshot(plan)
        # Track usage — side effect, does not alter the snapshot
        await mark_used(db, plan.id)

    task = ModelingTask(
        name=name,
        description=description,
        dataset_id=dataset_id,
        dataset_name=dataset_name,
        dataset_version_id=dataset_version_id,
        target_column=target_column,
        task_type=task_type,
        objective_metric=objective_metric,
        objective_direction=objective_direction,
        status="CREATED",
        config=config,
        training_plan_id=training_plan_id,
        training_plan_snapshot=training_plan_snapshot,
    )
    db.add(task)
    await db.flush()
    await db.refresh(task)
    return serialize_modeling_task(task)


def _validate_target_column(
    dataset: Dataset,
    target_column: str | None,
    task_type: str,
) -> None:
    """Reject obviously invalid target columns before creating V3 work."""
    if not target_column:
        return

    try:
        df = load_dataframe(dataset.file_path)
    except Exception as exc:  # noqa: BLE001
        # Historical tests and stale dev DB rows can point at unavailable
        # files. Keep creation best-effort; actual training still fails with
        # the concrete file error if the user launches a run.
        logger.warning(
            "Skipping target validation for dataset %s; failed to read %s: %s",
            dataset.id,
            dataset.file_path,
            exc,
        )
        return

    if target_column not in df.columns:
        raise HTTPException(
            status_code=422,
            detail=f"目标列 {target_column!r} 不存在，请从数据集列中重新选择。",
        )

    series = df[target_column].dropna()
    if series.empty:
        raise HTTPException(
            status_code=422,
            detail=f"目标列 {target_column!r} 全为空，不能用于建模。",
        )

    if task_type == "regression":
        if not pd.api.types.is_numeric_dtype(series):
            raise HTTPException(
                status_code=422,
                detail=f"回归目标列 {target_column!r} 必须是数值列。",
            )
        return

    counts = series.value_counts(dropna=True)
    unique_count = int(counts.shape[0])
    total = int(series.shape[0])
    min_class_count = int(counts.min()) if unique_count else 0
    unique_rate = unique_count / total if total else 0.0

    if unique_count < 2:
        raise HTTPException(
            status_code=422,
            detail=f"分类目标列 {target_column!r} 只有 {unique_count} 个类别，至少需要 2 个类别。",
        )

    if min_class_count < 2 or unique_count == total or unique_rate > 0.5:
        raise HTTPException(
            status_code=422,
            detail=(
                f"分类目标列 {target_column!r} 不适合作为标签："
                f"类别数 {unique_count} / 样本数 {total}，最小类别样本数 {min_class_count}。"
                "这通常是 ID、序号或高基数字段；请改选 Target、label、Failure Type 等真实标签列。"
            ),
        )


async def get_modeling_task(db: AsyncSession, task_id: str) -> dict[str, Any]:
    task = await _get_task_or_404(db, task_id)
    payload = serialize_modeling_task(task)

    # -----------------------------------------------------------------------
    # V3 Phase 2 — plan staleness indicator
    # When the linked plan's current version != the version frozen at bind
    # time, the UI surfaces a "plan updated since bind" banner.  We never
    # overwrite the snapshot automatically; that would defeat reproducibility.
    # -----------------------------------------------------------------------
    if task.training_plan_id and task.training_plan_snapshot:
        from app.models.database import TrainingPlan

        plan_row = (
            await db.execute(
                select(TrainingPlan.version, TrainingPlan.name)
                .where(TrainingPlan.id == task.training_plan_id)
            )
        ).first()
        snapshot_version = (
            (task.training_plan_snapshot or {}).get("plan_version")
            or 1
        )
        payload["training_plan_status"] = {
            "plan_exists": plan_row is not None,
            "snapshot_version": snapshot_version,
            "current_version": plan_row[0] if plan_row else None,
            "stale": bool(plan_row and plan_row[0] != snapshot_version),
            "current_name": plan_row[1] if plan_row else None,
        }

    exp_rows = await db.execute(
        select(PlatformExperiment)
        .where(PlatformExperiment.modeling_task_id == task.id)
        .order_by(PlatformExperiment.created_at.desc())
    )
    experiments = [serialize_experiment(e) for e in exp_rows.scalars().all()]
    payload["experiments"] = experiments

    run_count = 0
    success_count = 0
    running_count = 0
    failed_count = 0
    if experiments:
        exp_ids = [e["id"] for e in experiments]
        rows = await db.execute(
            select(ExperimentRun.status, func.count(ExperimentRun.id))
            .where(ExperimentRun.experiment_id.in_(exp_ids))
            .group_by(ExperimentRun.status)
        )
        counts = {s: int(c) for s, c in rows.all()}
        run_count = sum(counts.values())
        success_count = counts.get("SUCCESS", 0)
        running_count = counts.get("RUNNING", 0) + counts.get("PENDING", 0) + counts.get("QUEUED", 0)
        failed_count = counts.get("FAILED", 0)

    payload["run_stats"] = {
        "total": run_count,
        "success": success_count,
        "running": running_count,
        "failed": failed_count,
    }
    return payload


async def update_modeling_task(
    db: AsyncSession,
    task_id: str,
    *,
    name: str | None = None,
    description: str | None = None,
    status: str | None = None,
    objective_metric: str | None = None,
    objective_direction: str | None = None,
) -> dict[str, Any]:
    task = await _get_task_or_404(db, task_id)
    if name is not None:
        task.name = name
    if description is not None:
        task.description = description
    if status is not None:
        status_u = status.upper()
        if status_u not in ("CREATED", "RUNNING", "COMPLETED", "FAILED", "ARCHIVED"):
            raise HTTPException(status_code=422, detail="Invalid status")
        task.status = status_u
        if status_u in ("COMPLETED", "FAILED", "ARCHIVED") and task.finished_at is None:
            task.finished_at = _utcnow()
    if objective_metric is not None:
        task.objective_metric = objective_metric
    if objective_direction is not None:
        if objective_direction not in ("max", "min"):
            raise HTTPException(status_code=422, detail="objective_direction must be max|min")
        task.objective_direction = objective_direction
    await db.flush()
    await db.refresh(task)
    return serialize_modeling_task(task)


async def delete_modeling_task(db: AsyncSession, task_id: str) -> None:
    task = await _get_task_or_404(db, task_id)
    if task.status == "RUNNING":
        raise HTTPException(status_code=400, detail="Cannot delete a running modeling task")
    await db.delete(task)
    await db.flush()


# ---------------------------------------------------------------------------
# Aggregated leaderboard across all experiments of a task
# ---------------------------------------------------------------------------

def _pick_metric(metrics: dict, metric_key: str) -> float | None:
    v = (metrics or {}).get(metric_key)
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


async def task_leaderboard(db: AsyncSession, task_id: str, top_k: int = 20) -> list[dict[str, Any]]:
    """
    Global leaderboard: best runs across every experiment under this task,
    ranked by the task's objective metric.
    """
    task = await _get_task_or_404(db, task_id)

    exp_rows = await db.execute(
        select(PlatformExperiment.id, PlatformExperiment.name, PlatformExperiment.strategy_type)
        .where(PlatformExperiment.modeling_task_id == task_id)
    )
    exp_index = {eid: {"name": name, "strategy_type": strategy}
                 for eid, name, strategy in exp_rows.all()}
    if not exp_index:
        return []

    run_rows = await db.execute(
        select(ExperimentRun)
        .where(
            ExperimentRun.experiment_id.in_(exp_index.keys()),
            ExperimentRun.status == "SUCCESS",
        )
    )
    runs = run_rows.scalars().all()

    scored = []
    for run in runs:
        v = _pick_metric(run.metrics or {}, task.objective_metric or "accuracy")
        if v is None:
            continue
        scored.append((run, v))

    scored.sort(key=lambda x: x[1], reverse=(task.objective_direction or "max") == "max")

    return [
        {
            "rank": idx + 1,
            "run_id": run.id,
            "experiment_id": run.experiment_id,
            "experiment_name": exp_index[run.experiment_id]["name"],
            "strategy_type": exp_index[run.experiment_id]["strategy_type"]
                              or run.source_experiment_type,
            "trial_no": run.trial_no,
            "objective_value": value,
            "metric_name": task.objective_metric,
            "params": run.params or {},
            "metrics": run.metrics or {},
            "finished_at": run.finished_at.isoformat() if run.finished_at else None,
        }
        for idx, (run, value) in enumerate(scored[:top_k])
    ]


async def task_runs(
    db: AsyncSession,
    task_id: str,
    *,
    status: str | None = None,
) -> dict[str, Any]:
    """Return all runs for a modeling task, including scheduler progress."""
    task = await _get_task_or_404(db, task_id)

    exp_rows = await db.execute(
        select(PlatformExperiment)
        .where(PlatformExperiment.modeling_task_id == task_id)
        .order_by(PlatformExperiment.created_at.desc())
    )
    experiments = exp_rows.scalars().all()
    if not experiments:
        return {"items": [], "total": 0}

    exp_index = {
        exp.id: {
            "name": exp.name,
            "strategy_type": exp.strategy_type,
            "status": exp.status,
        }
        for exp in experiments
    }

    stmt = select(ExperimentRun).where(ExperimentRun.experiment_id.in_(exp_index.keys()))
    if status:
        stmt = stmt.where(ExperimentRun.status == status.upper())
    run_rows = await db.execute(
        stmt.order_by(
            ExperimentRun.created_at.desc(),
            ExperimentRun.trial_no.asc(),
        )
    )
    runs = run_rows.scalars().all()

    task_ids = [run.task_id for run in runs if run.task_id]
    platform_tasks: dict[str, PlatformTask] = {}
    if task_ids:
        ptask_rows = await db.execute(select(PlatformTask).where(PlatformTask.id.in_(task_ids)))
        platform_tasks = {pt.id: pt for pt in ptask_rows.scalars().all()}

    def _platform_payload(pt: PlatformTask | None) -> dict[str, Any] | None:
        if pt is None:
            return None
        return {
            "id": pt.id,
            "kind": pt.kind,
            "status": pt.status,
            "progress": pt.progress,
            "payload_ref": pt.payload_ref,
            "retry_count": pt.retry_count,
            "max_retries": pt.max_retries,
            "error_message": pt.error_message,
            "queued_at": pt.queued_at.isoformat() if pt.queued_at else None,
            "started_at": pt.started_at.isoformat() if pt.started_at else None,
            "finished_at": pt.finished_at.isoformat() if pt.finished_at else None,
        }

    items = []
    for run in runs:
        exp_meta = exp_index.get(run.experiment_id, {})
        pt = platform_tasks.get(run.task_id) if run.task_id else None
        objective_value = _pick_metric(run.metrics or {}, task.objective_metric or "accuracy")
        items.append({
            "run_id": run.id,
            "experiment_id": run.experiment_id,
            "experiment_name": exp_meta.get("name"),
            "experiment_status": exp_meta.get("status"),
            "strategy_type": exp_meta.get("strategy_type") or run.source_experiment_type,
            "trial_no": run.trial_no,
            "rank": run.rank,
            "status": run.status,
            "objective_value": objective_value,
            "metric_name": task.objective_metric,
            "params": run.params or {},
            "metrics": run.metrics or {},
            "search_meta": run.search_meta or {},
            "platform_task": _platform_payload(pt),
            "created_at": run.created_at.isoformat() if run.created_at else None,
            "started_at": run.started_at.isoformat() if run.started_at else None,
            "finished_at": run.finished_at.isoformat() if run.finished_at else None,
        })

    reverse = (task.objective_direction or "max") == "max"
    items.sort(
        key=lambda row: (
            row["objective_value"] is None,
            -float(row["objective_value"] or 0) if reverse else float(row["objective_value"] or 0),
        )
    )
    return {"items": items, "total": len(items)}


# ---------------------------------------------------------------------------
# Strategy comparison — baseline vs grid_search vs bayesian_search
# ---------------------------------------------------------------------------


def _quartiles(values: list[float]) -> dict[str, float] | None:
    """Return {min, q1, median, q3, max} using linear interpolation.

    Avoids pulling in numpy for a five-number summary — tiny helper so the
    aggregation endpoint stays hot-reload-friendly and doesn't shell out to
    heavy deps.  Returns None for empty input.
    """
    if not values:
        return None
    xs = sorted(values)
    n = len(xs)

    def _pct(p: float) -> float:
        if n == 1:
            return xs[0]
        rank = p * (n - 1)
        lo = int(rank)
        hi = min(lo + 1, n - 1)
        frac = rank - lo
        return xs[lo] + (xs[hi] - xs[lo]) * frac

    return {
        "min": float(xs[0]),
        "q1": float(_pct(0.25)),
        "median": float(_pct(0.50)),
        "q3": float(_pct(0.75)),
        "max": float(xs[-1]),
    }


async def strategy_comparison(db: AsyncSession, task_id: str) -> dict[str, Any]:
    """Compare baseline vs grid_search vs bayesian_search for this task.

    For each strategy:
      - run_count (SUCCESS only for stats; full_count for book-keeping)
      - five-number summary of the objective metric
      - best_run (the top run by direction)
    Also returns `raw_points` — one row per SUCCESS run — so the UI can
    render a box plot, strip plot, or table without a second round-trip.
    """
    task = await _get_task_or_404(db, task_id)
    metric_name = task.objective_metric or "accuracy"
    reverse = (task.objective_direction or "max") == "max"

    exp_rows = await db.execute(
        select(
            PlatformExperiment.id,
            PlatformExperiment.name,
            PlatformExperiment.strategy_type,
        ).where(PlatformExperiment.modeling_task_id == task_id)
    )
    experiments = exp_rows.all()
    if not experiments:
        return {
            "task_id": task_id,
            "metric_name": metric_name,
            "objective_direction": task.objective_direction or "max",
            "strategies": [],
            "raw_points": [],
        }

    exp_index = {
        eid: {"name": name, "strategy_type": strategy}
        for eid, name, strategy in experiments
    }

    run_rows = await db.execute(
        select(ExperimentRun).where(
            ExperimentRun.experiment_id.in_(exp_index.keys())
        )
    )
    runs = list(run_rows.scalars().all())

    # Bucket runs by strategy_type — fall back to run.source_experiment_type
    # when the exp no longer carries a label (legacy data).
    buckets: dict[str, list[tuple[ExperimentRun, float]]] = {}
    full_counts: dict[str, int] = {}
    raw_points: list[dict[str, Any]] = []
    for run in runs:
        strategy = (
            exp_index.get(run.experiment_id, {}).get("strategy_type")
            or run.source_experiment_type
            or "unknown"
        )
        full_counts[strategy] = full_counts.get(strategy, 0) + 1
        if run.status != "SUCCESS":
            continue
        value = _pick_metric(run.metrics or {}, metric_name)
        if value is None:
            continue
        buckets.setdefault(strategy, []).append((run, value))
        raw_points.append({
            "strategy_type": strategy,
            "run_id": run.id,
            "experiment_id": run.experiment_id,
            "trial_no": run.trial_no,
            "value": value,
            "model_type": (run.params or {}).get("model_type")
                or (run.search_meta or {}).get("model_type"),
        })

    strategies: list[dict[str, Any]] = []
    # Stable ordering: canonical strategies first, then anything else alphabetically.
    canonical = ["baseline", "grid_search", "bayesian_search"]
    ordered_keys = [s for s in canonical if s in full_counts] + sorted(
        k for k in full_counts if k not in canonical
    )
    for strategy in ordered_keys:
        items = buckets.get(strategy, [])
        values = [v for _, v in items]
        stats = _quartiles(values)
        best_run: dict[str, Any] | None = None
        if items:
            best_tuple = max(items, key=lambda t: t[1]) if reverse else min(items, key=lambda t: t[1])
            run, value = best_tuple
            best_run = {
                "run_id": run.id,
                "experiment_id": run.experiment_id,
                "trial_no": run.trial_no,
                "objective_value": value,
                "params": run.params or {},
                "metrics": run.metrics or {},
                "model_type": (run.params or {}).get("model_type")
                    or (run.search_meta or {}).get("model_type"),
                "finished_at": run.finished_at.isoformat() if run.finished_at else None,
            }
        strategies.append({
            "strategy_type": strategy,
            "run_count": len(items),          # SUCCESS-only, used for stats
            "full_run_count": full_counts.get(strategy, 0),
            "stats": stats,
            "best_run": best_run,
        })

    return {
        "task_id": task_id,
        "metric_name": metric_name,
        "objective_direction": task.objective_direction or "max",
        "strategies": strategies,
        "raw_points": raw_points,
    }


async def refresh_task_summary(db: AsyncSession, task_id: str) -> None:
    """
    Recompute ``best_experiment_id`` / ``best_run_id`` / ``summary_snapshot``
    from the latest run state.  Called after every run completion.
    """
    task = await _get_task_or_404(db, task_id)
    board = await task_leaderboard(db, task_id, top_k=1)
    if board:
        top = board[0]
        task.best_experiment_id = top["experiment_id"]
        task.best_run_id = top["run_id"]

    # Stats block for the task card — small, serialisable, cheap.
    exp_rows = await db.execute(
        select(PlatformExperiment.id, PlatformExperiment.status)
        .where(PlatformExperiment.modeling_task_id == task_id)
    )
    exp_rows_list = exp_rows.all()
    exp_total = len(exp_rows_list)
    exp_running = sum(1 for _, s in exp_rows_list if s == "RUNNING")
    exp_completed = sum(1 for _, s in exp_rows_list if s == "COMPLETED")
    exp_failed = sum(1 for _, s in exp_rows_list if s == "FAILED")

    task.summary_snapshot = {
        "experiment_count": exp_total,
        "experiment_running": exp_running,
        "experiment_completed": exp_completed,
        "experiment_failed": exp_failed,
        "top_objective_value": board[0]["objective_value"] if board else None,
        "top_strategy_type": board[0]["strategy_type"] if board else None,
        "refreshed_at": _utcnow().isoformat(),
    }

    # Task status rollup.
    if exp_total > 0:
        if exp_running > 0:
            task.status = "RUNNING"
        elif exp_completed == exp_total:
            task.status = "COMPLETED"
            if task.finished_at is None:
                task.finished_at = _utcnow()
        elif exp_failed > 0 and exp_completed + exp_failed == exp_total:
            task.status = "FAILED" if exp_completed == 0 else "COMPLETED"

    await db.flush()


# ---------------------------------------------------------------------------
# Tuning-space registry access
# ---------------------------------------------------------------------------

_TUNING_SPACES_PATH = Path(__file__).resolve().parents[2] / "registry" / "tuning_spaces.yaml"


def load_tuning_spaces(task_type: str = "classification") -> dict[str, Any]:
    """Load parameter templates (grid + distribution) from the YAML registry."""
    if not _TUNING_SPACES_PATH.exists():
        raise FileNotFoundError(f"tuning_spaces.yaml not found at {_TUNING_SPACES_PATH}")
    with _TUNING_SPACES_PATH.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if task_type not in data:
        raise ValueError(f"task_type {task_type!r} not in tuning_spaces.yaml")
    return data[task_type]

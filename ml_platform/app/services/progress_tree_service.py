"""
V3 Phase 4 — per-model progress tree.

``GET /api/v3/tasks/{id}/progress-tree`` returns a three-level aggregate:

    ModelingTask
      └── PlatformExperiment (strategy_type)
            └── ExperimentRun  (one per model / per trial)

For each level we compute a single ``progress_aggregated ∈ [0, 1]`` so the
frontend can draw a single progress bar per node.  For run leaves we
surface a ``current_step`` hint — epoch counter for DL, fold index
heuristic for ML — to make "where is it now" obvious without needing to
click into logs.

Design notes
------------
* Aggregation is plain averaging of child-node progress.  It is
  intentionally not time-weighted: for the user's mental model, "I
  launched 5 models, 3 done, 2 at 50%" reading 80% is more useful than
  something weighted by historical step counts.
* The service never *writes* progress — it only reads what the executors
  (training_service / dl_service) have already persisted on the
  underlying ``PlatformTask`` / ``DLTrainingTask`` rows.
* Domain lookups are batched (one SELECT per kind over the whole tree)
  to keep the endpoint O(1) per HTTP request even for 50-run sweeps.
"""

from __future__ import annotations

from typing import Any, Iterable

from fastapi import HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.database import (
    DLTrainingTask,
    ExperimentRun,
    ModelingTask,
    PlatformExperiment,
    PlatformTask,
    TrainingTask,
)


# ---------------------------------------------------------------------------
# Model family classification (shared with tuning_service)
# ---------------------------------------------------------------------------

def _infer_family(model_type: str | None, payload_kind: str | None) -> str:
    """Return 'dl' | 'ml' | 'unknown' — best-effort family label."""
    if payload_kind == "dl_train":
        return "dl"
    if payload_kind == "train":
        return "ml"
    # Fall back to registry inspection; tolerant of unknown strings.
    if model_type:
        try:
            from app.core.dl_registry import DL_MODELS  # type: ignore
            if model_type in DL_MODELS:
                return "dl"
        except Exception:  # pragma: no cover — registry not mandatory
            pass
    return "ml" if model_type else "unknown"


# ---------------------------------------------------------------------------
# Per-run current-step hint
# ---------------------------------------------------------------------------

def _current_step(
    run: ExperimentRun,
    platform_task: PlatformTask | None,
    dl_task: DLTrainingTask | None,
) -> str | None:
    """Return a short "what's happening right now" string for the leaf."""
    status = (run.status or "").upper()
    if status in ("SUCCESS", "COMPLETED"):
        return "已完成"
    if status == "FAILED":
        return "失败"
    if status == "CANCELLED":
        return "已取消"
    if status in ("PENDING", "QUEUED"):
        return "等待调度"

    # RUNNING — prefer DL epoch counter when available
    if dl_task is not None and dl_task.total_epochs:
        return f"epoch {dl_task.current_epoch}/{dl_task.total_epochs}"

    # ML fallback — derive from progress as a coarse indicator.  We use
    # the PlatformTask.progress because it's the writer-of-record during
    # training (training_service writes it on every fold completion).
    progress = 0.0
    if platform_task and platform_task.progress is not None:
        progress = float(platform_task.progress)
    pct = int(round(progress * 100))
    return f"训练中 ({pct}%)"


# ---------------------------------------------------------------------------
# Progress extraction
# ---------------------------------------------------------------------------

def _run_progress(
    run: ExperimentRun,
    platform_task: PlatformTask | None,
    dl_task: DLTrainingTask | None,
) -> float:
    """Return progress in [0, 1] for a single run leaf."""
    status = (run.status or "").upper()
    if status in ("SUCCESS", "COMPLETED"):
        return 1.0
    if status in ("FAILED", "CANCELLED"):
        # These are terminal — count as "done" for aggregation purposes so
        # one failed run doesn't keep the whole batch stuck at 99%.
        return 1.0

    # Prefer DL's epoch-derived fraction because it updates per-epoch.
    if dl_task is not None and dl_task.total_epochs:
        return max(0.0, min(1.0, (dl_task.current_epoch or 0) / dl_task.total_epochs))
    if platform_task and platform_task.progress is not None:
        return max(0.0, min(1.0, float(platform_task.progress)))
    return 0.0


def _avg(values: Iterable[float]) -> float:
    vals = list(values)
    if not vals:
        return 0.0
    return sum(vals) / len(vals)


# ---------------------------------------------------------------------------
# Public entrypoint
# ---------------------------------------------------------------------------

async def get_progress_tree(db: AsyncSession, modeling_task_id: str) -> dict[str, Any]:
    """Build the {modeling_task, experiments:[{runs:[…]}]} tree."""
    # 1. Modeling task itself
    mt_row = await db.execute(
        select(ModelingTask).where(ModelingTask.id == modeling_task_id)
    )
    mt = mt_row.scalar_one_or_none()
    if mt is None:
        raise HTTPException(
            status_code=404, detail=f"ModelingTask {modeling_task_id!r} not found"
        )

    # 2. Experiments + runs (two batched queries)
    exps_row = await db.execute(
        select(PlatformExperiment)
        .where(PlatformExperiment.modeling_task_id == modeling_task_id)
        .order_by(PlatformExperiment.created_at.asc())
    )
    experiments = exps_row.scalars().all()

    runs: list[ExperimentRun] = []
    if experiments:
        exp_ids = [e.id for e in experiments]
        runs_row = await db.execute(
            select(ExperimentRun)
            .where(ExperimentRun.experiment_id.in_(exp_ids))
            .order_by(
                ExperimentRun.rank.is_(None),
                ExperimentRun.rank.asc(),
                ExperimentRun.trial_no.asc(),
                ExperimentRun.created_at.asc(),
            )
        )
        runs = list(runs_row.scalars().all())

    # 3. Platform tasks behind each run (one batched SELECT)
    ptask_ids = [r.task_id for r in runs if r.task_id]
    platform_tasks: dict[str, PlatformTask] = {}
    if ptask_ids:
        ptask_row = await db.execute(
            select(PlatformTask).where(PlatformTask.id.in_(ptask_ids))
        )
        platform_tasks = {t.id: t for t in ptask_row.scalars().all()}

    # 4. DL domain tasks (one batched SELECT) — only for payload_ref=dl_train:*
    dl_task_ids = []
    ml_task_ids = []
    for ptask in platform_tasks.values():
        if not ptask.payload_ref:
            continue
        kind, _, domain_id = ptask.payload_ref.partition(":")
        if not domain_id:
            continue
        if kind == "dl_train":
            dl_task_ids.append(domain_id)
        elif kind == "train":
            ml_task_ids.append(domain_id)

    dl_tasks_by_id: dict[str, DLTrainingTask] = {}
    if dl_task_ids:
        r = await db.execute(
            select(DLTrainingTask).where(DLTrainingTask.id.in_(dl_task_ids))
        )
        dl_tasks_by_id = {t.id: t for t in r.scalars().all()}

    # Also pull ML TrainingTask rows so we can render the model_type when
    # the run.params didn't stash it.
    ml_tasks_by_id: dict[str, TrainingTask] = {}
    if ml_task_ids:
        r = await db.execute(
            select(TrainingTask).where(TrainingTask.id.in_(ml_task_ids))
        )
        ml_tasks_by_id = {t.id: t for t in r.scalars().all()}

    # 5. Assemble nested payload
    experiments_payload: list[dict[str, Any]] = []
    all_run_progresses: list[float] = []
    any_active = False
    active_statuses = {"RUNNING", "QUEUED", "PENDING", "RETRY"}

    for exp in experiments:
        exp_runs = [r for r in runs if r.experiment_id == exp.id]
        exp_payload_runs: list[dict[str, Any]] = []
        exp_run_progresses: list[float] = []

        for run in exp_runs:
            ptask = platform_tasks.get(run.task_id) if run.task_id else None
            payload_kind: str | None = None
            domain_id: str | None = None
            if ptask and ptask.payload_ref:
                payload_kind, _, domain_id = ptask.payload_ref.partition(":")

            dl_task = dl_tasks_by_id.get(domain_id) if payload_kind == "dl_train" else None
            ml_task = ml_tasks_by_id.get(domain_id) if payload_kind == "train" else None

            model_type = (
                (run.params or {}).get("model_type")
                or (dl_task.model_type if dl_task else None)
                or (ml_task.model_type if ml_task else None)
            )
            family = _infer_family(model_type, payload_kind)

            progress = _run_progress(run, ptask, dl_task)
            exp_run_progresses.append(progress)
            all_run_progresses.append(progress)

            status_upper = (run.status or "").upper()
            if status_upper in active_statuses:
                any_active = True

            exp_payload_runs.append({
                "id": run.id,
                "trial_no": run.trial_no,
                "rank": run.rank,
                "model_type": model_type,
                "family": family,
                "status": run.status,
                "progress": round(progress, 4),
                "current_step": _current_step(run, ptask, dl_task),
                "platform_task_id": ptask.id if ptask else None,
                "domain_kind": payload_kind,
                "domain_id": domain_id,
            })

        experiments_payload.append({
            "id": exp.id,
            "name": exp.name,
            "strategy_type": exp.strategy_type,
            "status": exp.status,
            "selected_models": exp.selected_models or [],
            "progress_aggregated": round(_avg(exp_run_progresses), 4),
            "run_count": len(exp_runs),
            "runs": exp_payload_runs,
        })

    return {
        "modeling_task": {
            "id": mt.id,
            "name": mt.name,
            "status": mt.status,
            "progress_aggregated": round(_avg(all_run_progresses), 4),
            "experiment_count": len(experiments),
            "run_count": len(runs),
        },
        "experiments": experiments_payload,
        "has_active_runs": any_active,
    }

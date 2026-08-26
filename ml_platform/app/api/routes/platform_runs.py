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

from app.core.auth import current_username_from_authorization, owner_scope_username
from app.core.ownership import ensure_task_owner
from app.models.database import (
    Dataset,
    DLTrainingLog,
    DLTrainingTask,
    ExperimentRun,
    ExperimentRunLog,
    PlatformExperiment,
    PlatformTask,
    TrainingLog,
    TrainingTask,
    get_db,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/platform/runs", tags=["V3 Run Inspector"])


async def owned_run_id(
    run_id: str,
    db: AsyncSession = Depends(get_db),
    username: str = Depends(current_username_from_authorization),
) -> str:
    await ensure_task_owner(db, run_id, owner_scope_username(username))
    return run_id


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
        # M2c: terminal failure reason, survives PlatformTask cleanup.
        "error_message": run.error_message,
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
        "family": "ml",
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


def _serialize_dl_training_task(
    task: DLTrainingTask, dataset: Dataset | None
) -> dict[str, Any]:
    return {
        "id": task.id,
        "family": "dl",
        "name": task.name,
        "model_type": task.model_type,
        "task_type": task.task_type,
        "arch_config": task.arch_config or {},
        "opt_config": task.opt_config or {},
        "train_config": task.train_config or {},
        "status": task.status,
        "progress": task.progress,
        "current_epoch": task.current_epoch,
        "total_epochs": task.total_epochs,
        "model_path": task.model_path,
        "result_metrics": task.result_metrics or {},
        "dataset": {
            "id": dataset.id if dataset else task.dataset_id,
            "name": dataset.name if dataset else None,
            "row_count": dataset.row_count if dataset else None,
            "column_count": dataset.column_count if dataset else None,
        } if (dataset or task.dataset_id) else None,
    }


def _serialize_log(
    log: TrainingLog | DLTrainingLog | ExperimentRunLog,
) -> dict[str, Any]:
    """Shape-compatible serializer for both legacy TrainingLog and V3-native
    ExperimentRunLog rows — both expose level/message/extra/created_at.
    """
    return {
        "level": log.level,
        "message": log.message,
        "extra": log.extra or {},
        "created_at": _iso(log.created_at),
    }


# ---------------------------------------------------------------------------
# Inspector endpoint
# ---------------------------------------------------------------------------


async def _run_for_domain_task(db: AsyncSession, domain_id: str) -> ExperimentRun | None:
    """Find the ExperimentRun a domain training task belongs to, if any.

    PlatformTask.payload_ref is "<kind>:<domain_id>", and ExperimentRun.task_id
    points at the PlatformTask — so the hop is payload_ref -> task -> run.
    """
    refs = [f"{kind}:{domain_id}" for kind in ("train", "dl_train")]
    platform_task = (
        await db.execute(select(PlatformTask).where(PlatformTask.payload_ref.in_(refs)))
    ).scalars().first()
    if platform_task is None:
        return None
    return (
        await db.execute(
            select(ExperimentRun).where(ExperimentRun.task_id == platform_task.id)
        )
    ).scalars().first()


async def _domain_task_exists(db: AsyncSession, domain_id: str) -> bool:
    """True when the id names a training task, even one with no run."""
    for model in (TrainingTask, DLTrainingTask):
        found = (
            await db.execute(select(model.id).where(model.id == domain_id))
        ).scalar_one_or_none()
        if found is not None:
            return True
    return False


@router.get("/{run_id}/inspector", summary="Aggregated run detail for the Run Inspector drawer")
async def inspect_run(
    run_id: str = Depends(owned_run_id),
    log_limit: int = Query(100, ge=1, le=500),
    include_siblings: bool = Query(True),
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    # --- 1. Load run
    #
    # `run_id` is not always an ExperimentRun id. 模型管理 lists *domain* tasks
    # (training_tasks / dl_training_tasks) and links straight here, so accept a
    # domain id too and walk domain -> PlatformTask -> ExperimentRun.
    #
    # Some domain tasks have no run at all — V2-era rows, and DL tasks started
    # outside a V3 batch (3 of 7 on the current deployment). Those are not an
    # error: the model exists and its logs, config and charts are all worth
    # showing, so the response carries `run: null` and the caller degrades to
    # the task-level view rather than getting a 404.
    run = (
        await db.execute(select(ExperimentRun).where(ExperimentRun.id == run_id))
    ).scalar_one_or_none()
    if run is None:
        run = await _run_for_domain_task(db, run_id)
    if run is None and not await _domain_task_exists(db, run_id):
        raise HTTPException(status_code=404, detail=f"Run {run_id!r} not found")
    domain_fallback_id = None if run is not None else run_id

    # --- 2. Experiment context (for direction / metric / strategy)
    exp = None
    if run is not None:
        exp = (
            await db.execute(
                select(PlatformExperiment).where(PlatformExperiment.id == run.experiment_id)
            )
        ).scalar_one_or_none()

    # --- 3. Platform task (unified scheduler view)
    platform_task: PlatformTask | None = None
    domain_task_id: str | None = domain_fallback_id
    if run is not None and run.task_id:
        platform_task = (
            await db.execute(select(PlatformTask).where(PlatformTask.id == run.task_id))
        ).scalar_one_or_none()
        if platform_task and platform_task.payload_ref and ":" in platform_task.payload_ref:
            _, _, domain_task_id = platform_task.payload_ref.partition(":")

    # --- 4. Domain training task + dataset
    #
    # Logs / TrainingTask rows are keyed by the legacy trainer id, which lives
    # in PlatformTask.payload_ref = "train:<legacy_id>". When the legacy row
    # has been purged (or the run never created one), we still want to surface
    # the on-disk log file and synthesize a training_task view via the
    # resolver. That keeps the inspector useful for V3 runs whose log file is
    # keyed by a different id than run.id.
    training_task_payload: dict[str, Any] | None = None
    logs_payload: list[dict[str, Any]] = []
    resolved_log_task_id: str | None = None

    # ---- V3 native logs (preferred path, since v3.3.0) --------------------
    # `experiment_run_logs` is keyed by ExperimentRun.id and survives any
    # `DELETE FROM training_tasks` (which would CASCADE-kill the legacy
    # `training_logs` rows for this run).  We read here first; the legacy
    # chain below stays as a fallback for runs older than v3.3.0 (mirror
    # only started populating from that release on).
    v3_log_rows = (
        await db.execute(
            select(ExperimentRunLog)
            .where(ExperimentRunLog.run_id == run_id)
            .order_by(ExperimentRunLog.created_at.asc())
            .limit(log_limit)
        )
    ).scalars().all()
    if v3_log_rows:
        logs_payload = [_serialize_log(lg) for lg in v3_log_rows]
        resolved_log_task_id = run_id

    tt: TrainingTask | None = None
    if domain_task_id:
        # PlatformTask.kind normally says which family this is. A task that
        # never ran through a V3 batch has no PlatformTask at all, so probe the
        # DL table before defaulting to ML — otherwise a DL model opened from
        # 模型管理 silently resolves to "no training task".
        if platform_task is not None:
            is_dl_domain = platform_task.kind == "dl_train"
        else:
            is_dl_domain = (
                await db.execute(
                    select(DLTrainingTask.id).where(DLTrainingTask.id == domain_task_id)
                )
            ).scalar_one_or_none() is not None

        if is_dl_domain:
            dl_task = (
                await db.execute(
                    select(DLTrainingTask).where(DLTrainingTask.id == domain_task_id)
                )
            ).scalar_one_or_none()
            if dl_task:
                ds = (
                    await db.execute(select(Dataset).where(Dataset.id == dl_task.dataset_id))
                ).scalar_one_or_none()
                training_task_payload = _serialize_dl_training_task(dl_task, ds)

                if not logs_payload:
                    log_rows = await db.execute(
                        select(DLTrainingLog)
                        .where(DLTrainingLog.task_id == dl_task.id)
                        .order_by(DLTrainingLog.created_at.desc())
                        .limit(log_limit)
                    )
                    logs = list(log_rows.scalars().all())
                    logs.reverse()
                    logs_payload = [_serialize_log(log) for log in logs]
                    if logs_payload:
                        resolved_log_task_id = dl_task.id
        else:
            tt = (
                await db.execute(select(TrainingTask).where(TrainingTask.id == domain_task_id))
            ).scalar_one_or_none()
        if tt is not None and not is_dl_domain:
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
            resolved_log_task_id = tt.id

    # ---- Fallback path: no TrainingTask row (V3-native or purged) ----------
    # Walk the id-candidate chain (run.id → run.task_id → payload_ref legacy id)
    # and probe each for TrainingLog rows or a {id}.log file on disk. First hit
    # wins; synthesize a training_task facade from the resolver output.
    if not logs_payload or training_task_payload is None:
        from app.services.resolver import resolve_legacy_id_candidates, resolve_task_and_dataset

        candidates = await resolve_legacy_id_candidates(run_id, db)

        # DB-log search across the candidate chain
        if not logs_payload:
            for cid in candidates:
                log_rows = await db.execute(
                    select(TrainingLog)
                    .where(TrainingLog.task_id == cid)
                    .order_by(TrainingLog.created_at.desc())
                    .limit(log_limit)
                )
                logs = list(log_rows.scalars().all())
                if logs:
                    logs.reverse()
                    logs_payload = [_serialize_log(lg) for lg in logs]
                    resolved_log_task_id = cid
                    break

        # On-disk log fallback — `storage/logs/{id}.log` for purged / V3-native runs
        if not logs_payload:
            from app.config import get_settings
            settings = get_settings()
            for cid in candidates:
                log_path = settings.storage_logs / f"{cid}.log"
                if not log_path.exists():
                    continue
                try:
                    lines = log_path.read_text(errors="ignore").splitlines()[-log_limit:]
                except Exception:
                    continue
                # Parse "LEVEL timestamp | message" style lines — our TrainingLogger
                # writes plain text, so we do a best-effort split rather than
                # pretending these are ORM rows.
                for line in lines:
                    stripped = line.strip()
                    if not stripped:
                        continue
                    level = "INFO"
                    for candidate_level in ("ERROR", "WARNING", "INFO", "DEBUG"):
                        if candidate_level in stripped[:40]:
                            level = candidate_level
                            break
                    logs_payload.append({
                        "level": level,
                        "message": stripped,
                        "extra": {},
                        "created_at": None,
                    })
                if logs_payload:
                    resolved_log_task_id = cid
                    break

        # Synthesize a training_task view so the context tab has something
        # human-readable even when the legacy row is gone.
        if training_task_payload is None:
            try:
                facade, dataset = await resolve_task_and_dataset(run_id, db)
                if isinstance(facade, DLTrainingTask):
                    training_task_payload = _serialize_dl_training_task(facade, dataset)
                else:
                    training_task_payload = {
                        "id": getattr(facade, "id", run_id),
                        "family": "ml",
                        "name": None,
                        "model_type": getattr(facade, "model_type", None),
                        "hyperparameters": {},
                        "target_column": getattr(facade, "target_column", None),
                        "test_size": getattr(facade, "test_size", 0.2),
                        "eval_metrics": [],
                        "status": getattr(facade, "status", "UNKNOWN"),
                        "progress": 100,
                        "model_path": getattr(facade, "model_path", None),
                        "dataset": {
                            "id": dataset.id,
                            "name": dataset.name,
                            "row_count": dataset.row_count,
                            "column_count": dataset.column_count,
                        } if dataset else None,
                        "task_kind": getattr(facade, "task_kind", None),
                        "synthesized": True,
                    }
            except HTTPException:
                pass  # resolver raised — the run may still be pending
            except Exception as exc:
                logger.warning("Inspector training_task synthesis failed for %s: %s", run_id, exc)

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
                "params": r.params or {},  # needed for param-impact correlation
                "source_experiment_type": r.source_experiment_type,
            }
            for r in sib_rows.scalars().all()
        ]

    # --- 6. SHAP importances if already computed (inline)
    metrics_dict = (run.metrics if run is not None else None) or {}
    shap_importances = metrics_dict.get("shap_importances")
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
            "method": metrics_dict.get("shap_method", "shap"),
            "sample_count": metrics_dict.get("shap_sample_size"),
            "base_value": metrics_dict.get("shap_base_value"),
            "task_kind": metrics_dict.get("shap_task_kind"),
        }
    else:
        shap_summary = {"has_explanation": False}

    # --- 7. Auto-diagnosis narrative (overfit / failure / param impact / peer rank)
    experiment_payload = {
        "id": exp.id,
        "name": exp.name,
        "strategy_type": exp.strategy_type,
        "objective_metric": exp.objective_metric,
        "objective_direction": exp.objective_direction,
        "modeling_task_id": exp.modeling_task_id,
        "status": exp.status,
    } if exp else None

    try:
        from app.services.run_diagnosis_service import diagnose_run
        diagnosis = diagnose_run(
            run=_serialize_run(run),
            experiment=experiment_payload,
            siblings=siblings,
            logs=logs_payload,
        )
    except Exception as exc:  # pragma: no cover — diagnosis is best-effort
        logger.warning("Run diagnosis failed for %s: %s", run_id, exc)
        diagnosis = None

    return {
        "run": _serialize_run(run) if run is not None else None,
        "experiment": experiment_payload,
        "platform_task": _serialize_platform_task(platform_task) if platform_task else None,
        "training_task": training_task_payload,
        "logs": logs_payload,
        "log_task_id": resolved_log_task_id,
        "siblings": siblings,
        "shap": shap_summary,
        "diagnosis": diagnosis,
    }


# ---------------------------------------------------------------------------
# Full SHAP payload — fetches the MinIO artifact for beeswarm / dependence plots
# ---------------------------------------------------------------------------

@router.get(
    "/{run_id}/shap",
    summary="Full SHAP payload for a run (inline importances + per-sample values)",
)
async def get_shap_payload(
    run_id: str = Depends(owned_run_id),
    compute: bool = Query(False, description="Compute on-demand if no cached payload exists"),
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    """Return the full SHAP payload for ``run_id``.

    Shape:
    {
        "run_id": "...",
        "status": "ready" | "pending",
        "feature_importances": {feature: importance},
        "feature_names": [...],
        "sample_size": int,
        "feature_count": int,
        "explanation_method": "shap" | "model_feature_importances" | ...,
        "samples": { "feature_values": [[...]], "shap_values": [[...]] } | None,
        "source": "inline" | "minio"
    }

    Priority:
      1. Download the full payload from MinIO (contains per-sample values).
      2. Fall back to inline aggregated importances stored on ``run.metrics``.
      3. Return ``{"status": "pending"}`` if no explanation has been computed yet.
    """
    import json

    run = (
        await db.execute(select(ExperimentRun).where(ExperimentRun.id == run_id))
    ).scalar_one_or_none()
    if run is None:
        raise HTTPException(status_code=404, detail=f"Run {run_id!r} not found")

    inline_importances = (run.metrics or {}).get("shap_importances")
    inline_method = (run.metrics or {}).get("shap_method", "shap")
    inline_sample_size = (run.metrics or {}).get("shap_sample_size")

    # MinIO path — when available, it's richer than the inline summary.
    if run.artifacts_uri:
        try:
            from app.services.object_storage import download_object_bytes

            raw = download_object_bytes(run.artifacts_uri)
            if raw:
                try:
                    payload = json.loads(raw)
                    payload.setdefault("run_id", run_id)
                    payload["status"] = "ready"
                    payload["source"] = "minio"
                    payload["artifacts_uri"] = run.artifacts_uri
                    return payload
                except Exception as exc:
                    logger.warning(
                        "SHAP artifact at %s is not valid JSON: %s",
                        run.artifacts_uri,
                        exc,
                    )
        except Exception as exc:
            logger.warning("Could not fetch SHAP artifact for run %s: %s", run_id, exc)

    # Inline fallback — aggregated importances only, no per-sample data.
    if inline_importances:
        return {
            "run_id": run_id,
            "status": "ready",
            "feature_importances": inline_importances,
            "feature_names": list(inline_importances.keys()),
            "sample_size": inline_sample_size,
            "feature_count": len(inline_importances),
            "explanation_method": inline_method,
            "samples": None,
            "source": "inline",
        }

    # On-demand computation — lets the UI show a real payload the first time the
    # inspector opens a run whose explain task never ran.
    if compute:
        try:
            from app.services import shap_service
            payload = await shap_service.compute_shap_summary(run_id, db)
            payload.setdefault("run_id", run_id)
            payload["source"] = "computed"
            return payload
        except HTTPException:
            raise
        except Exception as exc:
            logger.warning("On-demand SHAP computation for run %s failed: %s", run_id, exc)
            return {
                "run_id": run_id,
                "status": "error",
                "error": str(exc),
                "source": None,
            }

    # No explanation computed yet.
    return {
        "run_id": run_id,
        "status": "pending",
        "source": None,
    }

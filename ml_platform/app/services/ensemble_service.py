"""Weighted multi-model deployments — creation, listing, and fan-out inference.

Threading note. `run_inference` in deploy_service does its model loading and
`predict` synchronously on the event loop; with one small model that is merely
sloppy, but fanning out to N members would multiply the block and freeze every
other request for the duration — the same failure mode as the SHAP incident.
So the whole fan-out here runs in one worker thread via `asyncio.to_thread`.

Members run *sequentially* inside that thread on purpose. XGBoost and LightGBM
are already internally multi-threaded, so running several at once oversubscribes
the cores and can be slower than doing them in turn. Parallelism is a change to
make against a measurement, not on the assumption that N models want N threads.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any

from fastapi import HTTPException
from sqlalchemy import select

from app.core.dl_registry import get_dl_trainer
from app.models.database import (
    AsyncSession,
    Dataset,
    DLTrainingTask,
    EnsembleDeployment,
    EnsembleMember,
    ModelingTask,
    TrainingTask,
)
from app.services.ensemble_fusion import fuse_classification, fuse_regression, normalise_weights
from app.services.resolver import load_model
from app.services.object_storage import restore_dataset_file, restore_model_bundle
from app.services.prediction_service import load_dataframe, predict_with_model

logger = logging.getLogger(__name__)

_MIN_MEMBERS = 2


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _serialize_member(m: EnsembleMember) -> dict[str, Any]:
    return {
        "id": m.id,
        "run_id": m.run_id,
        "domain_task_id": m.domain_task_id,
        "family": m.family,
        "model_type": m.model_type,
        "weight": m.weight,
    }


def _serialize(e: EnsembleDeployment) -> dict[str, Any]:
    return {
        "id": e.id,
        "name": e.name,
        "description": e.description,
        "strategy": e.strategy,
        "task_type": e.task_type,
        "status": e.status,
        "modeling_task_id": e.modeling_task_id,
        "request_count": e.request_count,
        "member_count": len(e.members or []),
        "members": [_serialize_member(m) for m in (e.members or [])],
        "created_at": e.created_at.isoformat() if e.created_at else None,
        "endpoints": {"predict": f"/inference/ensembles/{e.id}/predict"},
    }


# ---------------------------------------------------------------------------
# CRUD
# ---------------------------------------------------------------------------

async def create_ensemble(
    db: AsyncSession,
    modeling_task_id: str,
    *,
    name: str,
    description: str | None,
    members: list[dict[str, Any]],
    owner_username: str | None = None,
) -> dict[str, Any]:
    """Create a weighted ensemble from resolved run members."""
    if not name or not name.strip():
        raise HTTPException(status_code=400, detail="请填写部署名称")
    if len(members) < _MIN_MEMBERS:
        raise HTTPException(
            status_code=400,
            detail=f"融合部署至少需要 {_MIN_MEMBERS} 个成员，当前 {len(members)} 个",
        )

    task_stmt = select(ModelingTask).where(ModelingTask.id == modeling_task_id)
    if owner_username:
        task_stmt = task_stmt.where(ModelingTask.owner_username == owner_username)
    task = (await db.execute(task_stmt)).scalar_one_or_none()
    if task is None:
        raise HTTPException(status_code=404, detail="建模任务不存在")

    weights = normalise_weights([float(m.get("weight") or 0.0) for m in members])

    ensemble = EnsembleDeployment(
        owner_username=owner_username,
        modeling_task_id=modeling_task_id,
        name=name.strip(),
        description=(description or "").strip() or None,
        strategy="weighted_average",
        task_type=task.task_type or "classification",
        status="active",
    )
    db.add(ensemble)
    await db.flush()

    for spec, weight in zip(members, weights):
        domain_id = spec.get("domain_task_id")
        family = (spec.get("family") or "ml").lower()
        if not domain_id:
            raise HTTPException(status_code=400, detail="成员缺少 domain_task_id")

        # Verify the model exists and is usable before accepting the ensemble —
        # a member that only fails at call time is a worse outcome than a 400.
        if family == "dl":
            row = (await db.execute(
                select(DLTrainingTask).where(DLTrainingTask.id == domain_id)
            )).scalar_one_or_none()
        else:
            row = (await db.execute(
                select(TrainingTask).where(TrainingTask.id == domain_id)
            )).scalar_one_or_none()
        if row is None or not row.model_path:
            raise HTTPException(
                status_code=422,
                detail=f"成员模型不可用（{spec.get('model_type') or domain_id}）",
            )

        db.add(EnsembleMember(
            ensemble_id=ensemble.id,
            run_id=spec.get("run_id"),
            ml_task_id=domain_id if family != "dl" else None,
            dl_task_id=domain_id if family == "dl" else None,
            model_type=spec.get("model_type"),
            weight=weight,
        ))

    await db.flush()
    await db.refresh(ensemble)
    return _serialize(ensemble)


async def list_ensembles(
    db: AsyncSession,
    *,
    modeling_task_id: str | None = None,
    owner_username: str | None = None,
) -> list[dict[str, Any]]:
    stmt = select(EnsembleDeployment).order_by(EnsembleDeployment.created_at.desc())
    if modeling_task_id:
        stmt = stmt.where(EnsembleDeployment.modeling_task_id == modeling_task_id)
    if owner_username:
        stmt = stmt.where(EnsembleDeployment.owner_username == owner_username)
    rows = (await db.execute(stmt)).scalars().all()
    return [_serialize(e) for e in rows]


async def _get_or_404(
    db: AsyncSession, ensemble_id: str, owner_username: str | None
) -> EnsembleDeployment:
    stmt = select(EnsembleDeployment).where(EnsembleDeployment.id == ensemble_id)
    if owner_username:
        stmt = stmt.where(EnsembleDeployment.owner_username == owner_username)
    ensemble = (await db.execute(stmt)).scalar_one_or_none()
    if ensemble is None:
        raise HTTPException(status_code=404, detail="融合部署不存在")
    return ensemble


async def delete_ensemble(
    db: AsyncSession, ensemble_id: str, owner_username: str | None = None
) -> dict[str, str]:
    ensemble = await _get_or_404(db, ensemble_id, owner_username)
    await db.delete(ensemble)
    return {"message": "deleted"}


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def _predict_one_member_sync(spec: dict[str, Any], rows: list[dict]) -> dict[str, Any]:
    """Run one member and return its predictions, labels and probabilities.

    Synchronous by design — the caller runs the whole fan-out in one thread.
    """
    if spec["family"] == "dl":
        trainer = get_dl_trainer(spec["model_type"])
        meta = trainer.load_for_inference(spec["model_path"])
        artifact = meta.get("preprocessing_artifact")
        training_df = load_dataframe(spec["dataset_path"]) if artifact is None else None

        from app.services.dl_service import _prepare_dl_prediction_input

        X, _cols = _prepare_dl_prediction_input(meta, training_df, rows, spec["target_column"])
        preds, probas = trainer.predict(X, meta["task_type"])

        if meta["task_type"] != "classification":
            return {
                "predictions": preds.tolist(),
                "class_labels": [],
                "probabilities": None,
            }

        # A DL classifier predicts *class indices*, and its probability columns
        # are indexed too — while an ML member returns the original labels. Only
        # the preprocessing artifact's encoder can bridge the two. Without it
        # the member would be fused as if "0" were a class name, quietly mixing
        # one class into another, so refuse it instead.
        class_labels = [str(c) for c in (getattr(artifact, "class_labels", None) or [])]
        if not class_labels:
            raise ValueError(
                f"DL 成员 {spec.get('model_type')} 缺少标签编码器（旧版模型产物），"
                "无法与其它模型对齐类别，请重新训练该模型后再加入融合"
            )
        decoded = [str(v) for v in artifact.decode_predictions(preds)]
        return {
            "predictions": decoded,
            "class_labels": class_labels,
            "probabilities": probas.tolist() if probas is not None else None,
        }

    model = load_model(spec["model_path"])
    training_df = load_dataframe(spec["dataset_path"])
    return predict_with_model(
        model, training_df, rows, spec["target_column"], include_probabilities=True
    )


def _fan_out_sync(
    specs: list[dict[str, Any]], rows: list[dict], task_type: str
) -> dict[str, Any]:
    """Predict with every member, then fuse. Runs in a worker thread.

    A member that fails does not sink the request: the rest are refused only if
    fewer than two survive. The response names the failures and the weights
    actually used, because a quietly renormalised blend is a different model
    from the one the caller configured and they need to be able to see that.
    """
    results: list[dict[str, Any]] = []
    used: list[dict[str, Any]] = []
    failed: list[dict[str, str]] = []

    for spec in specs:
        try:
            results.append(_predict_one_member_sync(spec, rows))
            used.append(spec)
        except Exception as exc:  # noqa: BLE001 — one bad member must not 500 the blend
            logger.warning("Ensemble member %s failed: %s", spec.get("domain_task_id"), exc)
            failed.append({
                "domain_task_id": spec.get("domain_task_id"),
                "model_type": spec.get("model_type"),
                "error": str(exc),
            })

    if len(results) < _MIN_MEMBERS:
        raise ValueError(
            f"可用成员不足 {_MIN_MEMBERS} 个（成功 {len(results)}，失败 {len(failed)}）："
            + "；".join(f['error'] for f in failed)
        )

    weights = normalise_weights([s["weight"] for s in used])

    if task_type == "regression":
        fused = {
            "predictions": fuse_regression([r["predictions"] for r in results], weights),
            "class_labels": [],
            "probabilities": None,
        }
    else:
        fused = fuse_classification(results, weights)

    fused["members_used"] = [
        {
            "domain_task_id": s["domain_task_id"],
            "model_type": s["model_type"],
            "family": s["family"],
            "weight": round(w, 6),
        }
        for s, w in zip(used, weights)
    ]
    fused["members_failed"] = failed
    return fused


async def run_ensemble_inference(
    db: AsyncSession,
    ensemble_id: str,
    rows: list[dict],
    owner_username: str | None = None,
) -> dict[str, Any]:
    """Fan out to every member and return the fused prediction."""
    if not rows:
        raise HTTPException(status_code=400, detail="请求需要至少一行数据")

    ensemble = await _get_or_404(db, ensemble_id, owner_username)
    if ensemble.status != "active":
        raise HTTPException(status_code=400, detail="该融合部署已暂停")
    if len(ensemble.members or []) < _MIN_MEMBERS:
        raise HTTPException(status_code=422, detail="融合部署的成员不足，无法推理")

    # Resolve every path on the event loop (these are DB + object-store calls),
    # so the worker thread only does CPU work.
    specs: list[dict[str, Any]] = []
    for m in ensemble.members:
        if m.family == "dl":
            row = (await db.execute(
                select(DLTrainingTask).where(DLTrainingTask.id == m.dl_task_id)
            )).scalar_one_or_none()
        else:
            row = (await db.execute(
                select(TrainingTask).where(TrainingTask.id == m.ml_task_id)
            )).scalar_one_or_none()
        if row is None or not row.model_path:
            logger.warning("Ensemble %s: member %s has no model row", ensemble_id, m.id)
            continue

        model_path = restore_model_bundle(row.model_path)
        dataset = (await db.execute(
            select(Dataset).where(Dataset.id == row.dataset_id)
        )).scalar_one_or_none()
        dataset_path = (
            restore_dataset_file(dataset.id, dataset.file_path) if dataset else None
        )
        if model_path is None or dataset_path is None:
            logger.warning("Ensemble %s: member %s missing artifacts", ensemble_id, m.id)
            continue

        specs.append({
            "domain_task_id": m.domain_task_id,
            "family": m.family,
            "model_type": m.model_type or row.model_type,
            "weight": m.weight,
            "model_path": str(model_path),
            "dataset_path": str(dataset_path),
            "target_column": row.target_column,
        })

    if len(specs) < _MIN_MEMBERS:
        raise HTTPException(
            status_code=422,
            detail=f"可用成员不足 {_MIN_MEMBERS} 个，请检查成员模型文件是否仍存在",
        )

    try:
        fused = await asyncio.to_thread(_fan_out_sync, specs, rows, ensemble.task_type)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    ensemble.request_count = (ensemble.request_count or 0) + 1

    return {
        "deployment_id": ensemble.id,
        "strategy": ensemble.strategy,
        "task_type": ensemble.task_type,
        "input_rows": len(rows),
        **fused,
    }

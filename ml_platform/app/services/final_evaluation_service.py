"""Winner-only evaluation on the outer hold-out for V3 classic ML runs."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
from typing import Any
from uuid import uuid4

import joblib
from fastapi import HTTPException
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.model_artifact import is_tabular_artifact
from app.core.trainer import detect_task_type, get_trainer
from app.models.database import (
    Dataset,
    ExperimentRun,
    ModelingTask,
    PlatformExperiment,
    PlatformTask,
    TrainingTask,
)
from app.services.modeling_task_service import (
    FINAL_EVALUATION_VERSION,
    set_task_final_evaluation_state,
    task_final_evaluation_state,
    task_leaderboard,
)
from app.services.training_service import _prepare_data
from app.services.task_lifecycle_lock import task_lifecycle_guard
from app.utils.storage_paths import resolve_runtime_path


FINAL_SPLIT_SEED = 42
FINAL_CLAIM_STALE_AFTER = timedelta(minutes=30)
ACTIVE_RUN_STATUSES = ("PENDING", "QUEUED", "RUNNING")


async def finalize_task_winner(
    db: AsyncSession,
    modeling_task_id: str,
) -> dict[str, Any]:
    async with task_lifecycle_guard(modeling_task_id):
        return await _finalize_task_winner_locked(db, modeling_task_id)


async def _finalize_task_winner_locked(
    db: AsyncSession,
    modeling_task_id: str,
) -> dict[str, Any]:
    task = await _lock_modeling_task(db, modeling_task_id)
    current = task_final_evaluation_state(task)
    if current.get("state") == "FINALIZED":
        return {"status": "already_finalized", "final_evaluation": current}
    if current.get("state") == "EVALUATING":
        detail = (
            "最终确认 claim 已超时；为避免重复打开封存测试集，请先人工检查原请求。"
            if _claim_is_stale(current)
            else "最终模型正在确认，请稍后刷新。"
        )
        raise HTTPException(status_code=409, detail=detail)

    running_experiments = (
        await db.execute(
            select(func.count(PlatformExperiment.id)).where(
                PlatformExperiment.modeling_task_id == modeling_task_id,
                PlatformExperiment.status == "RUNNING",
            )
        )
    ).scalar_one()
    if running_experiments:
        raise HTTPException(
            status_code=409,
            detail=(
                f"仍有 {int(running_experiments)} 个实验批次运行中，"
                "全部结束后才能确认最终模型。"
            ),
        )

    active_count = (
        await db.execute(
            select(func.count(ExperimentRun.id))
            .select_from(ExperimentRun)
            .join(
                PlatformExperiment,
                PlatformExperiment.id == ExperimentRun.experiment_id,
            )
            .where(
                PlatformExperiment.modeling_task_id == modeling_task_id,
                ExperimentRun.status.in_(ACTIVE_RUN_STATUSES),
            )
        )
    ).scalar_one()
    if active_count:
        raise HTTPException(
            status_code=409,
            detail=f"仍有 {int(active_count)} 个 Run 运行中，全部结束后才能确认最终模型。",
        )

    winner, run = await _validate_finalization_winner(db, modeling_task_id)
    now = datetime.now(timezone.utc)
    claim_id = str(uuid4())
    claim = {
        "state": "EVALUATING",
        "version": FINAL_EVALUATION_VERSION,
        "claim_id": claim_id,
        "winner_run_id": run.id,
        "requested_at": now.isoformat(),
        "attempt": int(current.get("attempt") or 0) + 1,
    }
    set_task_final_evaluation_state(task, claim)
    await db.commit()

    try:
        result = await evaluate_task_winner(db, modeling_task_id)
        if result.get("status") not in {"evaluated", "skipped"}:
            raise HTTPException(
                status_code=422,
                detail=f"当前冠军不支持最终评估：{result.get('reason') or result.get('status')}",
            )
        if result.get("status") == "skipped" and result.get("reason") != "already_evaluated":
            raise HTTPException(
                status_code=422,
                detail=f"当前冠军不能最终确认：{result.get('reason')}",
            )

        locked_task = await _lock_modeling_task(db, modeling_task_id)
        locked_state = task_final_evaluation_state(locked_task)
        _assert_claim_owner(locked_state, claim_id)
        locked_run = (
            await db.execute(
                select(ExperimentRun)
                .where(ExperimentRun.id == run.id)
                .with_for_update()
            )
        ).scalar_one()
        run_audit = (locked_run.search_meta or {}).get("final_evaluation") or {}
        evaluation_id = result.get("evaluation_id") or run_audit.get("evaluation_id")
        final_metrics = {
            key: value
            for key, value in (locked_run.metrics or {}).items()
            if key.startswith("final_test_")
        }
        finalized = {
            **locked_state,
            "state": "FINALIZED",
            "winner_run_id": winner["run_id"],
            "evaluation_id": evaluation_id,
            "final_metrics": final_metrics,
            "finalized_at": datetime.now(timezone.utc).isoformat(),
        }
        set_task_final_evaluation_state(locked_task, finalized)
        await db.commit()
        return {"status": "finalized", "final_evaluation": finalized}
    except Exception as exc:
        await db.rollback()
        await _persist_failed_claim(db, modeling_task_id, claim_id, exc)
        raise


async def _lock_modeling_task(db: AsyncSession, task_id: str) -> ModelingTask:
    task = (
        await db.execute(
            select(ModelingTask)
            .where(ModelingTask.id == task_id)
            .with_for_update()
        )
    ).scalar_one_or_none()
    if task is None:
        raise HTTPException(status_code=404, detail=f"ModelingTask {task_id!r} not found")
    return task


async def _validate_finalization_winner(
    db: AsyncSession,
    modeling_task_id: str,
) -> tuple[dict[str, Any], ExperimentRun]:
    board = await task_leaderboard(db, modeling_task_id, top_k=1)
    if not board:
        raise HTTPException(status_code=422, detail="没有可确认的成功 Run。")
    winner = board[0]
    if winner.get("family") != "ml":
        raise HTTPException(status_code=422, detail="深度学习 Run 暂不支持 sealed final 评估。")
    run = (
        await db.execute(
            select(ExperimentRun).where(ExperimentRun.id == winner["run_id"])
        )
    ).scalar_one()
    if (run.search_meta or {}).get("evaluation_mode") != "selection":
        raise HTTPException(status_code=422, detail="当前冠军不是 selection-only Run，不能最终确认。")
    return winner, run


def _claim_is_stale(state: dict[str, Any]) -> bool:
    try:
        requested_at = datetime.fromisoformat(str(state["requested_at"]))
        if requested_at.tzinfo is None:
            requested_at = requested_at.replace(tzinfo=timezone.utc)
    except (KeyError, TypeError, ValueError):
        return True
    return datetime.now(timezone.utc) - requested_at >= FINAL_CLAIM_STALE_AFTER


def _assert_claim_owner(state: dict[str, Any], claim_id: str) -> None:
    if state.get("state") != "EVALUATING" or state.get("claim_id") != claim_id:
        raise HTTPException(status_code=409, detail="最终确认 claim 已被其他请求接管。")


async def _persist_failed_claim(
    db: AsyncSession,
    task_id: str,
    claim_id: str,
    exc: Exception,
) -> None:
    task = await _lock_modeling_task(db, task_id)
    state = task_final_evaluation_state(task)
    if state.get("claim_id") != claim_id:
        await db.rollback()
        return
    failed = {
        **state,
        "state": "FAILED",
        "error": str(exc)[:500] or exc.__class__.__name__,
        "failed_at": datetime.now(timezone.utc).isoformat(),
    }
    set_task_final_evaluation_state(task, failed)
    await db.commit()


async def evaluate_task_winner(
    db: AsyncSession,
    modeling_task_id: str,
) -> dict[str, Any]:
    board = await task_leaderboard(db, modeling_task_id, top_k=1)
    if not board:
        return {"status": "skipped", "reason": "no_successful_run"}

    winner = board[0]
    if winner.get("family") != "ml":
        return {
            "status": "unsupported",
            "reason": "dl_winner_has_no_sealed_cv_flow",
            "run_id": winner["run_id"],
        }

    run = (
        await db.execute(
            select(ExperimentRun).where(ExperimentRun.id == winner["run_id"])
        )
    ).scalar_one()
    if (run.search_meta or {}).get("evaluation_mode") != "selection":
        return {
            "status": "skipped",
            "reason": "winner_not_selection_only",
            "run_id": run.id,
        }
    platform_task = (
        await db.execute(
            select(PlatformTask).where(PlatformTask.id == run.task_id)
        )
    ).scalar_one_or_none()
    domain_task_id = _training_task_id(platform_task)
    if domain_task_id is None:
        return {"status": "skipped", "reason": "missing_training_task", "run_id": run.id}

    training_task = (
        await db.execute(
            select(TrainingTask).where(TrainingTask.id == domain_task_id)
        )
    ).scalar_one_or_none()
    if training_task is None or not training_task.model_path:
        return {"status": "skipped", "reason": "missing_model_artifact", "run_id": run.id}
    dataset = (
        await db.execute(select(Dataset).where(Dataset.id == training_task.dataset_id))
    ).scalar_one_or_none()
    if dataset is None:
        return {"status": "skipped", "reason": "missing_dataset", "run_id": run.id}

    model_path = resolve_runtime_path(training_task.model_path)
    dataset_path = resolve_runtime_path(dataset.file_path)
    test_size = float(training_task.test_size or 0.2)
    eval_metrics = training_task.eval_metrics or (
        ["rmse", "mae", "r2"]
        if detect_task_type(training_task.model_type) == "regression"
        else ["accuracy"]
    )
    evaluation_id = await asyncio.to_thread(
        _evaluation_id,
        run_id=run.id,
        dataset_id=dataset.id,
        dataset_path=dataset_path,
        model_path=model_path,
        target_column=training_task.target_column,
        test_size=test_size,
        eval_metrics=eval_metrics,
    )
    existing = (run.search_meta or {}).get("final_evaluation") or {}
    if existing.get("evaluation_id") == evaluation_id:
        return {
            "status": "skipped",
            "reason": "already_evaluated",
            "run_id": run.id,
            "evaluation_id": evaluation_id,
        }

    computed = await asyncio.to_thread(
        _evaluate_artifact,
        dataset_path=dataset_path,
        model_path=model_path,
        target_column=training_task.target_column,
        test_size=test_size,
        model_type=training_task.model_type,
        eval_metrics=eval_metrics,
    )

    # Re-read JSON columns under a row lock after the long-running file/model
    # work so a concurrent writer cannot be overwritten by this merge.
    await db.refresh(
        run,
        attribute_names=["metrics", "search_meta"],
        with_for_update=True,
    )
    existing = (run.search_meta or {}).get("final_evaluation") or {}
    if existing.get("evaluation_id") == evaluation_id:
        return {
            "status": "skipped",
            "reason": "already_evaluated",
            "run_id": run.id,
            "evaluation_id": evaluation_id,
        }

    metrics = dict(run.metrics or {})
    metrics.update({f"final_test_{key}": value for key, value in computed.items()})
    run.metrics = metrics
    search_meta = dict(run.search_meta or {})
    search_meta["final_evaluation"] = {
        "evaluation_id": evaluation_id,
        "version": FINAL_EVALUATION_VERSION,
        "split_seed": FINAL_SPLIT_SEED,
        "test_size": test_size,
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
    }
    run.search_meta = search_meta
    await db.flush()
    return {
        "status": "evaluated",
        "run_id": run.id,
        "evaluation_id": evaluation_id,
        "metrics": {f"final_test_{key}": value for key, value in computed.items()},
    }


def _training_task_id(platform_task: PlatformTask | None) -> str | None:
    if platform_task is None or not platform_task.payload_ref:
        return None
    kind, _, domain_id = platform_task.payload_ref.partition(":")
    return domain_id if kind == "train" and domain_id else None


def _evaluate_artifact(
    *,
    dataset_path: Path,
    model_path: Path,
    target_column: str,
    test_size: float,
    model_type: str,
    eval_metrics: list[str],
) -> dict[str, Any]:
    is_regression = detect_task_type(model_type) == "regression"
    _, X_holdout, _, y_holdout = _prepare_data(
        str(dataset_path), target_column, test_size, is_regression
    )
    model = joblib.load(model_path)
    if is_tabular_artifact(model):
        metric_targets = model.encode_target(y_holdout)
        predictions = model.predict_encoded(X_holdout)
    else:
        metric_targets = y_holdout
        predictions = model.predict(X_holdout)
    trainer = get_trainer(model_type)
    if is_regression:
        return trainer._compute_regression_metrics(
            metric_targets, predictions, eval_metrics
        )

    probabilities = None
    try:
        probabilities = model.predict_proba(X_holdout)
    except (AttributeError, TypeError, ValueError):
        pass
    return trainer._compute_metrics(
        metric_targets, predictions, probabilities, eval_metrics
    )


def _evaluation_id(
    *,
    run_id: str,
    dataset_id: str,
    dataset_path: Path,
    model_path: Path,
    target_column: str,
    test_size: float,
    eval_metrics: list[str],
) -> str:
    payload = {
        "version": FINAL_EVALUATION_VERSION,
        "split_seed": FINAL_SPLIT_SEED,
        "run_id": run_id,
        "dataset_id": dataset_id,
        "dataset_file": _file_identity(dataset_path),
        "model_file": _file_identity(model_path),
        "target_column": target_column,
        "test_size": test_size,
        "eval_metrics": sorted(set(eval_metrics)),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()[:24]


def _file_identity(path: Path) -> dict[str, int | str]:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {
        "size": path.stat().st_size,
        "sha256": digest.hexdigest(),
    }

"""Tests for winner-only sealed hold-out evaluation."""

import joblib
import asyncio
import os
import pandas as pd
import pytest
import shutil
import threading
from datetime import datetime, timedelta, timezone
from fastapi import HTTPException
from sqlalchemy import select
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression

from app.core.evaluation_metrics import resolve_objective_metrics
from app.core.model_artifact import fit_tabular_artifact
from app.models.database import (
    Dataset,
    ExperimentRun,
    ModelingTask,
    PlatformExperiment,
    PlatformTask,
    TrainingTask,
)
from app.services import final_evaluation_service as svc
from app.services.modeling_task_service import serialize_modeling_task
from app.services.training_service import _prepare_data


async def _seed_sealed_task(db, tmp_path, *, boolean_target=False):
    labels = [False, True] * 10 if boolean_target else [0, 1] * 10
    frame = pd.DataFrame(
        {
            "value": list(range(20)),
            "group": ["a", "b"] * 10,
            "label": labels,
        }
    )
    dataset_path = tmp_path / (
        "sealed-boolean.csv" if boolean_target else "sealed.csv"
    )
    frame.to_csv(dataset_path, index=False)
    dataset = Dataset(
        name="sealed.csv",
        file_path=str(dataset_path),
        file_size=dataset_path.stat().st_size,
        row_count=len(frame),
    )
    db.add(dataset)
    await db.flush()
    modeling_task = ModelingTask(
        name="sealed-task",
        dataset_id=dataset.id,
        dataset_name=dataset.name,
        target_column="label",
        task_type="classification",
        objective_metric="accuracy",
        objective_direction="max",
    )
    db.add(modeling_task)
    await db.flush()
    experiment = PlatformExperiment(
        modeling_task_id=modeling_task.id,
        name="batch",
        strategy_type="grid_search",
        dataset_id=dataset.id,
        objective_metric="accuracy",
        objective_direction="max",
    )
    db.add(experiment)
    await db.flush()

    runs = []
    for trial_no, selection_value in [(1, 0.91), (2, 0.80)]:
        training_task = TrainingTask(
            dataset_id=dataset.id,
            model_type="logistic_regression",
            target_column="label",
            hyperparameters={},
            test_size=0.2,
            cv_folds=2,
            eval_metrics=["accuracy", "f1"],
            status="SUCCESS",
        )
        db.add(training_task)
        await db.flush()
        platform_task = PlatformTask(
            kind="train",
            status="SUCCESS",
            payload_ref=f"train:{training_task.id}",
        )
        db.add(platform_task)
        await db.flush()
        run = ExperimentRun(
            experiment_id=experiment.id,
            task_id=platform_task.id,
            status="SUCCESS",
            trial_no=trial_no,
            params={"family": "ml", "model_type": "logistic_regression"},
            metrics={"selection_cv_mean_accuracy": selection_value},
            search_meta={"evaluation_mode": "selection"},
        )
        db.add(run)
        await db.flush()
        runs.append((run, training_task))

    X_train, _, y_train, _ = _prepare_data(
        str(dataset_path), "label", 0.2, False
    )
    artifact = fit_tabular_artifact(
        LogisticRegression(random_state=42),
        X_train,
        y_train,
        task_kind="classification",
    )
    model_path = tmp_path / "winner.joblib"
    joblib.dump(artifact, model_path)
    runs[0][1].model_path = str(model_path)
    await db.flush()
    return modeling_task, runs


async def _seed_sealed_regression_task(db, tmp_path):
    frame = pd.DataFrame(
        {
            "value": list(range(20)),
            "group": ["a", "b"] * 10,
            "target": [float(value * 3 + 1) for value in range(20)],
        }
    )
    dataset_path = tmp_path / "sealed-regression.csv"
    frame.to_csv(dataset_path, index=False)
    dataset = Dataset(
        name=dataset_path.name,
        file_path=str(dataset_path),
        file_size=dataset_path.stat().st_size,
        row_count=len(frame),
    )
    db.add(dataset)
    await db.flush()
    modeling_task = ModelingTask(
        name="sealed-regression-task",
        dataset_id=dataset.id,
        dataset_name=dataset.name,
        target_column="target",
        task_type="regression",
        objective_metric="rmse",
        objective_direction="min",
    )
    db.add(modeling_task)
    await db.flush()
    experiment = PlatformExperiment(
        modeling_task_id=modeling_task.id,
        name="regression-batch",
        strategy_type="grid_search",
        dataset_id=dataset.id,
        objective_metric="rmse",
        objective_direction="min",
    )
    db.add(experiment)
    await db.flush()
    training_task = TrainingTask(
        dataset_id=dataset.id,
        model_type="linear_regression",
        target_column="target",
        hyperparameters={},
        test_size=0.2,
        cv_folds=2,
        eval_metrics=["rmse", "mae", "r2"],
        status="SUCCESS",
    )
    db.add(training_task)
    await db.flush()
    platform_task = PlatformTask(
        kind="train",
        status="SUCCESS",
        payload_ref=f"train:{training_task.id}",
    )
    db.add(platform_task)
    await db.flush()
    run = ExperimentRun(
        experiment_id=experiment.id,
        task_id=platform_task.id,
        status="SUCCESS",
        trial_no=1,
        params={"family": "ml", "model_type": "linear_regression"},
        metrics={"selection_cv_mean_rmse": 0.2},
        search_meta={"evaluation_mode": "selection"},
    )
    db.add(run)
    await db.flush()

    X_train, _, y_train, _ = _prepare_data(
        str(dataset_path), "target", 0.2, True
    )
    artifact = fit_tabular_artifact(
        LinearRegression(), X_train, y_train, task_kind="regression"
    )
    model_path = tmp_path / "regression-winner.joblib"
    joblib.dump(artifact, model_path)
    training_task.model_path = str(model_path)
    await db.flush()
    return modeling_task, run


async def test_evaluate_task_winner_writes_only_final_metrics_to_selection_winner(
    db, tmp_path
):
    task, runs = await _seed_sealed_task(db, tmp_path)

    result = await svc.evaluate_task_winner(db, task.id)
    await db.refresh(runs[0][0])
    await db.refresh(runs[1][0])

    assert result["status"] == "evaluated"
    assert result["run_id"] == runs[0][0].id
    assert "final_test_accuracy" in runs[0][0].metrics
    assert "final_test_f1" in runs[0][0].metrics
    assert "accuracy" not in runs[0][0].metrics
    assert "final_test_accuracy" not in runs[1][0].metrics
    resolved = resolve_objective_metrics(runs[0][0].metrics, "accuracy")
    assert resolved.selection_value == 0.91


async def test_evaluate_task_winner_is_idempotent_for_same_version(db, tmp_path):
    task, runs = await _seed_sealed_task(db, tmp_path)

    first = await svc.evaluate_task_winner(db, task.id)
    dataset_path = tmp_path / "sealed.csv"
    stat = dataset_path.stat()
    os.utime(dataset_path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))
    second = await svc.evaluate_task_winner(db, task.id)
    await db.refresh(runs[0][0])

    assert first["status"] == "evaluated"
    assert second["status"] == "skipped"
    assert second["evaluation_id"] == first["evaluation_id"]
    audit = runs[0][0].search_meta["final_evaluation"]
    assert audit["split_seed"] == 42
    assert audit["version"] == 1


async def test_evaluation_id_uses_content_not_storage_path(db, tmp_path):
    task, runs = await _seed_sealed_task(db, tmp_path)
    training_task = runs[0][1]
    dataset = await db.get(Dataset, training_task.dataset_id)

    first = await svc.evaluate_task_winner(db, task.id)
    relocated_dataset = tmp_path / "relocated-dataset.csv"
    relocated_model = tmp_path / "relocated-model.joblib"
    shutil.copy2(dataset.file_path, relocated_dataset)
    shutil.copy2(training_task.model_path, relocated_model)
    dataset.file_path = str(relocated_dataset)
    training_task.model_path = str(relocated_model)
    await db.flush()

    second = await svc.evaluate_task_winner(db, task.id)

    assert second["status"] == "skipped"
    assert second["reason"] == "already_evaluated"
    assert second["evaluation_id"] == first["evaluation_id"]


async def test_evaluation_id_changes_when_requested_metrics_change(db, tmp_path):
    task, runs = await _seed_sealed_task(db, tmp_path)
    training_task = runs[0][1]

    first = await svc.evaluate_task_winner(db, task.id)
    training_task.eval_metrics = ["accuracy", "f1", "roc_auc"]
    await db.flush()
    second = await svc.evaluate_task_winner(db, task.id)

    assert second["status"] == "evaluated"
    assert second["evaluation_id"] != first["evaluation_id"]
    assert "final_test_roc_auc" in second["metrics"]


async def test_evaluate_task_winner_skips_legacy_standard_run(db, tmp_path):
    task, runs = await _seed_sealed_task(db, tmp_path)
    runs[0][0].search_meta = {"evaluation_mode": "standard"}
    await db.flush()

    result = await svc.evaluate_task_winner(db, task.id)
    await db.refresh(runs[0][0])

    assert result == {
        "status": "skipped",
        "reason": "winner_not_selection_only",
        "run_id": runs[0][0].id,
    }
    assert "final_test_accuracy" not in runs[0][0].metrics


async def test_evaluate_task_winner_offloads_file_and_model_work(db, tmp_path, monkeypatch):
    task, _ = await _seed_sealed_task(db, tmp_path)
    event_loop_thread = threading.get_ident()
    worker_threads = []
    original = svc._evaluate_artifact

    def recording_evaluate_artifact(**kwargs):
        worker_threads.append(threading.get_ident())
        return original(**kwargs)

    monkeypatch.setattr(svc, "_evaluate_artifact", recording_evaluate_artifact)

    result = await svc.evaluate_task_winner(db, task.id)

    assert result["status"] == "evaluated"
    assert worker_threads
    assert worker_threads[0] != event_loop_thread


async def test_evaluate_task_winner_supports_regression_metrics(db, tmp_path):
    task, run = await _seed_sealed_regression_task(db, tmp_path)

    result = await svc.evaluate_task_winner(db, task.id)
    await db.refresh(run)

    assert result["status"] == "evaluated"
    assert set(result["metrics"]) == {
        "final_test_rmse",
        "final_test_mae",
        "final_test_r2",
    }
    assert run.metrics["selection_cv_mean_rmse"] == 0.2


async def test_evaluate_task_winner_supports_boolean_classification_targets(
    db, tmp_path
):
    task, run_pairs = await _seed_sealed_task(
        db, tmp_path, boolean_target=True
    )

    result = await svc.evaluate_task_winner(db, task.id)

    assert result["status"] == "evaluated"
    assert result["metrics"]["final_test_accuracy"] is not None
    assert result["metrics"]["final_test_f1"] is not None


async def test_evaluate_task_winner_refreshes_run_under_lock_before_json_merge(
    db, tmp_path, monkeypatch
):
    task, run_pairs = await _seed_sealed_task(db, tmp_path)
    winner_run = run_pairs[0][0]
    events = []
    original_evaluate = svc._evaluate_artifact
    original_refresh = db.refresh

    def recording_evaluate(**kwargs):
        events.append("evaluate")
        return original_evaluate(**kwargs)

    async def recording_refresh(instance, *args, **kwargs):
        if instance.id == winner_run.id and kwargs.get("with_for_update"):
            events.append("locked_refresh")
        return await original_refresh(instance, *args, **kwargs)

    monkeypatch.setattr(svc, "_evaluate_artifact", recording_evaluate)
    monkeypatch.setattr(db, "refresh", recording_refresh)

    result = await svc.evaluate_task_winner(db, task.id)

    assert result["status"] == "evaluated"
    assert events == ["evaluate", "locked_refresh"]


def test_modeling_task_serializes_open_final_evaluation_without_private_config():
    task = ModelingTask(
        name="open-task",
        config={"user_setting": True},
    )

    payload = serialize_modeling_task(task)

    assert payload["config"] == {"user_setting": True}
    assert payload["final_evaluation"] == {"state": "OPEN", "version": 1}


async def test_finalize_task_winner_is_explicit_and_task_level_idempotent(
    db, tmp_path, monkeypatch
):
    task, runs = await _seed_sealed_task(db, tmp_path)

    first = await svc.finalize_task_winner(db, task.id)
    await db.refresh(task)
    state = task.config["_final_evaluation"]

    assert first["status"] == "finalized"
    assert state["state"] == "FINALIZED"
    assert state["winner_run_id"] == runs[0][0].id
    assert state["final_metrics"]["final_test_accuracy"] is not None

    async def must_not_evaluate_again(*args, **kwargs):
        raise AssertionError("finalized task opened the hold-out again")

    monkeypatch.setattr(svc, "evaluate_task_winner", must_not_evaluate_again)
    second = await svc.finalize_task_winner(db, task.id)

    assert second["status"] == "already_finalized"
    assert second["final_evaluation"]["evaluation_id"] == state["evaluation_id"]


async def test_finalize_task_winner_rejects_active_runs(db, tmp_path):
    task, runs = await _seed_sealed_task(db, tmp_path)
    runs[1][0].status = "RUNNING"
    await db.flush()

    with pytest.raises(HTTPException) as exc:
        await svc.finalize_task_winner(db, task.id)

    assert exc.value.status_code == 409
    assert "运行中" in exc.value.detail
    await db.refresh(task)
    assert "_final_evaluation" not in (task.config or {})


async def test_finalize_task_winner_rejects_live_claim(db, tmp_path):
    task, _ = await _seed_sealed_task(db, tmp_path)
    task.config = {
        "_final_evaluation": {
            "state": "EVALUATING",
            "version": 1,
            "claim_id": "live-claim",
            "requested_at": datetime.now(timezone.utc).isoformat(),
        }
    }
    await db.flush()

    with pytest.raises(HTTPException) as exc:
        await svc.finalize_task_winner(db, task.id)

    assert exc.value.status_code == 409
    assert "正在确认" in exc.value.detail


async def test_finalize_task_winner_never_takes_over_stale_claim(db, tmp_path):
    task, runs = await _seed_sealed_task(db, tmp_path)
    task.config = {
        "_final_evaluation": {
            "state": "EVALUATING",
            "version": 1,
            "claim_id": "stale-claim",
            "requested_at": (
                datetime.now(timezone.utc) - timedelta(minutes=31)
            ).isoformat(),
        }
    }
    await db.flush()

    with pytest.raises(HTTPException) as exc:
        await svc.finalize_task_winner(db, task.id)

    assert exc.value.status_code == 409
    assert "人工检查" in exc.value.detail
    await db.refresh(task)
    assert task.config["_final_evaluation"]["claim_id"] == "stale-claim"


async def test_finalize_task_winner_rejects_running_experiment_between_trials(
    db, tmp_path
):
    task, _ = await _seed_sealed_task(db, tmp_path)
    experiment = (
        await db.execute(
            select(PlatformExperiment).where(
                PlatformExperiment.modeling_task_id == task.id
            )
        )
    ).scalar_one()
    experiment.status = "RUNNING"
    await db.flush()

    with pytest.raises(HTTPException) as exc:
        await svc.finalize_task_winner(db, task.id)

    assert exc.value.status_code == 409
    assert "实验批次" in exc.value.detail


async def test_concurrent_finalization_waits_for_task_lifecycle_lock(
    session_factory, tmp_path, monkeypatch
):
    async with session_factory() as seed_db:
        task, runs = await _seed_sealed_task(seed_db, tmp_path)
        await seed_db.commit()
        task_id = task.id
        winner_id = runs[0][0].id

    evaluation_started = asyncio.Event()
    release_evaluation = asyncio.Event()

    async def blocking_evaluate(db, modeling_task_id):
        evaluation_started.set()
        await release_evaluation.wait()
        return {
            "status": "evaluated",
            "run_id": winner_id,
            "evaluation_id": "concurrent-evaluation",
            "metrics": {"final_test_accuracy": 0.9},
        }

    monkeypatch.setattr(svc, "evaluate_task_winner", blocking_evaluate)

    async def run_finalize():
        async with session_factory() as session:
            return await svc.finalize_task_winner(session, task_id)

    first = asyncio.create_task(run_finalize())
    await evaluation_started.wait()
    second = asyncio.create_task(run_finalize())
    await asyncio.sleep(0.02)
    second_was_waiting = not second.done()
    release_evaluation.set()
    first_result, second_result = await asyncio.gather(
        first, second, return_exceptions=True
    )
    assert second_was_waiting
    assert first_result["status"] == "finalized"
    assert second_result["status"] == "already_finalized"


async def test_finalize_task_winner_persists_failed_claim_for_retry(
    db, tmp_path, monkeypatch
):
    task, _ = await _seed_sealed_task(db, tmp_path)

    async def fail_evaluation(*args, **kwargs):
        raise RuntimeError("artifact read failed")

    monkeypatch.setattr(svc, "evaluate_task_winner", fail_evaluation)

    with pytest.raises(RuntimeError, match="artifact read failed"):
        await svc.finalize_task_winner(db, task.id)

    await db.refresh(task)
    state = task.config["_final_evaluation"]
    assert state["state"] == "FAILED"
    assert state["error"] == "artifact read failed"


@pytest.mark.parametrize(
    ("winner_kind", "evaluation_mode", "detail_fragment"),
    [
        ("dl_train", "selection", "深度学习"),
        ("train", "standard", "selection-only"),
    ],
)
async def test_finalize_task_winner_rejects_unsupported_winner_before_evaluation(
    db, tmp_path, monkeypatch, winner_kind, evaluation_mode, detail_fragment
):
    task, runs = await _seed_sealed_task(db, tmp_path)
    winner_run = runs[0][0]
    winner_run.search_meta = {"evaluation_mode": evaluation_mode}
    platform_task = await db.get(PlatformTask, winner_run.task_id)
    _, _, domain_id = platform_task.payload_ref.partition(":")
    platform_task.payload_ref = f"{winner_kind}:{domain_id}"
    await db.flush()

    async def must_not_evaluate(*args, **kwargs):
        raise AssertionError("unsupported winner reached holdout evaluation")

    monkeypatch.setattr(svc, "evaluate_task_winner", must_not_evaluate)

    with pytest.raises(HTTPException) as exc:
        await svc.finalize_task_winner(db, task.id)

    assert exc.value.status_code == 422
    assert detail_fragment in exc.value.detail


async def test_failed_task_claim_recovers_from_existing_run_evaluation(
    db, tmp_path, monkeypatch
):
    task, runs = await _seed_sealed_task(db, tmp_path)
    run_result = await svc.evaluate_task_winner(db, task.id)
    task.config = {
        "_final_evaluation": {
            "state": "FAILED",
            "version": 1,
            "claim_id": "failed-claim",
            "attempt": 1,
            "error": "task audit write failed",
        }
    }
    await db.commit()
    original = svc.evaluate_task_winner
    calls = []

    async def recording_evaluate(*args, **kwargs):
        calls.append("evaluate")
        return await original(*args, **kwargs)

    monkeypatch.setattr(svc, "evaluate_task_winner", recording_evaluate)

    result = await svc.finalize_task_winner(db, task.id)

    assert calls == ["evaluate"]
    assert result["status"] == "finalized"
    assert result["final_evaluation"]["evaluation_id"] == run_result["evaluation_id"]
    assert result["final_evaluation"]["attempt"] == 2

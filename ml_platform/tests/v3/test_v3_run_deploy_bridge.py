"""Tests for the V3 run→deployment bridge.

Covers modeling_task_service._resolve_domain_ref and deploy_run(), which let
the unified workflow's 部署 step deploy an ExperimentRun's model by reusing the
existing per-domain-task deployment services (ML TrainingTask / DL task).

The rows are constructed directly (no real tuning engine) so the test is fast
and deterministic; a real joblib model file is written so create_deployment's
on-disk check passes.
"""

from __future__ import annotations

import joblib
import pytest
from fastapi import HTTPException
from sklearn.linear_model import LogisticRegression

from app.models.database import (
    Dataset,
    ExperimentRun,
    ModelingTask,
    PlatformExperiment,
    PlatformTask,
    TrainingTask,
)
from app.services import modeling_task_service as svc


def test_resolve_domain_ref_parses_family():
    assert svc._resolve_domain_ref("train:abc123") == ("abc123", "ml")
    assert svc._resolve_domain_ref("dl_train:xyz789") == ("xyz789", "dl")
    assert svc._resolve_domain_ref(None) == (None, None)
    assert svc._resolve_domain_ref("garbage-no-colon") == (None, None)


async def _make_ml_run(db, tmp_path, *, run_status="SUCCESS"):
    """Build task→experiment→run→platform_task→TrainingTask(+model file)."""
    ds = Dataset(name="iris.csv", file_path="/tmp/iris.csv", file_size=1, row_count=150)
    db.add(ds)
    await db.flush()

    task = ModelingTask(
        name="t", dataset_id=ds.id, dataset_name=ds.name,
        target_column="species", task_type="classification",
        objective_metric="accuracy", objective_direction="max",
    )
    db.add(task)
    await db.flush()

    exp = PlatformExperiment(
        name="baseline", modeling_task_id=task.id, dataset_id=ds.id,
        strategy_type="baseline", status="COMPLETED",
        selected_models=["logistic_regression"],
    )
    db.add(exp)
    await db.flush()

    # Real model file on disk (absolute path passes resolve_runtime_path check)
    model = LogisticRegression().fit([[0, 0], [1, 1]], [0, 1])
    model_file = tmp_path / "domain-model.joblib"
    joblib.dump(model, model_file)

    domain = TrainingTask(
        name="domain", dataset_id=ds.id, model_type="logistic_regression",
        target_column="species", status="SUCCESS", model_path=str(model_file),
    )
    db.add(domain)
    await db.flush()

    ptask = PlatformTask(kind="train", status="SUCCESS", payload_ref=f"train:{domain.id}")
    db.add(ptask)
    await db.flush()

    run = ExperimentRun(
        experiment_id=exp.id, task_id=ptask.id, status=run_status,
        params={"model_type": "logistic_regression"}, metrics={"accuracy": 0.96},
        trial_no=1,
    )
    db.add(run)
    await db.flush()
    return task, run


async def test_deploy_run_ml_success(db, tmp_path):
    task, run = await _make_ml_run(db, tmp_path)

    result = await svc.deploy_run(db, task.id, run.id, name="iris-prod")

    assert result["family"] == "ml"
    assert result["deployment_id"]
    assert "predict" in result["endpoints"]
    assert result["endpoints"]["predict"] == f"/inference/{result['deployment_id']}/predict"
    assert result["status"] == "active"


async def test_deploy_run_rejects_non_success(db, tmp_path):
    task, run = await _make_ml_run(db, tmp_path, run_status="RUNNING")

    with pytest.raises(HTTPException) as exc:
        await svc.deploy_run(db, task.id, run.id, name="nope")
    assert exc.value.status_code == 422


async def test_deploy_run_rejects_foreign_run(db, tmp_path):
    task, run = await _make_ml_run(db, tmp_path)
    # A second, unrelated task must not be able to deploy the first task's run.
    other = ModelingTask(name="other", task_type="classification")
    db.add(other)
    await db.flush()

    with pytest.raises(HTTPException) as exc:
        await svc.deploy_run(db, other.id, run.id, name="nope")
    assert exc.value.status_code == 404

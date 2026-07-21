from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import joblib
import pandas as pd
import pytest
from fastapi import HTTPException

from app.services import (
    deploy_service,
    final_evaluation_service,
    modeling_task_service,
    prediction_service,
    resolver,
    viz_service,
)


class _ScalarResult:
    def __init__(self, value):
        self.value = value

    def scalar_one_or_none(self):
        return self.value


class _SequenceDb:
    def __init__(self, *values):
        self.values = iter(values)
        self.added = []

    async def execute(self, _statement):
        return _ScalarResult(next(self.values))

    def add(self, value):
        self.added.append(value)

    async def flush(self):
        for value in self.added:
            if getattr(value, "id", None) is None:
                value.id = "generated-job"


def test_load_model_restores_missing_local_artifact(monkeypatch, tmp_path):
    model_path = tmp_path / "storage" / "models" / "task.joblib"
    expected = {"model": "from-minio"}

    def restore_model_bundle(path):
        assert path == "storage/models/task.joblib"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(expected, model_path)
        return model_path

    monkeypatch.setattr(resolver, "resolve_runtime_path", lambda _path: model_path)
    monkeypatch.setattr(resolver, "restore_model_bundle", restore_model_bundle)

    assert resolver.load_model("storage/models/task.joblib") == expected


@pytest.mark.asyncio
async def test_learning_curve_restores_metrics_from_object_storage(monkeypatch, tmp_path):
    settings = SimpleNamespace(storage_logs=tmp_path / "storage" / "logs")

    async def candidates(_task_id, _db):
        return ["legacy-task"]

    def restore_file(path, keys):
        assert keys == ["logs/legacy-task_metrics.json"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "model_type": "xgboost",
            "steps": [{"step": 1, "metrics": {"accuracy": 0.9}}],
        }))
        return keys[0]

    monkeypatch.setattr(viz_service, "get_settings", lambda: settings)
    monkeypatch.setattr(viz_service, "resolve_legacy_id_candidates", candidates)
    monkeypatch.setattr(viz_service, "restore_file", restore_file)

    payload = await viz_service.get_learning_curve("run-id", object())

    assert payload["resolved_id"] == "legacy-task"
    assert payload["steps"][0]["metrics"]["accuracy"] == 0.9


def test_require_dataset_file_restores_missing_dataset(monkeypatch, tmp_path):
    dataset = SimpleNamespace(id="dataset-123", file_path="storage/uploads/source.csv")
    restored = tmp_path / "storage" / "uploads" / "source.csv"
    restored.parent.mkdir(parents=True)
    restored.write_text("feature,target\n1,0\n")

    monkeypatch.setattr(
        resolver,
        "restore_dataset_file",
        lambda dataset_id, file_path: restored,
    )

    assert resolver._require_dataset_file(dataset) is dataset


def test_require_dataset_file_reports_missing_remote_dataset(monkeypatch):
    dataset = SimpleNamespace(id="dataset-123", file_path="storage/uploads/source.csv")
    monkeypatch.setattr(resolver, "restore_dataset_file", lambda *_args: None)

    with pytest.raises(HTTPException) as exc:
        resolver._require_dataset_file(dataset)

    assert exc.value.status_code == 404
    assert exc.value.detail == "Dataset artifact not found for this task"


@pytest.mark.asyncio
async def test_final_evaluation_specs_restore_model_dataset_and_dl_sidecar(
    monkeypatch,
    tmp_path,
):
    ml_model = tmp_path / "ml.joblib"
    dl_model = tmp_path / "dl.pt"
    Path(str(dl_model) + ".preprocessor.joblib").touch()
    dataset_path = tmp_path / "source.csv"
    restored_models = []
    restored_datasets = []

    def restore_model_bundle(path):
        restored_models.append(path)
        return ml_model if str(path).endswith(".joblib") else dl_model

    def restore_dataset_file(dataset_id, path):
        restored_datasets.append((dataset_id, path))
        return dataset_path

    monkeypatch.setattr(final_evaluation_service, "restore_model_bundle", restore_model_bundle)
    monkeypatch.setattr(final_evaluation_service, "restore_dataset_file", restore_dataset_file)

    ml_task = SimpleNamespace(
        model_path="storage/models/ml.joblib",
        dataset_id="dataset-1",
        target_column="target",
        model_type="random_forest",
        test_size=0.2,
        eval_metrics=["accuracy"],
    )
    dl_task = SimpleNamespace(
        model_path="storage/models/dl.pt",
        dataset_id="dataset-1",
        target_column="target",
        model_type="mlp",
        task_type="classification",
        train_config={},
    )
    dataset = SimpleNamespace(
        id="dataset-1",
        file_path="storage/uploads/source.csv",
    )

    ml_spec = await final_evaluation_service._ml_evaluation_spec(
        _SequenceDb(ml_task, dataset), SimpleNamespace(params={}), "ml-task"
    )
    dl_spec = await final_evaluation_service._dl_evaluation_spec(
        _SequenceDb(dl_task, dataset), SimpleNamespace(params={}), "dl-task"
    )

    assert ml_spec["model_path"] == ml_model
    assert ml_spec["dataset_path"] == dataset_path
    assert dl_spec["model_path"] == dl_model
    assert dl_spec["dataset_path"] == dataset_path
    assert dl_spec["aux_files"] == [Path(str(dl_model) + ".preprocessor.joblib")]
    assert restored_models == ["storage/models/ml.joblib", "storage/models/dl.pt"]
    assert restored_datasets == [
        ("dataset-1", "storage/uploads/source.csv"),
        ("dataset-1", "storage/uploads/source.csv"),
    ]


@pytest.mark.asyncio
async def test_final_evaluation_returns_structured_skip_when_remote_artifact_missing(
    monkeypatch,
):
    task = SimpleNamespace(
        model_path="storage/models/missing.joblib",
        dataset_id="dataset-1",
    )
    dataset = SimpleNamespace(
        id="dataset-1",
        file_path="storage/uploads/source.csv",
    )
    monkeypatch.setattr(final_evaluation_service, "restore_model_bundle", lambda _path: None)
    monkeypatch.setattr(
        final_evaluation_service,
        "restore_dataset_file",
        lambda _dataset_id, _path: pytest.fail("dataset restore must not run after model failure"),
    )

    spec = await final_evaluation_service._ml_evaluation_spec(
        _SequenceDb(task, dataset), SimpleNamespace(params={}), "ml-task"
    )

    assert spec == {"skip": "missing_model_artifact"}


@pytest.mark.asyncio
async def test_deploy_inference_reports_missing_restored_model(monkeypatch):
    deployment = SimpleNamespace(
        id="deployment-1",
        task_id="task-1",
        status="active",
        request_count=0,
    )
    task = SimpleNamespace(
        id="task-1",
        model_path="storage/models/missing.joblib",
        dataset_id="dataset-1",
    )
    calls = []
    monkeypatch.setattr(
        deploy_service,
        "restore_model_bundle",
        lambda path: calls.append(path) or None,
    )

    with pytest.raises(HTTPException) as exc:
        await deploy_service.run_inference(
            "deployment-1",
            [{"feature": 1}],
            _SequenceDb(deployment, task),
        )

    assert calls == ["storage/models/missing.joblib"]
    assert exc.value.status_code == 404
    assert exc.value.detail == "模型文件不存在，且对象存储中无可恢复副本"


@pytest.mark.asyncio
async def test_predict_rows_restores_model_and_dataset(monkeypatch, tmp_path):
    model_path = tmp_path / "model.joblib"
    dataset_path = tmp_path / "source.csv"
    task = SimpleNamespace(
        id="task-1",
        status="SUCCESS",
        model_path="storage/models/model.joblib",
        dataset_id="dataset-1",
        target_column="target",
        model_type="random_forest",
    )
    dataset = SimpleNamespace(
        id="dataset-1",
        file_path="storage/uploads/source.csv",
    )
    calls = []

    monkeypatch.setattr(
        prediction_service,
        "restore_model_bundle",
        lambda path: calls.append(("model", path)) or model_path,
    )
    monkeypatch.setattr(
        prediction_service,
        "restore_dataset_file",
        lambda dataset_id, path: calls.append(("dataset", dataset_id, path)) or dataset_path,
    )
    monkeypatch.setattr(
        prediction_service,
        "load_dataframe",
        lambda path: calls.append(("load", path)) or pd.DataFrame({"feature": [1], "target": [0]}),
    )
    monkeypatch.setattr(prediction_service.joblib, "load", lambda path: object())
    monkeypatch.setattr(
        prediction_service,
        "predict_with_model",
        lambda *_args, **_kwargs: {
            "predictions": [0],
            "class_labels": ["0"],
            "probabilities": None,
        },
    )

    response = await prediction_service.predict_rows(
        "task-1", [{"feature": 1}], _SequenceDb(task, dataset)
    )

    assert response["predictions"] == [0]
    assert calls == [
        ("model", "storage/models/model.joblib"),
        ("dataset", "dataset-1", "storage/uploads/source.csv"),
        ("load", dataset_path),
    ]


def test_validate_target_column_restores_before_best_effort_read(monkeypatch, tmp_path):
    restored = tmp_path / "source.csv"
    dataset = SimpleNamespace(
        id="dataset-1",
        file_path="storage/uploads/source.csv",
    )
    calls = []
    monkeypatch.setattr(
        modeling_task_service,
        "restore_dataset_file",
        lambda dataset_id, path: calls.append(("restore", dataset_id, path)) or restored,
    )
    monkeypatch.setattr(
        modeling_task_service,
        "load_dataframe",
        lambda path: calls.append(("load", path))
        or pd.DataFrame({"target": [0, 0, 1, 1]}),
    )

    modeling_task_service._validate_target_column(dataset, "target", "classification")

    assert calls == [
        ("restore", "dataset-1", "storage/uploads/source.csv"),
        ("load", restored),
    ]

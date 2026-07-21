from __future__ import annotations

import json
from types import SimpleNamespace

import joblib
import pytest
from fastapi import HTTPException

from app.services import resolver, viz_service


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

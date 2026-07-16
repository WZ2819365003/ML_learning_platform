"""Tests for leakage-safe DL preprocessing and sidecar compatibility."""

import joblib
import numpy as np
import pandas as pd
import pytest
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from app.core.dl_trainer import BaseDLTrainer
from app.core.model_artifact import (
    ARTIFACT_VERSION,
    DLPreprocessingArtifact,
    fit_dl_preprocessing_artifact,
)
from app.services import dl_service
from app.services.dl_service import _prepare_dl_data, _prepare_dl_prediction_input


class TinyDLTrainer(BaseDLTrainer):
    def build_model(self, input_dim: int, output_dim: int, arch_config: dict):
        return nn.Linear(input_dim, output_dim)


def test_dl_preprocessing_uses_train_statistics_and_contiguous_labels():
    X_train = pd.DataFrame(
        {
            "temperature": [1.0, np.nan, 3.0, 5.0],
            "kind": ["cold", "cold", "hot", "hot"],
        }
    )
    y_train = pd.Series([10, 20, 10, 20])
    artifact = fit_dl_preprocessing_artifact(
        X_train,
        y_train,
        task_kind="classification",
    )

    transformed = artifact.transform_features(
        pd.DataFrame(
            {"temperature": [1000.0, np.nan], "kind": ["hot", "cold"]}
        )
    )

    assert artifact.preprocessor.numeric_fill_values == {"temperature": 3.0}
    assert transformed[:, 0].tolist() == [1000.0, 3.0]
    assert artifact.encode_target(pd.Series([10, 20])).tolist() == [0, 1]
    assert artifact.decode_predictions(np.array([1, 0])).tolist() == [20, 10]


def test_dl_preprocessing_round_trip_and_version_guard(tmp_path):
    artifact = fit_dl_preprocessing_artifact(
        pd.DataFrame({"value": [1, 2, 8, 9]}),
        pd.Series(["normal", "normal", "fault", "fault"]),
        task_kind="classification",
    )
    path = tmp_path / "model.pt.preprocessor.joblib"
    joblib.dump(artifact, path)
    loaded = joblib.load(path)

    assert isinstance(loaded, DLPreprocessingArtifact)
    assert loaded.encode_target(pd.Series(["fault", "normal"])).tolist() == [0, 1]
    assert loaded.decode_predictions(np.array([1, 0])).tolist() == ["normal", "fault"]

    loaded.artifact_version = ARTIFACT_VERSION + 1
    with pytest.raises(ValueError, match="Unsupported DL preprocessing artifact version"):
        loaded.transform_features(pd.DataFrame({"value": [3]}))


def test_prepare_dl_data_splits_before_fitting_preprocessing(tmp_path):
    frame = pd.DataFrame(
        {
            "temperature": [1.0, np.nan, 3.0, 5.0, 7.0, 9.0, 11.0, 1000.0],
            "kind": ["cold", "hot"] * 4,
            "label": [10, 20] * 4,
        }
    )
    path = tmp_path / "training.csv"
    frame.to_csv(path, index=False)

    raw_X = frame.drop(columns=["label"])
    raw_y = frame["label"]
    expected_X_train, _, _, _ = train_test_split(
        raw_X,
        raw_y,
        test_size=0.25,
        random_state=42,
        stratify=raw_y,
    )

    X_train, X_val, y_train, y_val, artifact, resolved_task_type = (
        _prepare_dl_data(str(path), "label", 0.25, "classification")
    )

    assert resolved_task_type == "classification"
    assert artifact.preprocessor.numeric_fill_values["temperature"] == pytest.approx(
        expected_X_train["temperature"].median()
    )
    assert X_train.dtype == np.float32
    assert X_val.dtype == np.float32
    assert set(np.concatenate([y_train, y_val])) == {0, 1}


def test_prepare_dl_data_auto_detects_string_labels_as_classification(tmp_path):
    frame = pd.DataFrame(
        {
            "value": np.arange(12, dtype=float),
            "label": ["normal", "fault"] * 6,
        }
    )
    path = tmp_path / "training.csv"
    frame.to_csv(path, index=False)

    *_, artifact, resolved_task_type = _prepare_dl_data(
        str(path), "label", 0.25, "auto"
    )

    assert resolved_task_type == "classification"
    assert artifact.class_labels == ["fault", "normal"]


def test_dl_trainer_persists_and_loads_preprocessing_sidecar(tmp_path):
    artifact = fit_dl_preprocessing_artifact(
        pd.DataFrame({"value": [1.0, 2.0, 3.0, 4.0]}),
        pd.Series([0, 1, 0, 1]),
        task_kind="classification",
    )
    model_path = tmp_path / "model.pt"
    trainer = TinyDLTrainer()
    trainer.model = trainer.build_model(1, 2, {})
    trainer.num_classes = 2
    trainer.scaler = StandardScaler().fit(np.array([[1.0], [2.0]]))

    trainer.save(
        str(model_path),
        input_dim=1,
        task_type="classification",
        feature_columns=["value"],
        preprocessing_artifact=artifact,
    )

    assert (tmp_path / "model.pt.preprocessor.joblib").exists()
    loaded = TinyDLTrainer()
    metadata = loaded.load_for_inference(str(model_path))
    assert isinstance(metadata["preprocessing_artifact"], DLPreprocessingArtifact)
    assert metadata["preprocessing_artifact"].feature_names == ["value"]


def test_dl_trainer_loads_legacy_model_without_preprocessing_sidecar(tmp_path):
    model_path = tmp_path / "legacy.pt"
    trainer = TinyDLTrainer()
    trainer.model = trainer.build_model(1, 1, {})
    trainer.save(
        str(model_path),
        input_dim=1,
        task_type="regression",
        feature_columns=["value"],
    )

    loaded = TinyDLTrainer()
    metadata = loaded.load_for_inference(str(model_path))
    assert metadata["preprocessing_artifact"] is None


def test_dl_prediction_uses_sidecar_without_refitting_training_data(monkeypatch):
    artifact = fit_dl_preprocessing_artifact(
        pd.DataFrame({"value": [1.0, np.nan, 5.0]}),
        pd.Series([0, 1, 0]),
        task_kind="classification",
    )

    def fail_legacy_preprocessing(*args, **kwargs):
        raise AssertionError("legacy preprocessing should not run for sidecar models")

    monkeypatch.setattr(
        dl_service, "prepare_prediction_frame", fail_legacy_preprocessing
    )
    values, columns = _prepare_dl_prediction_input(
        {"preprocessing_artifact": artifact},
        None,
        [{"value": None}, {"value": 9.0}],
        "label",
    )

    assert columns == ["value"]
    assert values.dtype == np.float32
    assert values[:, 0].tolist() == [3.0, 9.0]


def test_dl_prediction_legacy_fallback_uses_training_frame(monkeypatch):
    training_df = pd.DataFrame({"value": [1.0, 2.0], "label": [0, 1]})
    calls = []

    def fake_prepare(reference, rows, target_column):
        calls.append((reference, rows, target_column))
        return pd.DataFrame({"value": [7.0]})

    monkeypatch.setattr(dl_service, "prepare_prediction_frame", fake_prepare)
    values, columns = _prepare_dl_prediction_input(
        {"preprocessing_artifact": None},
        training_df,
        [{"value": 7.0}],
        "label",
    )

    assert calls == [(training_df, [{"value": 7.0}], "label")]
    assert columns == ["value"]
    assert values.tolist() == [[7.0]]

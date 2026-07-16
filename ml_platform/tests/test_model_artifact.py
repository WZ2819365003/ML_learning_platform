"""Tests for self-contained, leakage-safe classic ML artifacts."""

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from app.core.model_artifact import (
    ARTIFACT_VERSION,
    TabularModelArtifact,
    TabularPreprocessor,
    fit_tabular_artifact,
    is_tabular_artifact,
)


def test_preprocessor_uses_only_fit_frame_statistics():
    fit_frame = pd.DataFrame(
        {
            "temperature": [1.0, np.nan, 3.0],
            "kind": ["b", "c", "b"],
        }
    )
    preprocessor = TabularPreprocessor().fit(fit_frame)

    transformed = preprocessor.transform(
        pd.DataFrame({"temperature": [1000.0, np.nan], "kind": ["b", "c"]})
    )

    assert transformed[:, 0].tolist() == [1000.0, 2.0]
    assert preprocessor.numeric_fill_values == {"temperature": 2.0}
    assert preprocessor.categorical_mappings == {"kind": {"b": 0, "c": 1}}


def test_preprocessor_encodes_unknown_categories_without_refitting():
    preprocessor = TabularPreprocessor().fit(
        pd.DataFrame({"kind": ["known", "known"], "value": [1, 2]})
    )

    transformed = preprocessor.transform(
        pd.DataFrame({"kind": ["new"], "value": [3]})
    )

    assert transformed.tolist() == [[-1.0, 3.0]]
    assert preprocessor.categorical_mappings == {"kind": {"known": 0}}


def test_classifier_cv_handles_fold_local_high_cardinality_categories():
    from app.core.trainer import LogisticRegressionTrainer

    frame = pd.DataFrame(
        {
            "Product ID": [f"P{i:03d}" for i in range(18)],
            "temperature": np.linspace(20.0, 80.0, 18),
        }
    )
    labels = pd.Series([0, 1] * 9)
    trainer = LogisticRegressionTrainer()
    trainer.configure({})

    metrics = trainer.train(
        frame.iloc[:15].reset_index(drop=True),
        labels.iloc[:15].reset_index(drop=True),
        frame.iloc[15:].reset_index(drop=True),
        labels.iloc[15:].reset_index(drop=True),
        eval_metrics=["accuracy", "f1"],
        cv_folds=3,
    )

    assert metrics["selection_cv_mean_accuracy"] is not None
    assert len(metrics["cv_folds"]) == 3


def test_artifact_round_trip_preserves_decoded_predictions(tmp_path):
    X = pd.DataFrame(
        {
            "temperature": [1, 2, 3, 7, 8, 9],
            "kind": ["low", "low", "low", "high", "high", "high"],
        }
    )
    y = pd.Series(["normal", "normal", "normal", "fault", "fault", "fault"])
    artifact = fit_tabular_artifact(
        LogisticRegression(random_state=42),
        X,
        y,
        task_kind="classification",
    )

    rows = pd.DataFrame(
        {"temperature": [1.5, 8.5], "kind": ["low", "high"]}
    )
    expected = artifact.predict(rows).tolist()
    model_path = tmp_path / "artifact.joblib"
    joblib.dump(artifact, model_path)
    loaded = joblib.load(model_path)

    assert isinstance(loaded, TabularModelArtifact)
    assert is_tabular_artifact(loaded)
    assert loaded.artifact_version == ARTIFACT_VERSION
    assert loaded.predict(rows).tolist() == expected
    assert loaded.class_labels == ["fault", "normal"]
    assert loaded.predict_proba(rows).shape == (2, 2)


def test_training_logs_self_contained_joblib_to_mlflow(tmp_path):
    from app.services.training_service import _log_model_artifact

    class FakeMlflow:
        def __init__(self):
            self.calls = []

        def log_artifact(self, path, artifact_path=None):
            self.calls.append((path, artifact_path))

    model_file = tmp_path / "model.joblib"
    model_file.write_bytes(b"artifact")
    mlflow = FakeMlflow()

    _log_model_artifact(mlflow, model_file)

    assert mlflow.calls == [(str(model_file), "model")]


def test_artifact_rejects_unsupported_version():
    X = pd.DataFrame({"value": [1, 2, 8, 9]})
    artifact = fit_tabular_artifact(
        LogisticRegression(random_state=42),
        X,
        pd.Series([0, 0, 1, 1]),
        task_kind="classification",
    )
    artifact.artifact_version = ARTIFACT_VERSION + 1

    with pytest.raises(ValueError, match="Unsupported tabular model artifact version"):
        artifact.predict(pd.DataFrame({"value": [2]}))

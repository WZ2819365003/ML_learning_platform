import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from app.api.routes.model_mgmt import router as model_mgmt_router
from app.core.model_artifact import fit_tabular_artifact
from app.models.schemas import PredictionResponse
from app.services import prediction_service
from app.services.prediction_service import prepare_prediction_frame, predict_with_model


def test_prepare_prediction_frame_drops_leaked_target_columns():
    training_df = pd.DataFrame(
        {
            "UDI": [1, 2, 3],
            "Type": ["L", "M", "H"],
            "Target": [0, 1, 0],
            "Failure Type": ["No Failure", "Heat Dissipation Failure", "No Failure"],
        }
    )

    frame = prepare_prediction_frame(
        training_df,
        [{"UDI": 4, "Type": "M", "Target": 1, "Failure Type": "Power Failure"}],
        target_column="Target",
    )

    assert list(frame.columns) == ["UDI", "Type"]
    assert frame.shape == (1, 2)


def test_prepare_prediction_frame_rejects_unknown_categories():
    training_df = pd.DataFrame(
        {
            "UDI": [1, 2, 3],
            "Type": ["L", "M", "H"],
            "Target": [0, 1, 0],
        }
    )

    with pytest.raises(ValueError, match="Unknown category"):
        prepare_prediction_frame(
            training_df,
            [{"UDI": 4, "Type": "X"}],
            target_column="Target",
        )


def test_prepare_prediction_frame_requires_all_features():
    training_df = pd.DataFrame(
        {
            "UDI": [1, 2, 3],
            "Torque [Nm]": [40.2, 42.1, 38.7],
            "Target": [0, 1, 0],
        }
    )

    with pytest.raises(ValueError, match="Missing required feature columns"):
        prepare_prediction_frame(
            training_df,
            [{"UDI": 4}],
            target_column="Target",
        )


def test_new_artifact_prediction_does_not_refit_from_training_dataframe(monkeypatch):
    artifact = fit_tabular_artifact(
        LogisticRegression(random_state=42),
        pd.DataFrame(
            {
                "temperature": [1, 2, 3, 7, 8, 9],
                "kind": ["low", "low", "low", "high", "high", "high"],
            }
        ),
        pd.Series(["normal", "normal", "normal", "fault", "fault", "fault"]),
        task_kind="classification",
    )
    training_df = pd.DataFrame(
        {
            "temperature": [1, 9, 1000],
            "kind": ["low", "high", "validation-only"],
            "label": ["normal", "fault", "fault"],
        }
    )

    def fail_if_legacy_preparation_runs(*args, **kwargs):
        raise AssertionError("artifact inference must not refit preprocessing")

    monkeypatch.setattr(
        prediction_service,
        "prepare_prediction_frame",
        fail_if_legacy_preparation_runs,
    )
    result = predict_with_model(
        artifact,
        training_df,
        [{"temperature": 2, "kind": "low"}],
        "label",
        include_probabilities=True,
    )

    assert result["predictions"] == ["normal"]
    assert result["class_labels"] == ["fault", "normal"]
    assert np.asarray(result["probabilities"]).shape == (1, 2)


def test_prediction_adapter_preserves_string_class_label_contract():
    training_df = pd.DataFrame(
        {
            "value": [1, 2, 3, 7, 8, 9],
            "label": [0, 0, 0, 1, 1, 1],
        }
    )
    legacy_model = RandomForestClassifier(n_estimators=5, random_state=42)
    legacy_model.fit(training_df[["value"]].values, training_df["label"].values)

    legacy_result = predict_with_model(
        legacy_model,
        training_df,
        [{"value": 2}],
        "label",
    )
    artifact = fit_tabular_artifact(
        RandomForestClassifier(n_estimators=5, random_state=42),
        training_df[["value"]],
        training_df["label"],
        task_kind="classification",
    )
    artifact_result = predict_with_model(
        artifact,
        training_df,
        [{"value": 2}],
        "label",
    )

    assert legacy_result["class_labels"] == ["0", "1"]
    assert artifact_result["class_labels"] == ["0", "1"]


def test_regression_prediction_omits_classification_metadata_for_legacy_and_artifact():
    training_df = pd.DataFrame(
        {
            "value": [1, 2, 3, 7, 8, 9],
            "target": [10.5, 20.25, 31.75, 70.5, 82.25, 95.0],
        }
    )
    legacy_model = RandomForestRegressor(n_estimators=5, random_state=42)
    legacy_model.fit(training_df[["value"]].values, training_df["target"].values)

    legacy_result = predict_with_model(
        legacy_model,
        training_df,
        [{"value": 4}],
        "target",
    )
    artifact = fit_tabular_artifact(
        RandomForestRegressor(n_estimators=5, random_state=42),
        training_df[["value"]],
        training_df["target"],
        task_kind="regression",
    )
    artifact_result = predict_with_model(
        artifact,
        training_df,
        [{"value": 4}],
        "target",
    )

    for result in (legacy_result, artifact_result):
        assert len(result["predictions"]) == 1
        assert result["class_labels"] == []
        assert result["probabilities"] is None


def test_regression_api_response_serialization_omits_classification_only_fields():
    payload = PredictionResponse(
        task_id="regression-task",
        model_type="xgboost_regressor",
        target_column="load",
        rows=1,
        predictions=[6924.63],
    ).model_dump(exclude_none=True)

    assert payload["predictions"] == [6924.63]
    assert "class_labels" not in payload
    assert "probabilities" not in payload

    predict_route = next(
        route
        for route in model_mgmt_router.routes
        if getattr(route, "path", None) == "/models/{task_id}/predict"
    )
    assert predict_route.response_model_exclude_none is True

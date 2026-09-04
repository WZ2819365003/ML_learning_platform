"""Regression tests for training data preparation."""

from pathlib import Path

import numpy as np
import pandas as pd

from app.services import training_service
from app.services.training_service import _prepare_data


def test_prepare_data_excludes_failure_type_when_target_is_target(tmp_path: Path):
    """Derived label columns must not leak into binary target training."""
    dataset_path = tmp_path / "predictive_maintenance_sample.csv"
    df = pd.DataFrame(
        {
            "sensor_1": [10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
            "Target": [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
            "Failure Type": [
                "No Failure",
                "No Failure",
                "No Failure",
                "No Failure",
                "No Failure",
                "Power Failure",
                "Power Failure",
                "Overstrain Failure",
                "Tool Wear Failure",
                "Heat Dissipation Failure",
            ],
        }
    )
    df.to_csv(dataset_path, index=False)

    X_train, X_val, y_train, y_val = _prepare_data(
        str(dataset_path),
        target_column="Target",
        test_size=0.2,
    )

    assert X_train.shape[1] == 1
    assert X_val.shape[1] == 1
    assert len(y_train) == 8
    assert len(y_val) == 2


def test_prepare_data_splits_raw_rows_before_fitted_transforms(tmp_path: Path):
    dataset_path = tmp_path / "raw_categories.csv"
    df = pd.DataFrame(
        {
            "temperature": [1.0, np.nan, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            "kind": ["a", "a", "b", "b", "a", "a", "b", "b"],
            "label": [0, 1, 0, 1, 0, 1, 0, 1],
        }
    )
    df.to_csv(dataset_path, index=False)

    X_train, X_val, y_train, y_val = _prepare_data(
        str(dataset_path),
        target_column="label",
        test_size=0.25,
    )

    assert isinstance(X_train, pd.DataFrame)
    assert isinstance(X_val, pd.DataFrame)
    assert isinstance(y_train, pd.Series)
    assert isinstance(y_val, pd.Series)
    assert X_train["kind"].dtype == object
    assert set(pd.concat([X_train["kind"], X_val["kind"]])) == {"a", "b"}
    assert int(pd.concat([X_train["temperature"], X_val["temperature"]]).isna().sum()) == 1


def test_selection_mode_hides_outer_holdout_from_trainer():
    X_val = pd.DataFrame({"value": [9.0]})
    y_val = pd.Series([1])

    assert training_service._trainer_validation_inputs(
        "selection", X_val, y_val
    ) == (None, None)
    standard_X, standard_y = training_service._trainer_validation_inputs(
        "standard", X_val, y_val
    )
    assert standard_X is X_val
    assert standard_y is y_val


def test_temporal_regression_keeps_latest_rows_as_holdout(tmp_path: Path):
    dataset_path = tmp_path / "load_forecast.csv"
    frame = pd.DataFrame({
        "load_lag_1": np.arange(10, dtype=float),
        "load": np.arange(100, 110, dtype=float),
    })
    frame.to_csv(dataset_path, index=False)

    X_train, X_val, y_train, y_val = _prepare_data(
        str(dataset_path), "load", 0.2, is_regression=True,
    )

    assert X_train["load_lag_1"].tolist() == list(range(8))
    assert X_val["load_lag_1"].tolist() == [8.0, 9.0]
    assert y_val.tolist() == [108.0, 109.0]

"""Regression tests for training data preparation."""

from pathlib import Path

import pandas as pd

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

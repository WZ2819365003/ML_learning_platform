import pandas as pd
import pytest

from app.services.prediction_service import prepare_prediction_frame


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

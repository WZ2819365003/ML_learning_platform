"""Regression tests for train/validation isolation during model evaluation."""

import numpy as np
import pandas as pd

from app.core.model_artifact import is_tabular_artifact
from app.core.regression_trainers import RegressionMixin
from app.core.trainer import BaseTrainer, get_trainer


class RecordingEstimator:
    """Minimal estimator that records how many rows each fit receives."""

    def __init__(self):
        self.fit_sizes: list[int] = []

    def fit(self, X, y):
        self.fit_sizes.append(len(X))
        return self

    def predict(self, X):
        return np.zeros(len(X))


class RecordingClassifierTrainer(BaseTrainer):
    def configure(self, hyperparameters: dict):
        self.model = RecordingEstimator()


class RecordingRegressionTrainer(RegressionMixin, BaseTrainer):
    def configure(self, hyperparameters: dict):
        self.model = RecordingEstimator()


def test_classification_cv_never_fits_validation_rows():
    trainer = RecordingClassifierTrainer()
    trainer.configure({})

    X_train = np.arange(16).reshape(8, 2)
    y_train = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    X_val = np.arange(16, 20).reshape(2, 2)
    y_val = np.array([0, 1])

    metrics = trainer.train(
        X_train,
        y_train,
        X_val,
        y_val,
        eval_metrics=["accuracy"],
        cv_folds=2,
    )

    assert trainer.model.fit_sizes == [4, 4, 8]
    assert metrics["selection_cv_mean_accuracy"] == metrics["cv_avg_accuracy"]
    assert metrics["selection_cv_std_accuracy"] == metrics["cv_std_accuracy"]
    assert metrics["final_test_accuracy"] == metrics["accuracy"]


def test_regression_cv_never_fits_validation_rows():
    trainer = RecordingRegressionTrainer()
    trainer.configure({})

    X_train = np.arange(16).reshape(8, 2)
    y_train = np.arange(8, dtype=float)
    X_val = np.arange(16, 20).reshape(2, 2)
    y_val = np.arange(8, 10, dtype=float)

    metrics = trainer.train(
        X_train,
        y_train,
        X_val,
        y_val,
        eval_metrics=["rmse"],
        cv_folds=2,
    )

    assert trainer.model.fit_sizes == [4, 4, 8]
    assert metrics["selection_cv_mean_rmse"] == metrics["cv_avg_rmse"]
    assert metrics["selection_cv_std_rmse"] == metrics["cv_std_rmse"]
    assert metrics["final_test_rmse"] == metrics["rmse"]


def test_temporal_regression_uses_expanding_window_cv():
    trainer = RecordingRegressionTrainer()
    trainer.configure({})
    class TemporalArray(np.ndarray):
        columns = ["load_lag_1"]

    X_train = np.arange(8, dtype=float).reshape(-1, 1).view(TemporalArray)
    y_train = np.arange(8, dtype=float)

    metrics = trainer.train(
        X_train, y_train, None, None, eval_metrics=["rmse"], cv_folds=2,
    )

    assert trainer.model.fit_sizes == [4, 6, 8]
    assert metrics["validation_strategy"] == "time_series_expanding"


def test_dataframe_training_persists_train_only_preprocessing_state():
    trainer = get_trainer("logistic_regression")
    trainer.configure({"random_state": 42})
    X_train = pd.DataFrame(
        {
            "temperature": [1.0, 3.0, 5.0, 7.0, 9.0, 11.0, 13.0, np.nan],
            "kind": ["a", "a", "a", "a", "b", "b", "b", "b"],
        }
    )
    y_train = pd.Series([0, 1, 0, 1, 0, 1, 0, 1])
    X_val = pd.DataFrame({"temperature": [1000.0, np.nan], "kind": ["a", "b"]})
    y_val = pd.Series([0, 1])

    trainer.train(
        X_train,
        y_train,
        X_val,
        y_val,
        eval_metrics=["accuracy"],
        cv_folds=2,
    )

    assert is_tabular_artifact(trainer.model)
    assert trainer.model.preprocessor.numeric_fill_values["temperature"] == 7.0


def test_selection_only_training_emits_cv_without_final_test_metrics():
    trainer = RecordingClassifierTrainer()
    trainer.configure({})
    metrics = trainer.train(
        np.arange(16).reshape(8, 2),
        np.array([0, 1, 0, 1, 0, 1, 0, 1]),
        None,
        None,
        eval_metrics=["accuracy"],
        cv_folds=2,
    )

    assert metrics["selection_cv_mean_accuracy"] == metrics["cv_avg_accuracy"]
    assert "accuracy" not in metrics
    assert "final_test_accuracy" not in metrics

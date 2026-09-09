"""A tree-model run keeps its validation predictions, like the DL trainer does.

The regression mixin computed ``y_val_pred`` for the final metrics and dropped
it, so an ML sub-report could never draw the predicted-vs-actual chart that a
DL run gets from ``metrics.val_scatter``.
"""
import numpy as np
import pandas as pd

from app.core.regression_trainers import (
    RandomForestRegressorTrainer,
    _validation_scatter,
)


def _frame(n: int, seed: int = 0) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n)})
    y = pd.Series(3 * X["a"] - X["b"] + rng.normal(scale=0.1, size=n), name="y")
    return X, y


def test_regression_training_stores_the_validation_tail():
    X, y = _frame(160)
    trainer = RandomForestRegressorTrainer()
    trainer.configure({"n_estimators": 5, "max_depth": 3})
    metrics = trainer.train(X.iloc[:120], y.iloc[:120], X.iloc[120:], y.iloc[120:], cv_folds=3)

    scatter = metrics["val_scatter"]
    assert scatter["ordered"] is True
    assert len(scatter["actual"]) == len(scatter["predicted"]) == 40
    # The window is the validation split in its own order, not a shuffle.
    assert scatter["actual"] == [round(float(v), 6) for v in y.iloc[120:]]
    # The mirror into final_test_* keys runs before the scatter is attached.
    assert "final_test_val_scatter" not in metrics
    assert "cv_folds" in metrics


def test_the_window_is_the_last_five_hundred_points():
    actual = np.arange(700, dtype=float)
    scatter = _validation_scatter(actual, actual + 1)
    assert len(scatter["actual"]) == 500
    assert scatter["actual"][0] == 200.0
    assert scatter["predicted"][-1] == 700.0


def test_no_validation_split_means_no_scatter():
    assert _validation_scatter(None, None) is None
    assert _validation_scatter([], []) is None


def test_selection_mode_falls_back_to_the_last_fold():
    # Selection withholds the sealed hold-out (X_val is None), which is why the
    # first three tree-model reruns still had no scatter. The last fold's
    # out-of-sample predictions are the honest substitute, and the source is
    # recorded so the report can say so.
    X, y = _frame(160)
    trainer = RandomForestRegressorTrainer()
    trainer.configure({"n_estimators": 5, "max_depth": 3})
    metrics = trainer.train(X, y, None, None, cv_folds=4)

    scatter = metrics["val_scatter"]
    assert metrics["val_scatter_source"] == "cv_last_fold"
    assert len(scatter["actual"]) == len(scatter["predicted"]) > 0
    # KFold(n_splits=4) on 160 rows: the last fold holds 40 rows.
    assert len(scatter["actual"]) == 40


def test_a_real_holdout_is_labelled_as_one():
    X, y = _frame(160)
    trainer = RandomForestRegressorTrainer()
    trainer.configure({"n_estimators": 5, "max_depth": 3})
    metrics = trainer.train(X.iloc[:120], y.iloc[:120], X.iloc[120:], y.iloc[120:], cv_folds=3)
    assert metrics["val_scatter_source"] == "holdout"

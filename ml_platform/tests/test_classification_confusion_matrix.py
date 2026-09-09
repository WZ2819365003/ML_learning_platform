"""A classification run keeps its confusion matrix, like the DL trainer does.

``BaseTrainer.train`` computed ``y_val_pred``/``y_val_proba``, reduced them to
a handful of scalars and dropped them — so a classification sub-report had
neither a confusion matrix nor a ROC curve, the two figures a reader of a
classifier actually wants. The keys are spelled the way ``dl_trainer`` spells
them (``confusion_matrix``, ``val_roc_fpr``, ``val_roc_tpr``) so the report has
one reading path, not two.
"""
import numpy as np
import pandas as pd

from app.core.trainer import (
    LogisticRegressionTrainer,
    RandomForestTrainer,
    _classification_diagnostics,
)


def _binary(n: int, seed: int = 0) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n)})
    y = pd.Series((2 * X["a"] - X["b"] + rng.normal(scale=0.4, size=n) > 0).astype(int), name="y")
    return X, y


def _three_class(n: int, seed: int = 1) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n)})
    score = 2 * X["a"] - X["b"] + rng.normal(scale=0.3, size=n)
    y = pd.Series(pd.cut(score, bins=3, labels=["low", "mid", "high"]).astype(str), name="y")
    return X, y


def _trained(X, y, X_val, y_val, cv_folds=3):
    trainer = RandomForestTrainer()
    trainer.configure({"n_estimators": 8, "max_depth": 4})
    return trainer.train(X, y, X_val, y_val, eval_metrics=["accuracy", "f1"], cv_folds=cv_folds)


class TestHoldout:
    def test_the_matrix_and_its_label_order_are_kept(self):
        X, y = _binary(200)
        metrics = _trained(X.iloc[:150], y.iloc[:150], X.iloc[150:], y.iloc[150:])

        matrix = metrics["confusion_matrix"]
        assert metrics["class_labels"] == ["0", "1"]
        assert len(matrix) == len(matrix[0]) == 2
        assert all(isinstance(cell, int) for row in matrix for cell in row)
        # Every hold-out row lands in exactly one cell.
        assert sum(sum(row) for row in matrix) == 50
        assert metrics["confusion_source"] == "holdout"

    def test_a_binary_run_keeps_the_roc_curve(self):
        X, y = _binary(200)
        metrics = _trained(X.iloc[:150], y.iloc[:150], X.iloc[150:], y.iloc[150:])

        fpr, tpr = metrics["val_roc_fpr"], metrics["val_roc_tpr"]
        assert 2 <= len(fpr) == len(tpr) <= 200
        assert fpr == sorted(fpr) and tpr == sorted(tpr)
        assert fpr[0] == 0.0 and fpr[-1] == 1.0
        assert tpr[0] == 0.0 and tpr[-1] == 1.0

    def test_the_matrix_is_not_mirrored_into_a_final_test_key(self):
        # The final_test_ mirror runs before the diagnostics are attached; a
        # second copy of the matrix would sit in every payload for nothing.
        X, y = _binary(200)
        metrics = _trained(X.iloc[:150], y.iloc[:150], X.iloc[150:], y.iloc[150:])
        assert "final_test_confusion_matrix" not in metrics
        assert "final_test_class_labels" not in metrics


class TestSelectionMode:
    def test_no_holdout_falls_back_to_the_last_fold(self):
        # Selection withholds the sealed hold-out (X_val is None). The last
        # fold's out-of-sample predictions are the honest substitute, and the
        # source is recorded so the chart can say which it is showing.
        X, y = _binary(200)
        metrics = _trained(X, y, None, None, cv_folds=4)

        assert metrics["confusion_source"] == "cv_last_fold"
        assert metrics["class_labels"] == ["0", "1"]
        # StratifiedKFold(4) on 200 rows: the last fold holds 50.
        assert sum(sum(row) for row in metrics["confusion_matrix"]) == 50
        assert len(metrics["val_roc_fpr"]) == len(metrics["val_roc_tpr"]) >= 2


class TestMulticlass:
    def test_three_classes_give_a_three_by_three_matrix_and_no_roc(self):
        X, y = _three_class(240)
        metrics = _trained(X.iloc[:180], y.iloc[:180], X.iloc[180:], y.iloc[180:])

        assert metrics["class_labels"] == ["high", "low", "mid"]
        assert len(metrics["confusion_matrix"]) == 3
        assert all(len(row) == 3 for row in metrics["confusion_matrix"])
        # ROC as drawn here is a binary figure; multiclass gets no curve.
        assert "val_roc_fpr" not in metrics

    def test_a_row_is_a_true_class_and_a_column_is_a_predicted_one(self):
        # The chart reads cell[y][x] as "true class y was called x", so the
        # orientation has to be pinned rather than inferred from a diagonal.
        result = _classification_diagnostics(
            ["low", "low", "mid", "high"], ["low", "mid", "mid", "high"], None,
        )
        assert result["class_labels"] == ["high", "low", "mid"]
        assert result["confusion_matrix"] == [
            [1, 0, 0],   # high → high
            [0, 1, 1],   # low  → low once, mid once
            [0, 0, 1],   # mid  → mid
        ]


class TestDiagnosticsHelper:
    def test_the_positive_column_follows_the_models_class_order(self):
        # predict_proba's columns follow the estimator's own class order, which
        # need not be the sorted order the matrix rows use. Handing the columns
        # back-to-front must not silently produce a mirrored ROC curve.
        y_true = np.array(["no", "no", "yes", "yes"])
        y_pred = np.array(["no", "no", "yes", "yes"])
        # Columns ordered ["yes", "no"], so P(yes) is column 0.
        proba = np.array([[0.1, 0.9], [0.2, 0.8], [0.9, 0.1], [0.8, 0.2]])
        result = _classification_diagnostics(y_true, y_pred, proba, classes=["yes", "no"])
        assert result["class_labels"] == ["no", "yes"]
        # A perfect separation: the curve passes through the top-left corner.
        assert (0.0, 1.0) in list(zip(result["val_roc_fpr"], result["val_roc_tpr"]))

    def test_a_guessed_column_would_have_inverted_it(self):
        y_true = np.array(["no", "no", "yes", "yes"])
        y_pred = np.array(["no", "no", "yes", "yes"])
        proba = np.array([[0.1, 0.9], [0.2, 0.8], [0.9, 0.1], [0.8, 0.2]])
        wrong = _classification_diagnostics(y_true, y_pred, proba, classes=None)
        assert (1.0, 0.0) in list(zip(wrong["val_roc_fpr"], wrong["val_roc_tpr"]))
        assert (0.0, 1.0) not in list(zip(wrong["val_roc_fpr"], wrong["val_roc_tpr"]))

    def test_no_predictions_means_no_diagnostics(self):
        assert _classification_diagnostics(None, None, None) == {}

    def test_the_curve_is_downsampled_to_two_hundred_points(self):
        rng = np.random.default_rng(3)
        y_true = rng.integers(0, 2, size=2000)
        score = rng.random(2000)
        proba = np.column_stack([1 - score, score])
        result = _classification_diagnostics(y_true, y_true, proba, classes=[0, 1])
        assert len(result["val_roc_fpr"]) == len(result["val_roc_tpr"]) == 200


def test_a_logistic_run_stores_the_same_keys():
    # The diagnostics live on BaseTrainer, so every classification trainer in
    # the registry gets them, not only the tree models.
    X, y = _binary(160, seed=7)
    trainer = LogisticRegressionTrainer()
    trainer.configure({})
    metrics = trainer.train(X.iloc[:120], y.iloc[:120], X.iloc[120:], y.iloc[120:], cv_folds=3)
    assert {"confusion_matrix", "class_labels", "confusion_source",
            "val_roc_fpr", "val_roc_tpr"} <= set(metrics)

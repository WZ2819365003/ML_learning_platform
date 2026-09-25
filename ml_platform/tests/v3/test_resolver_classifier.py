"""Tests for the model_type → classifier/regressor classifier in resolver.py.

Regression guard: `logistic_regression` ends with `_regression` and was
previously misclassified as a regressor by the suffix heuristic, breaking
classification-only viz endpoints (per_class / pr_curve / calibration /
threshold / confusion_matrix).
"""
import pytest

from app.services.resolver import is_regressor, task_kind_for


@pytest.mark.parametrize(
    "token",
    [
        "logistic_regression",
        "LOGISTIC_REGRESSION",
        "Logistic_Regression",
    ],
)
def test_logistic_regression_is_classifier(token):
    """logistic_regression must NOT be classified as a regressor."""
    assert is_regressor(token) is False
    assert task_kind_for(token) == "classification"


@pytest.mark.parametrize(
    "token",
    [
        "linear_regression",
        "random_forest_regressor",
        "xgboost_regressor",
        "lightgbm_regressor",
        "ridge",
        "lasso",
        "elasticnet",
        "svr",
        "mlp_regressor",
        "mlp_dl_regressor",
        "lstm_regressor",
        "tcn_regressor",
        "custom_regressor",  # suffix-only match still works
        "my_model_regression",  # suffix-only match still works
    ],
)
def test_known_regressors(token):
    assert is_regressor(token) is True
    assert task_kind_for(token) == "regression"


@pytest.mark.parametrize(
    "token",
    [
        "random_forest",
        "xgboost",
        "lightgbm",
        "svm",
        "mlp_dl",
        "decision_tree",
        None,
        "",
    ],
)
def test_classifiers_and_empty(token):
    assert is_regressor(token) is False

"""Tests for explicit model-selection versus final-test metric semantics."""

from app.core.evaluation_metrics import resolve_objective_metrics
from app.services.tuning_service import _normalise_run_metrics


def test_resolver_prefers_selection_cv_and_explicit_final_test():
    resolved = resolve_objective_metrics(
        {
            "accuracy": 0.99,
            "cv_avg_accuracy": 0.81,
            "selection_cv_mean_accuracy": 0.82,
            "final_test_accuracy": 0.99,
        },
        "accuracy",
    )

    assert resolved.selection_metric_key == "selection_cv_mean_accuracy"
    assert resolved.selection_value == 0.82
    assert resolved.final_test_metric_key == "final_test_accuracy"
    assert resolved.final_test_value == 0.99


def test_resolver_supports_legacy_cv_and_raw_metric_fallbacks():
    cv_resolved = resolve_objective_metrics(
        {"accuracy": 0.95, "cv_avg_accuracy": 0.84},
        "accuracy",
    )
    raw_resolved = resolve_objective_metrics({"accuracy": 0.77}, "accuracy")

    assert cv_resolved.selection_metric_key == "cv_avg_accuracy"
    assert cv_resolved.selection_value == 0.84
    assert cv_resolved.final_test_metric_key == "accuracy"
    assert cv_resolved.final_test_value == 0.95
    assert raw_resolved.selection_metric_key == "accuracy"
    assert raw_resolved.selection_value == 0.77
    assert raw_resolved.final_test_value == 0.77


def test_resolver_normalizes_prefixed_objective_and_ignores_invalid_values():
    resolved = resolve_objective_metrics(
        {
            "selection_cv_mean_rmse": "not-a-number",
            "cv_avg_rmse": 1.25,
            "final_test_rmse": 1.4,
        },
        "cv_avg_rmse",
    )

    assert resolved.objective_metric == "rmse"
    assert resolved.selection_metric_key == "cv_avg_rmse"
    assert resolved.selection_value == 1.25
    assert resolved.final_test_metric_key == "final_test_rmse"
    assert resolved.final_test_value == 1.4


def test_resolver_returns_explicit_empty_canonical_keys():
    resolved = resolve_objective_metrics({}, "f1")

    assert resolved.selection_metric_key == "selection_cv_mean_f1"
    assert resolved.selection_value is None
    assert resolved.final_test_metric_key == "final_test_f1"
    assert resolved.final_test_value is None


def test_dl_metric_normalization_marks_validation_metrics_as_final_test():
    metrics = _normalise_run_metrics({"val_acc": 0.91, "val_rmse": 1.2})

    assert metrics["accuracy"] == 0.91
    assert metrics["final_test_accuracy"] == 0.91
    assert metrics["rmse"] == 1.2
    assert metrics["final_test_rmse"] == 1.2

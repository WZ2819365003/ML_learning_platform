"""Canonical semantics for model-selection and final-test metrics."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any


_METRIC_PREFIXES = (
    "selection_cv_mean_",
    "selection_cv_std_",
    "final_test_",
    "cv_avg_",
    "cv_std_",
)


@dataclass(frozen=True)
class ResolvedObjectiveMetrics:
    objective_metric: str
    selection_metric_key: str
    selection_value: float | None
    final_test_metric_key: str
    final_test_value: float | None


def resolve_objective_metrics(
    metrics: dict[str, Any] | None,
    objective_metric: str | None,
) -> ResolvedObjectiveMetrics:
    """Resolve selection and final-test values with backward-compatible fallbacks."""
    metric = _base_metric_name(objective_metric)
    values = metrics or {}

    selection_default = f"selection_cv_mean_{metric}"
    selection_key, selection_value = _first_numeric(
        values,
        (selection_default, f"cv_avg_{metric}", metric),
        default_key=selection_default,
    )
    final_default = f"final_test_{metric}"
    final_key, final_value = _first_numeric(
        values,
        (final_default, metric),
        default_key=final_default,
    )
    return ResolvedObjectiveMetrics(
        objective_metric=metric,
        selection_metric_key=selection_key,
        selection_value=selection_value,
        final_test_metric_key=final_key,
        final_test_value=final_value,
    )


def _base_metric_name(metric: str | None) -> str:
    normalized = str(metric or "accuracy").strip() or "accuracy"
    for prefix in _METRIC_PREFIXES:
        if normalized.startswith(prefix):
            return normalized[len(prefix):]
    return normalized


def _first_numeric(
    metrics: dict[str, Any],
    candidates: tuple[str, ...],
    *,
    default_key: str,
) -> tuple[str, float | None]:
    for key in candidates:
        value = metrics.get(key)
        if value is None:
            continue
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(numeric):
            return key, numeric
    return default_key, None

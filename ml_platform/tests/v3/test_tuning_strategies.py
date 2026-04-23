"""Validation tests for tuning_service._validate_search_space.

These guard the payload contract between the workbench UI and the three
strategy expanders.  A misshaped search_space used to reach the expanders
and either produce zero trials or raise obscure KeyErrors — the validator
now rejects bad payloads with an actionable 422 before any DB work.
"""

from __future__ import annotations

import pytest
from fastapi import HTTPException

from app.services.tuning_service import _validate_search_space


# ---------------------------------------------------------------------------
# Happy paths — each strategy accepts its canonical shape
# ---------------------------------------------------------------------------

def test_baseline_accepts_free_form_overrides():
    """Baseline overrides are unconstrained — scalars, lists, dicts all fine."""
    _validate_search_space(
        "baseline",
        {
            "random_forest": {"n_estimators": 200, "max_depth": 6},
            "xgboost": {"learning_rate": 0.05},
        },
        ["random_forest", "xgboost"],
    )


def test_baseline_accepts_empty_payload():
    _validate_search_space("baseline", {}, ["random_forest"])
    _validate_search_space("baseline", None, ["random_forest"])


def test_grid_search_accepts_list_values():
    _validate_search_space(
        "grid_search",
        {
            "random_forest": {
                "n_estimators": [50, 100, 200],
                "max_depth": [None, 6, 12],
            }
        },
        ["random_forest"],
    )


def test_bayesian_search_accepts_dist_specs():
    _validate_search_space(
        "bayesian_search",
        {
            "random_forest": {
                "n_estimators": {"type": "int", "low": 10, "high": 300, "step": 10},
                "max_features": {
                    "type": "categorical",
                    "choices": ["sqrt", "log2"],
                },
                "min_samples_split": {"type": "float", "low": 0.01, "high": 0.5, "log": True},
            }
        },
        ["random_forest"],
    )


# ---------------------------------------------------------------------------
# Rejections — structural errors
# ---------------------------------------------------------------------------

def test_grid_search_rejects_scalar_leaf():
    with pytest.raises(HTTPException) as exc:
        _validate_search_space(
            "grid_search",
            {"random_forest": {"n_estimators": 100}},  # wrong: should be [100]
            ["random_forest"],
        )
    assert exc.value.status_code == 422
    assert "list of candidate values" in exc.value.detail
    assert "n_estimators" in exc.value.detail


def test_grid_search_rejects_empty_list():
    with pytest.raises(HTTPException) as exc:
        _validate_search_space(
            "grid_search",
            {"random_forest": {"n_estimators": []}},
            ["random_forest"],
        )
    assert exc.value.status_code == 422
    assert "empty" in exc.value.detail.lower()


def test_bayesian_rejects_missing_type():
    with pytest.raises(HTTPException) as exc:
        _validate_search_space(
            "bayesian_search",
            {"random_forest": {"n_estimators": {"low": 10, "high": 100}}},  # missing type
            ["random_forest"],
        )
    assert exc.value.status_code == 422
    assert "type" in exc.value.detail


def test_bayesian_rejects_bad_type():
    with pytest.raises(HTTPException) as exc:
        _validate_search_space(
            "bayesian_search",
            {"random_forest": {"lr": {"type": "uniform", "low": 0, "high": 1}}},
            ["random_forest"],
        )
    assert exc.value.status_code == 422
    assert "float" in exc.value.detail


def test_bayesian_rejects_missing_low_high():
    with pytest.raises(HTTPException) as exc:
        _validate_search_space(
            "bayesian_search",
            {"random_forest": {"lr": {"type": "float"}}},
            ["random_forest"],
        )
    assert exc.value.status_code == 422
    assert "low" in exc.value.detail and "high" in exc.value.detail


def test_bayesian_rejects_inverted_range():
    with pytest.raises(HTTPException) as exc:
        _validate_search_space(
            "bayesian_search",
            {"random_forest": {"lr": {"type": "float", "low": 1.0, "high": 0.1}}},
            ["random_forest"],
        )
    assert exc.value.status_code == 422
    assert "<" in exc.value.detail


def test_bayesian_rejects_empty_categorical():
    with pytest.raises(HTTPException) as exc:
        _validate_search_space(
            "bayesian_search",
            {"random_forest": {"solver": {"type": "categorical", "choices": []}}},
            ["random_forest"],
        )
    assert exc.value.status_code == 422
    assert "choices" in exc.value.detail


def test_bayesian_rejects_non_dict_spec():
    with pytest.raises(HTTPException) as exc:
        _validate_search_space(
            "bayesian_search",
            {"random_forest": {"lr": [0.01, 0.1, 1.0]}},  # grid shape submitted to bayesian
            ["random_forest"],
        )
    assert exc.value.status_code == 422
    assert "distribution dict" in exc.value.detail


def test_rejects_non_dict_top_level():
    with pytest.raises(HTTPException) as exc:
        _validate_search_space("grid_search", [1, 2, 3], ["rf"])  # type: ignore[arg-type]
    assert exc.value.status_code == 422


def test_rejects_non_dict_model_space():
    with pytest.raises(HTTPException) as exc:
        _validate_search_space(
            "grid_search",
            {"random_forest": [1, 2, 3]},  # missing param name layer
            ["random_forest"],
        )
    assert exc.value.status_code == 422
    assert "{param: spec}" in exc.value.detail


def test_extra_model_entries_are_ignored_not_rejected():
    """Stale entries from plan snapshots shouldn't fail the whole batch."""
    # Has an extra "old_model" key that's not in selected_models — just a warning.
    _validate_search_space(
        "grid_search",
        {
            "random_forest": {"n_estimators": [100, 200]},
            "old_model": {"foo": [1, 2]},  # stale — should be skipped
        },
        ["random_forest"],
    )


# ---------------------------------------------------------------------------
# End-to-end: dispatch_experiment_batch surfaces validator errors as 422
# ---------------------------------------------------------------------------

async def test_dispatch_rejects_scalar_grid_value(db):
    """The validator runs inside dispatch_experiment_batch, so the full
    HTTP path (modeling_tasks route → tuning_service) returns 422."""
    from app.models.database import Dataset, ModelingTask
    from app.services import tuning_service as svc

    ds = Dataset(name="x", file_path="/tmp/x", file_size=1, row_count=10)
    db.add(ds)
    await db.flush()
    task = ModelingTask(
        name="t", dataset_id=ds.id, target_column="y", task_type="classification"
    )
    db.add(task)
    await db.flush()

    with pytest.raises(HTTPException) as exc:
        await svc.dispatch_experiment_batch(
            db,
            modeling_task_id=task.id,
            name="bad-grid",
            strategy_type="grid_search",
            selected_models=["random_forest"],
            search_space={"random_forest": {"n_estimators": 100}},  # BAD: scalar
            budget_config={},
        )
    assert exc.value.status_code == 422
    assert "list" in exc.value.detail


async def test_dispatch_rejects_bayesian_missing_type(db):
    from app.models.database import Dataset, ModelingTask
    from app.services import tuning_service as svc

    ds = Dataset(name="x", file_path="/tmp/x", file_size=1, row_count=10)
    db.add(ds)
    await db.flush()
    task = ModelingTask(
        name="t", dataset_id=ds.id, target_column="y", task_type="classification"
    )
    db.add(task)
    await db.flush()

    with pytest.raises(HTTPException) as exc:
        await svc.dispatch_experiment_batch(
            db,
            modeling_task_id=task.id,
            name="bad-bayes",
            strategy_type="bayesian_search",
            selected_models=["random_forest"],
            search_space={
                "random_forest": {"lr": {"low": 0.01, "high": 0.1}}  # BAD: no type
            },
            budget_config={"n_trials_per_model": 3},
        )
    assert exc.value.status_code == 422

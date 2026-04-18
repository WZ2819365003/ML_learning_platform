"""
Tests for V3 tuning engines — trial expansion and distribution sampling.

Actual end-to-end training execution is NOT exercised here (that would need
pandas + sklearn + a real dataset on disk); instead we validate:

  - grid_search expands the cartesian product correctly
  - grid_search applies fixed params + user overrides
  - grid_search respects max_trials
  - baseline expansion is one run per model
  - bayesian distribution sampling produces valid Optuna suggestions
  - _count_bayesian_trials honours n_trials_per_model and max_trials
  - dispatch_experiment_batch rejects bad inputs
"""

from __future__ import annotations

import pytest
import optuna

from app.services import tuning_service as svc
from app.services.modeling_task_service import load_tuning_spaces


# ---------------------------------------------------------------------------
# grid_search expansion
# ---------------------------------------------------------------------------

def test_grid_search_expands_cartesian_product():
    defaults = load_tuning_spaces("classification")
    trials = svc._expand_grid_search(
        selected_models=["random_forest"],
        tuning_defaults=defaults,
        search_space={
            "random_forest": {
                "n_estimators": [100, 200],
                "max_depth": [None, 6],
            }
        },
        max_trials=None,
    )
    # 2 × 2 = 4 trials
    assert len(trials) == 4
    trial_nos = [t["trial_no"] for t in trials]
    assert trial_nos == [1, 2, 3, 4]
    for t in trials:
        assert t["model_type"] == "random_forest"
        assert t["search_meta"]["strategy"] == "grid_search"
        assert "grid_index" in t["search_meta"]
        assert t["hyperparameters"]["n_estimators"] in [100, 200]


def test_grid_search_applies_fixed_params():
    # svm has fixed: {probability: true} — must survive into every trial
    defaults = load_tuning_spaces("classification")
    trials = svc._expand_grid_search(
        selected_models=["svm"],
        tuning_defaults=defaults,
        search_space={"svm": {"C": [0.1, 1.0]}},
        max_trials=None,
    )
    assert len(trials) == 2
    assert all(t["hyperparameters"]["probability"] is True for t in trials)


def test_grid_search_falls_back_to_registry_defaults():
    """Empty user search_space → uses grid_values from YAML."""
    defaults = load_tuning_spaces("classification")
    trials = svc._expand_grid_search(
        selected_models=["logistic_regression"],
        tuning_defaults=defaults,
        search_space={},
        max_trials=None,
    )
    # YAML has C: 4 values × penalty: 1 × solver: 1 = 4 trials
    assert len(trials) == 4
    c_values = {t["hyperparameters"]["C"] for t in trials}
    assert c_values == {0.01, 0.1, 1.0, 10.0}


def test_grid_search_respects_max_trials():
    defaults = load_tuning_spaces("classification")
    trials = svc._expand_grid_search(
        selected_models=["random_forest", "xgboost"],
        tuning_defaults=defaults,
        search_space={},
        max_trials=3,
    )
    assert len(trials) == 3
    # Early termination should clip BEFORE moving on to xgboost if first
    # model already exceeds the budget — we expect only random_forest in
    # first 3 trials because RF yields 3 × 3 × 2 = 18 combos.
    assert {t["model_type"] for t in trials} == {"random_forest"}


def test_grid_search_multi_model_continues_across_models():
    defaults = load_tuning_spaces("classification")
    trials = svc._expand_grid_search(
        selected_models=["ridge" if False else "logistic_regression", "random_forest"],
        tuning_defaults=defaults,
        search_space={
            "logistic_regression": {"C": [0.1, 1.0]},
            "random_forest": {"n_estimators": [100]},
        },
        max_trials=None,
    )
    model_sequence = [t["model_type"] for t in trials]
    # 2 logreg trials then 1 rf trial
    assert model_sequence == ["logistic_regression", "logistic_regression", "random_forest"]


# ---------------------------------------------------------------------------
# baseline expansion
# ---------------------------------------------------------------------------

def test_baseline_emits_one_trial_per_model():
    defaults = load_tuning_spaces("classification")
    trials = svc._expand_baseline(
        selected_models=["logistic_regression", "random_forest", "xgboost"],
        tuning_defaults=defaults,
        overrides=None,
    )
    assert len(trials) == 3
    assert [t["trial_no"] for t in trials] == [1, 2, 3]
    assert all(t["search_meta"]["strategy"] == "baseline" for t in trials)


def test_baseline_overrides_merge_with_fixed():
    defaults = load_tuning_spaces("classification")
    trials = svc._expand_baseline(
        selected_models=["logistic_regression"],
        tuning_defaults=defaults,
        overrides={"logistic_regression": {"C": 0.5}},
    )
    params = trials[0]["hyperparameters"]
    assert params["C"] == 0.5
    # fixed.max_iter = 500 must still be present
    assert params["max_iter"] == 500


# ---------------------------------------------------------------------------
# Bayesian count + sampling
# ---------------------------------------------------------------------------

def test_count_bayesian_trials_respects_budget():
    assert svc._count_bayesian_trials(
        selected_models=["a", "b", "c"],
        budget_config={"n_trials_per_model": 5},
        max_trials=None,
    ) == 15
    assert svc._count_bayesian_trials(
        selected_models=["a", "b"],
        budget_config={"n_trials_per_model": 10},
        max_trials=12,
    ) == 12


def test_sample_from_distribution_honours_types():
    """Distribution spec → Optuna suggest_* round-trip."""
    study = optuna.create_study(direction="maximize")
    trial = study.ask()

    params = svc._sample_from_distribution(
        trial,
        {
            "lr": {"type": "float", "low": 0.001, "high": 1.0, "log": True},
            "n_estimators": {"type": "int", "low": 10, "high": 200, "step": 10},
            "kernel": {"type": "categorical", "choices": ["rbf", "linear"]},
        },
    )
    assert 0.001 <= params["lr"] <= 1.0
    assert 10 <= params["n_estimators"] <= 200
    assert params["n_estimators"] % 10 == 0
    assert params["kernel"] in ("rbf", "linear")


def test_sample_from_distribution_handles_list_choices():
    """List-valued choices (e.g. hidden_layer_sizes: [[64], [128]]) are indexed."""
    study = optuna.create_study(direction="maximize")
    trial = study.ask()
    params = svc._sample_from_distribution(
        trial,
        {"hidden_layer_sizes": {"type": "categorical", "choices": [[64], [128, 64]]}},
    )
    assert params["hidden_layer_sizes"] in ([64], [128, 64])


def test_sample_from_distribution_rejects_bad_type():
    study = optuna.create_study(direction="maximize")
    trial = study.ask()
    with pytest.raises(ValueError):
        svc._sample_from_distribution(trial, {"x": {"type": "uniform", "low": 0, "high": 1}})


# ---------------------------------------------------------------------------
# Dispatch entry-point validation
# ---------------------------------------------------------------------------

async def test_dispatch_requires_dataset_and_target(db):
    from fastapi import HTTPException
    from app.models.database import Dataset, ModelingTask

    ds = Dataset(name="x", file_path="/tmp/x", file_size=1, row_count=10)
    db.add(ds)
    await db.flush()

    # Modeling task WITHOUT dataset/target — dispatch should 400
    task = ModelingTask(name="bare", task_type="classification")
    db.add(task)
    await db.flush()

    with pytest.raises(HTTPException) as exc:
        await svc.dispatch_experiment_batch(
            db,
            modeling_task_id=task.id,
            name="batch",
            strategy_type="baseline",
            selected_models=["random_forest"],
            search_space={},
            budget_config={},
        )
    assert exc.value.status_code == 400


async def test_dispatch_rejects_unknown_model(db):
    from fastapi import HTTPException
    from app.models.database import Dataset, ModelingTask

    ds = Dataset(name="x", file_path="/tmp/x", file_size=1, row_count=10)
    db.add(ds)
    await db.flush()

    task = ModelingTask(
        name="t",
        dataset_id=ds.id,
        target_column="y",
        task_type="classification",
    )
    db.add(task)
    await db.flush()

    with pytest.raises(HTTPException) as exc:
        await svc.dispatch_experiment_batch(
            db,
            modeling_task_id=task.id,
            name="batch",
            strategy_type="baseline",
            selected_models=["not_a_real_model"],
            search_space={},
            budget_config={},
        )
    assert exc.value.status_code == 422

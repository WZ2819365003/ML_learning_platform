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
from unittest.mock import AsyncMock

from app.services import training_service as training_svc
from app.services import tuning_service as svc
from app.services.modeling_task_service import load_tuning_spaces


def test_tuning_spaces_cover_all_classical_ml_trainers():
    """Every classical ML trainer should be selectable by V3 tuning strategies."""
    from app.core.trainer import list_available_models

    tuning_models = set(load_tuning_spaces("classification")) | set(load_tuning_spaces("regression"))

    assert set(list_available_models()) <= tuning_models


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


def test_baseline_preserves_deep_learning_family_and_nested_configs():
    defaults = load_tuning_spaces("classification")
    trials = svc._expand_baseline(
        selected_models=["mlp_dl"],
        tuning_defaults=defaults,
        overrides=None,
    )
    assert len(trials) == 1
    assert trials[0]["family"] == "dl"
    params = trials[0]["hyperparameters"]
    assert "arch_config" in params
    assert "opt_config" in params
    assert "train_config" in params
    # Baseline used to force epochs down to 10, so a submitted 50 trained for 10
    # and nothing in the UI or the stored config said so. It now keeps the
    # registry default; only the registry bound applies (see the tests below).
    assert params["train_config"]["epochs"] == 50


def test_baseline_honours_submitted_epochs_instead_of_capping_them():
    """A submitted epochs value survives into the persisted train_config.

    Regression guard for the silent `epochs > 10 -> 10` baseline override: it
    made a 50-epoch request finish in 10 epochs while reporting `10/10`, which
    read as a normal completed run rather than a truncated one.
    """
    defaults = load_tuning_spaces("classification")
    trials = svc._expand_baseline(
        selected_models=["mlp_dl"],
        tuning_defaults=defaults,
        overrides={"mlp_dl": {"train_config": {"epochs": 50}}},
    )
    assert trials[0]["hyperparameters"]["train_config"]["epochs"] == 50


def test_baseline_clamps_epochs_to_the_registry_maximum():
    """Values above the registry bound are clamped, not accepted verbatim.

    The config form bounds this input from the same registry, but a direct API
    call or a stale frontend bypasses the form — so the bound is enforced
    server-side too.
    """
    defaults = load_tuning_spaces("classification")
    trials = svc._expand_baseline(
        selected_models=["mlp_dl"],
        tuning_defaults=defaults,
        overrides={"mlp_dl": {"train_config": {"epochs": 5000}}},
    )
    assert trials[0]["hyperparameters"]["train_config"]["epochs"] == 100


async def test_persist_trials_marks_ml_and_dl_runs_as_selection_mode(db):
    from sqlalchemy import select
    from app.models.database import Dataset, ExperimentRun, ModelingTask, PlatformExperiment

    ds = Dataset(name="sealed.csv", file_path="/tmp/sealed.csv", file_size=1, row_count=20)
    db.add(ds)
    await db.flush()
    task = ModelingTask(
        name="sealed-task",
        dataset_id=ds.id,
        target_column="label",
        task_type="classification",
        objective_metric="accuracy",
    )
    db.add(task)
    await db.flush()
    exp = PlatformExperiment(
        modeling_task_id=task.id,
        name="sealed-batch",
        strategy_type="baseline",
        dataset_id=ds.id,
        objective_metric="accuracy",
    )
    db.add(exp)
    await db.flush()

    await svc._persist_trials(
        db,
        exp,
        task,
        [
            {
                "family": "ml",
                "model_type": "logistic_regression",
                "hyperparameters": {},
                "trial_no": 1,
                "search_meta": {"strategy": "baseline"},
            },
            {
                "family": "dl",
                "model_type": "mlp_dl",
                "hyperparameters": {
                    "arch_config": {}, "opt_config": {}, "train_config": {}
                },
                "trial_no": 2,
                "search_meta": {"strategy": "baseline", "family": "dl"},
            },
        ],
        ["accuracy"],
    )

    runs = (
        await db.execute(select(ExperimentRun).order_by(ExperimentRun.trial_no))
    ).scalars().all()
    # B1: DL runs are selection-mode peers of ML — both seal the hold-out.
    assert runs[0].search_meta["evaluation_mode"] == "selection"
    assert runs[1].search_meta["evaluation_mode"] == "selection"
    assert await training_svc._resolve_evaluation_mode(db, runs[0].task_id) == "selection"
    assert await training_svc._resolve_evaluation_mode(db, runs[1].task_id) == "selection"


async def test_finalise_batch_never_opens_sealed_holdout(
    session_factory, monkeypatch
):
    from app.models.database import ExperimentRun, ModelingTask, PlatformExperiment

    async with session_factory() as db:
        task = ModelingTask(
            name="finalise-task",
            task_type="classification",
            objective_metric="accuracy",
            objective_direction="max",
        )
        db.add(task)
        await db.flush()
        exp = PlatformExperiment(
            modeling_task_id=task.id,
            name="done-batch",
            status="RUNNING",
            objective_metric="accuracy",
            objective_direction="max",
        )
        db.add(exp)
        await db.flush()
        db.add(
            ExperimentRun(
                experiment_id=exp.id,
                status="SUCCESS",
                metrics={"selection_cv_mean_accuracy": 0.9},
            )
        )
        await db.commit()
        task_id, exp_id = task.id, exp.id

    calls = []

    async def fake_refresh(db, modeling_task_id):
        calls.append(("summary", modeling_task_id))

    async def fake_shap(experiment_id, top_k):
        calls.append(("shap", experiment_id))

    async def record_forbidden_evaluation(db, modeling_task_id):
        calls.append(("final", modeling_task_id))
        return {"status": "skipped"}

    monkeypatch.setattr(svc, "async_session_factory", session_factory)
    monkeypatch.setattr(svc, "refresh_task_summary", fake_refresh)
    monkeypatch.setattr(
        svc, "evaluate_task_winner", record_forbidden_evaluation, raising=False
    )
    monkeypatch.setattr(svc, "_schedule_shap_for_top_runs", fake_shap)

    await svc._finalise_batch(exp_id, task_id)
    await svc._finalise_batch(exp_id, task_id)

    assert calls.count(("summary", task_id)) == 2
    assert calls.count(("shap", exp_id)) == 1


async def test_schedule_shap_includes_successful_dl_runs(
    session_factory, monkeypatch
):
    from sqlalchemy import select
    from app.models.database import ExperimentRun, PlatformExperiment, PlatformTask

    async with session_factory() as db:
        experiment = PlatformExperiment(
            name="mixed-shap",
            status="DONE",
            objective_metric="accuracy",
            objective_direction="max",
        )
        db.add(experiment)
        await db.flush()
        dl_task = PlatformTask(
            kind="dl_train",
            status="SUCCESS",
            payload_ref="dl_train:domain-dl-id",
        )
        db.add(dl_task)
        await db.flush()
        run = ExperimentRun(
            experiment_id=experiment.id,
            task_id=dl_task.id,
            params={"family": "dl", "model_type": "mlp_dl"},
            metrics={"accuracy": 0.9},
            status="SUCCESS",
            trial_no=1,
        )
        db.add(run)
        await db.commit()
        experiment_id = experiment.id
        run_id = run.id

    dispatched = []

    class FakeScheduler:
        async def submit(self, task_id):
            dispatched.append(task_id)

    monkeypatch.setattr(svc, "async_session_factory", session_factory)
    monkeypatch.setattr(svc, "get_scheduler", lambda kind: FakeScheduler())

    await svc._schedule_shap_for_top_runs(experiment_id, top_k=1)

    async with session_factory() as db:
        explain_tasks = (
            await db.execute(select(PlatformTask).where(PlatformTask.kind == "explain"))
        ).scalars().all()
    assert [task.payload_ref for task in explain_tasks] == [f"explain:{run_id}"]
    assert dispatched == [explain_tasks[0].id]


@pytest.mark.parametrize("state", ["EVALUATING", "FINALIZED", "FAILED"])
async def test_experiment_dispatch_lock_rejects_non_open_task(db, state):
    from fastapi import HTTPException
    from app.models.database import ModelingTask

    task = ModelingTask(
        name=f"sealed-{state.lower()}",
        task_type="classification",
        objective_metric="accuracy",
        config={
            "_final_evaluation": {
                "state": state,
                "version": 1,
            }
        },
    )
    db.add(task)
    await db.flush()

    with pytest.raises(HTTPException) as exc:
        await svc._lock_task_for_experiment_dispatch(db, task.id)

    assert exc.value.status_code == 409
    assert "最终确认" in exc.value.detail


async def test_experiment_dispatch_lock_allows_open_task(db):
    from app.models.database import ModelingTask

    task = ModelingTask(
        name="open-task",
        task_type="classification",
        objective_metric="accuracy",
    )
    db.add(task)
    await db.flush()

    locked = await svc._lock_task_for_experiment_dispatch(db, task.id)

    assert locked.id == task.id


async def test_dispatch_experiment_batch_uses_task_finalization_lock(db):
    from fastapi import HTTPException
    from app.models.database import ModelingTask

    task = ModelingTask(
        name="finalized-dispatch",
        task_type="classification",
        objective_metric="accuracy",
        config={
            "_final_evaluation": {
                "state": "FINALIZED",
                "version": 1,
            }
        },
    )
    db.add(task)
    await db.flush()

    with pytest.raises(HTTPException) as exc:
        await svc.dispatch_experiment_batch(
            db,
            modeling_task_id=task.id,
            name="must-not-start",
            strategy_type="baseline",
            selected_models=["logistic_regression"],
            search_space={},
            budget_config={},
        )

    assert exc.value.status_code == 409


async def test_dispatch_experiment_bundle_creates_one_batch_per_strategy(db, monkeypatch):
    """A multi-strategy submission should persist separate experiment batches."""
    from app.models.database import Dataset, ModelingTask, PlatformExperiment
    from sqlalchemy import select

    monkeypatch.setattr(svc, "_launch_concurrent", AsyncMock())
    monkeypatch.setattr(svc, "_launch_bayesian", lambda *args, **kwargs: None)

    ds = Dataset(name="demo.csv", file_path="/tmp/demo.csv", file_size=1, row_count=20)
    db.add(ds)
    await db.flush()

    task = ModelingTask(
        name="multi-strategy-task",
        dataset_id=ds.id,
        dataset_name=ds.name,
        target_column="Target",
        task_type="classification",
        objective_metric="accuracy",
        objective_direction="max",
    )
    db.add(task)
    await db.flush()

    result = await svc.dispatch_experiment_bundle(
        db,
        modeling_task_id=task.id,
        name="full-search",
        strategies=[
            {
                "strategy_type": "baseline",
                "selected_models": ["random_forest", "mlp_dl"],
                "search_space": {},
                "budget_config": {"max_trials": 2},
            },
            {
                "strategy_type": "grid_search",
                "selected_models": ["random_forest"],
                "search_space": {"random_forest": {"n_estimators": [10], "max_depth": [3]}},
                "budget_config": {"max_trials": 1},
            },
            {
                "strategy_type": "bayesian_search",
                "selected_models": ["random_forest"],
                "search_space": {
                    "random_forest": {
                        "n_estimators": {"type": "int", "low": 10, "high": 20, "step": 10}
                    }
                },
                "budget_config": {"max_trials": 1, "n_trials_per_model": 1},
            },
        ],
    )

    assert result["batch_count"] == 3
    assert result["strategy_types"] == ["baseline", "grid_search", "bayesian_search"]

    rows = await db.execute(
        select(PlatformExperiment)
        .where(PlatformExperiment.modeling_task_id == task.id)
        .order_by(PlatformExperiment.created_at.asc())
    )
    experiments = rows.scalars().all()
    assert [e.strategy_type for e in experiments] == [
        "baseline",
        "grid_search",
        "bayesian_search",
    ]
    assert experiments[0].selected_models == ["random_forest", "mlp_dl"]
    assert experiments[1].selected_models == ["random_forest"]


async def test_dispatch_persists_cv_folds_on_training_tasks(db, monkeypatch):
    """Budget cv_folds must propagate to the concrete TrainingTask executor."""
    from app.models.database import Dataset, ModelingTask, TrainingTask
    from sqlalchemy import select

    monkeypatch.setattr(svc, "_launch_concurrent", AsyncMock())

    ds = Dataset(name="demo.csv", file_path="/tmp/demo.csv", file_size=1, row_count=20)
    db.add(ds)
    await db.flush()
    task = ModelingTask(
        name="cv-task",
        dataset_id=ds.id,
        dataset_name=ds.name,
        target_column="Target",
        task_type="classification",
        objective_metric="accuracy",
        objective_direction="max",
    )
    db.add(task)
    await db.flush()

    await svc.dispatch_experiment_batch(
        db,
        modeling_task_id=task.id,
        name="cv-grid",
        strategy_type="grid_search",
        selected_models=["logistic_regression"],
        search_space={"logistic_regression": {"C": [0.1]}},
        budget_config={"cv_folds": 7, "max_trials": 1},
    )

    rows = await db.execute(select(TrainingTask).where(TrainingTask.dataset_id == ds.id))
    training_task = rows.scalar_one()
    assert training_task.cv_folds == 7


async def test_search_strategy_rejects_mixed_dl_models_instead_of_dropping_them(db, monkeypatch):
    """Grid/Bayesian do not support DL yet, so mixed batches must fail loudly."""
    from fastapi import HTTPException
    from app.models.database import Dataset, ModelingTask

    monkeypatch.setattr(svc, "_launch_concurrent", AsyncMock())

    ds = Dataset(name="demo.csv", file_path="/tmp/demo.csv", file_size=1, row_count=20)
    db.add(ds)
    await db.flush()
    task = ModelingTask(
        name="mixed-search-task",
        dataset_id=ds.id,
        dataset_name=ds.name,
        target_column="Target",
        task_type="classification",
        objective_metric="accuracy",
        objective_direction="max",
    )
    db.add(task)
    await db.flush()

    with pytest.raises(HTTPException) as exc:
        await svc.dispatch_experiment_batch(
            db,
            modeling_task_id=task.id,
            name="mixed-grid",
            strategy_type="grid_search",
            selected_models=["random_forest", "mlp_dl"],
            search_space={"random_forest": {"n_estimators": [10]}},
            budget_config={"max_trials": 1},
        )

    assert exc.value.status_code == 422
    assert "Deep-learning models" in exc.value.detail


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


async def test_bayesian_worker_loads_modeling_task(db, session_factory, monkeypatch):
    """The background worker must not depend on an unimported route helper."""
    from app.models.database import Dataset, ModelingTask, PlatformExperiment

    ds = Dataset(name="bayes.csv", file_path="/tmp/bayes.csv", file_size=1, row_count=20)
    db.add(ds)
    await db.flush()
    task = ModelingTask(
        name="bayes-task",
        dataset_id=ds.id,
        target_column="Target",
        task_type="classification",
        objective_metric="accuracy",
        objective_direction="max",
    )
    db.add(task)
    await db.flush()
    exp = PlatformExperiment(
        name="bayes-exp",
        modeling_task_id=task.id,
        strategy_type="bayesian_search",
        status="RUNNING",
    )
    db.add(exp)
    await db.commit()

    async def no_finalise(*_args, **_kwargs):
        return None

    monkeypatch.setattr(svc, "async_session_factory", session_factory)
    monkeypatch.setattr(svc, "_finalise_batch", no_finalise)
    await svc._run_bayesian_search(
        experiment_id=exp.id,
        modeling_task_id=task.id,
        selected_models=[],
        search_space={},
        tuning_defaults={},
        budget_config={},
        eval_metrics=["accuracy"],
    )


async def test_bayesian_guard_marks_startup_failure(db, session_factory, monkeypatch):
    """An uncaught background error must not strand the experiment in RUNNING."""
    from app.models.database import Dataset, ModelingTask, PlatformExperiment

    ds = Dataset(name="bayes.csv", file_path="/tmp/bayes.csv", file_size=1, row_count=20)
    db.add(ds)
    await db.flush()
    task = ModelingTask(name="bayes-task", dataset_id=ds.id, status="RUNNING")
    db.add(task)
    await db.flush()
    exp = PlatformExperiment(
        name="bayes-exp",
        modeling_task_id=task.id,
        strategy_type="bayesian_search",
        status="RUNNING",
        config={"submitted_from": "test"},
    )
    db.add(exp)
    await db.commit()

    async def fail_worker(**_kwargs):
        raise RuntimeError("worker startup failed")

    monkeypatch.setattr(svc, "async_session_factory", session_factory)
    monkeypatch.setattr(svc, "_run_bayesian_search", fail_worker)
    await svc._run_bayesian_search_guarded(
        experiment_id=exp.id,
        modeling_task_id=task.id,
    )

    await db.refresh(exp)
    assert exp.status == "FAILED"
    assert exp.finished_at is not None
    assert exp.config["worker_error"] == "worker startup failed"


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


def test_progress_fraction_uses_platform_task_scale():
    """PlatformTask.progress uses 0..1 while legacy domain rows often use 0..100."""
    assert training_svc._progress_fraction(1, 5) == pytest.approx(0.2)
    assert training_svc._progress_fraction(5, 5) == pytest.approx(1.0)
    assert training_svc._progress_fraction(0, 5) == pytest.approx(0.0)
    assert training_svc._progress_fraction(3, 0) == pytest.approx(0.0)


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


# ---------------------------------------------------------------------------
# V3 Phase 1 — DL trial expansion
# ---------------------------------------------------------------------------

def test_expand_dl_baseline_uses_registry_defaults_when_config_missing():
    # V3.1.1 canonical key convention: DL trials expose ``arch_config`` /
    # ``opt_config`` / ``train_config`` (matches ``_create_dl_training_task_record``
    # + the UI's ``DLConfigPanel`` form shape).
    from app.core.dl_registry import DL_MODEL_REGISTRY
    dl_token = next(m["id"] for m in DL_MODEL_REGISTRY
                    if "classification" in m.get("task_types", []))
    trials = svc._expand_dl_baseline(
        dl_models=[dl_token],
        dl_config={},
        task_type="classification",
    )
    assert len(trials) == 1
    t = trials[0]
    assert t["model_type"] == dl_token
    assert t["family"] == "dl"
    assert t["search_meta"]["family"] == "dl"
    hp = t["hyperparameters"]
    # Registry defaults must fill all three sections
    assert set(hp.keys()) == {"arch_config", "opt_config", "train_config"}
    assert hp["train_config"].get("epochs") is not None


def test_expand_dl_baseline_merges_partial_override():
    from app.core.dl_registry import DL_MODEL_REGISTRY
    dl_token = next(m["id"] for m in DL_MODEL_REGISTRY
                    if "classification" in m.get("task_types", []))
    trials = svc._expand_dl_baseline(
        dl_models=[dl_token],
        dl_config={dl_token: {"train_config": {"epochs": 3}}},
        task_type="classification",
    )
    hp = trials[0]["hyperparameters"]
    # Override survives
    assert hp["train_config"]["epochs"] == 3
    # Other train defaults still present (merged, not replaced)
    assert len(hp["train_config"]) > 1


def test_renumber_trials_produces_dense_indices():
    trials = [
        {"model_type": "a", "hyperparameters": {}, "trial_no": 1},
        {"model_type": "b", "hyperparameters": {}, "trial_no": 2},
    ]
    out = svc._renumber_trials(trials, start=5)
    assert [t["trial_no"] for t in out] == [5, 6]
    # Non-destructive
    assert [t["trial_no"] for t in trials] == [1, 2]


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

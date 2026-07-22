"""M3-3 — AutoML as a strategy of the V3 batch pipeline.

The previous implementation ran its own dispatch loop and wrote the Run and the
PlatformTask in two separate steps. That left it outside every guarantee the
platform had built: no ``evaluation_mode``, so its results could not be
compared with anything or promoted to a final evaluation; and no M2c write-back,
so a duplicate delivery could split the two records.

These tests pin the properties that fix required.
"""
from __future__ import annotations

from unittest.mock import patch

import pytest
from fastapi import HTTPException
from sqlalchemy import select

from app.models.database import (
    Dataset,
    ExperimentRun,
    ModelingTask,
    PlatformExperiment,
)
from app.services import tuning_service


@pytest.fixture(autouse=True)
def use_test_sessions(session_factory):
    with patch("app.models.database.async_session_factory", session_factory), \
            patch.object(tuning_service, "async_session_factory", session_factory):
        yield


@pytest.fixture(autouse=True)
def dont_launch(monkeypatch):
    """Persist trials but never run them — this is about what gets created."""
    async def noop(*a, **k):
        return None

    monkeypatch.setattr(tuning_service, "_launch_concurrent", noop)


async def _task(db, task_type="classification", metric="accuracy"):
    ds = Dataset(name="a.csv", file_path="/tmp/a.csv", file_size=1, row_count=100)
    db.add(ds)
    await db.flush()
    t = ModelingTask(
        name="automl-task", dataset_id=ds.id, dataset_name=ds.name,
        target_column="y", task_type=task_type, objective_metric=metric,
    )
    db.add(t)
    await db.flush()
    await db.commit()
    return t.id


async def _dispatch(db, task_id, **kwargs):
    return await tuning_service.dispatch_experiment_batch(
        db,
        modeling_task_id=task_id,
        name="AutoML",
        strategy_type="automl",
        selected_models=[],
        search_space={},
        budget_config=kwargs.pop("budget_config", {}),
        **kwargs,
    )


# ---------------------------------------------------------------------------
# The property the rewrite exists for
# ---------------------------------------------------------------------------

async def test_automl_runs_carry_selection_semantics(db, session_factory):
    """Without evaluation_mode an AutoML result is un-comparable and can never
    be promoted to a final evaluation — the whole reason it was rewritten."""
    task_id = await _task(db)
    result = await _dispatch(db, task_id)

    async with session_factory() as s:
        runs = (await s.execute(
            select(ExperimentRun).where(ExperimentRun.experiment_id == result["experiment"]["id"])
        )).scalars().all()

    assert runs, "no runs were persisted"
    for run in runs:
        # evaluation_mode lives on search_meta, which is what
        # training_service._resolve_evaluation_mode reads to seal the hold-out.
        assert (run.search_meta or {}).get("evaluation_mode") == "selection", (
            f"run {run.trial_no} has no selection semantics: {run.search_meta}"
        )


async def test_every_candidate_becomes_its_own_trial(db, session_factory):
    """The registry lists the same model_type several times with different
    hyperparameters. Collapsing by model_type would silently halve the search,
    which is what routing AutoML through `baseline` would have done."""
    from app.services.automl_service import load_candidates

    candidates = load_candidates("classification")
    duplicated = len(candidates) - len({c["model_type"] for c in candidates})
    assert duplicated > 0, "fixture assumption broken: registry has no duplicate model_types"

    task_id = await _task(db)
    result = await _dispatch(db, task_id)

    assert result["trials_planned"] == len(candidates), (
        f"expected one trial per candidate ({len(candidates)}), "
        f"got {result['trials_planned']} — duplicates were collapsed"
    )

    async with session_factory() as s:
        runs = (await s.execute(
            select(ExperimentRun).where(ExperimentRun.experiment_id == result["experiment"]["id"])
        )).scalars().all()
    assert len(runs) == len(candidates)

    # The duplicated model_types must differ in their hyperparameters.
    by_model: dict[str, list[dict]] = {}
    for run in runs:
        by_model.setdefault((run.params or {}).get("model_type"), []).append(
            (run.params or {}).get("hyperparameters") or {}
        )
    repeated = {m: hs for m, hs in by_model.items() if len(hs) > 1}
    assert repeated, "duplicate candidates did not survive as separate runs"
    for model, hyper_sets in repeated.items():
        assert len({str(sorted(h.items())) for h in hyper_sets}) == len(hyper_sets), (
            f"{model} produced identical trials — candidate hyperparameters were lost"
        )


async def test_trials_record_which_candidate_they_came_from(db, session_factory):
    task_id = await _task(db)
    result = await _dispatch(db, task_id)

    async with session_factory() as s:
        runs = (await s.execute(
            select(ExperimentRun).where(ExperimentRun.experiment_id == result["experiment"]["id"])
        )).scalars().all()

    metas = [r.search_meta or {} for r in runs]
    assert all(m.get("strategy") == "automl" for m in metas)
    assert all(m.get("candidate_index") for m in metas)


# ---------------------------------------------------------------------------
# Pipeline integration
# ---------------------------------------------------------------------------

async def test_experiment_is_a_normal_v3_experiment(db, session_factory):
    """It must land on the leaderboard and in the report like any other batch."""
    task_id = await _task(db)
    result = await _dispatch(db, task_id)

    async with session_factory() as s:
        exp = (await s.execute(
            select(PlatformExperiment).where(PlatformExperiment.id == result["experiment"]["id"])
        )).scalar_one()
        task = (await s.execute(select(ModelingTask).where(ModelingTask.id == task_id))).scalar_one()

    assert exp.strategy_type == "automl"
    assert exp.modeling_task_id == task_id
    assert exp.status == "RUNNING"
    assert task.status == "RUNNING", "the modeling task was not moved to RUNNING"


async def test_max_trials_caps_the_sweep(db):
    task_id = await _task(db)
    result = await _dispatch(db, task_id, budget_config={"max_trials": 3})
    assert result["trials_planned"] == 3


async def test_regression_task_uses_regression_candidates(db, session_factory):
    task_id = await _task(db, task_type="regression", metric="rmse")
    result = await _dispatch(db, task_id)

    from app.services.automl_service import load_candidates

    assert result["trials_planned"] == len(load_candidates("regression"))

    async with session_factory() as s:
        runs = (await s.execute(
            select(ExperimentRun).where(ExperimentRun.experiment_id == result["experiment"]["id"])
        )).scalars().all()
    models = {(r.params or {}).get("model_type") for r in runs}
    assert models, "no runs"
    assert not models & {"logistic_regression", "svm"}, (
        f"classification models leaked into a regression sweep: {models}"
    )


async def test_unknown_task_type_is_rejected_before_anything_is_created(db, session_factory):
    task_id = await _task(db, task_type="clustering", metric="accuracy")
    with pytest.raises(HTTPException) as exc:
        await _dispatch(db, task_id)
    assert exc.value.status_code == 422

    async with session_factory() as s:
        exps = (await s.execute(select(PlatformExperiment))).scalars().all()
    assert exps == [], "a rejected AutoML sweep still created an experiment"

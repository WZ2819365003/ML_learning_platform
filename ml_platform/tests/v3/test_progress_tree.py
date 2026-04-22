"""
Phase 4 — progress-tree aggregation tests.

Covers:
    * 404 for unknown modeling_task
    * Empty tree (task with no experiments yet)
    * Baseline experiment with ML + DL mixed runs
    * Aggregate math: per-experiment avg, per-task avg
    * current_step derivation — DL epoch counter vs ML progress pct
    * has_active_runs flag flips with run status
    * Terminal runs (FAILED / CANCELLED) count as done
"""
from __future__ import annotations

import pytest
from fastapi import HTTPException

from app.models.database import (
    DLTrainingTask,
    Dataset,
    ExperimentRun,
    ModelingTask,
    PlatformExperiment,
    PlatformTask,
    TrainingTask,
)
from app.services import progress_tree_service as svc


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@pytest.fixture
async def dataset(db):
    ds = Dataset(name="pt_ds", file_path="/tmp/pt.csv", file_size=0, row_count=10)
    db.add(ds)
    await db.flush()
    return ds


@pytest.fixture
async def modeling_task(db, dataset):
    mt = ModelingTask(
        name="progress-tree-fixture",
        dataset_id=dataset.id,
        target_column="y",
        task_type="classification",
        status="RUNNING",
    )
    db.add(mt)
    await db.flush()
    return mt


async def _mk_exp(db, modeling_task_id: str, **kwargs) -> PlatformExperiment:
    defaults = dict(name="exp", strategy_type="baseline", status="RUNNING")
    defaults.update(kwargs)
    exp = PlatformExperiment(modeling_task_id=modeling_task_id, **defaults)
    db.add(exp)
    await db.flush()
    return exp


async def _mk_ptask(db, kind: str, payload_ref: str, progress: float = 0.0,
                   status: str = "RUNNING") -> PlatformTask:
    t = PlatformTask(kind=kind, payload_ref=payload_ref, progress=progress, status=status)
    db.add(t)
    await db.flush()
    return t


async def _mk_run(db, experiment_id: str, task_id: str | None = None,
                 status: str = "RUNNING", model_type: str | None = None,
                 trial_no: int = 1) -> ExperimentRun:
    params = {"model_type": model_type} if model_type else None
    r = ExperimentRun(
        experiment_id=experiment_id,
        task_id=task_id,
        status=status,
        trial_no=trial_no,
        params=params,
    )
    db.add(r)
    await db.flush()
    return r


# ---------------------------------------------------------------------------
# Error paths + empty tree
# ---------------------------------------------------------------------------

async def test_progress_tree_404_for_unknown(db):
    with pytest.raises(HTTPException) as exc:
        await svc.get_progress_tree(db, "missing")
    assert exc.value.status_code == 404


async def test_progress_tree_empty(db, modeling_task):
    tree = await svc.get_progress_tree(db, modeling_task.id)
    assert tree["modeling_task"]["id"] == modeling_task.id
    assert tree["modeling_task"]["progress_aggregated"] == 0.0
    assert tree["experiments"] == []
    assert tree["has_active_runs"] is False


# ---------------------------------------------------------------------------
# ML-only baseline
# ---------------------------------------------------------------------------

async def test_progress_tree_aggregates_ml_runs(db, dataset, modeling_task):
    exp = await _mk_exp(db, modeling_task.id, selected_models=["random_forest", "xgboost"])

    # Two ML runs: one complete, one at 40% progress
    tt1 = TrainingTask(dataset_id=dataset.id, model_type="random_forest",
                       target_column="y", status="SUCCESS", progress=1.0)
    tt2 = TrainingTask(dataset_id=dataset.id, model_type="xgboost",
                       target_column="y", status="RUNNING", progress=0.4)
    db.add_all([tt1, tt2])
    await db.flush()

    p1 = await _mk_ptask(db, "train", f"train:{tt1.id}", progress=1.0, status="SUCCESS")
    p2 = await _mk_ptask(db, "train", f"train:{tt2.id}", progress=0.4, status="RUNNING")

    await _mk_run(db, exp.id, p1.id, status="SUCCESS", model_type="random_forest", trial_no=1)
    await _mk_run(db, exp.id, p2.id, status="RUNNING", model_type="xgboost", trial_no=2)

    tree = await svc.get_progress_tree(db, modeling_task.id)

    assert len(tree["experiments"]) == 1
    exp_payload = tree["experiments"][0]
    assert exp_payload["run_count"] == 2
    # (1.0 + 0.4) / 2 = 0.7
    assert exp_payload["progress_aggregated"] == 0.7
    assert tree["modeling_task"]["progress_aggregated"] == 0.7

    # Every run has family=ml and a model_type surfaced
    families = {r["family"] for r in exp_payload["runs"]}
    assert families == {"ml"}
    assert tree["has_active_runs"] is True


# ---------------------------------------------------------------------------
# DL + ML mixed experiment
# ---------------------------------------------------------------------------

async def test_progress_tree_mixes_ml_and_dl(db, dataset, modeling_task):
    exp = await _mk_exp(db, modeling_task.id, selected_models=["xgboost", "mlp_dl"])

    # 1 ML run at 100%
    tt_ml = TrainingTask(dataset_id=dataset.id, model_type="xgboost",
                         target_column="y", status="SUCCESS", progress=1.0)
    # 1 DL run at epoch 10/50 → 0.2
    tt_dl = DLTrainingTask(
        dataset_id=dataset.id, target_column="y", model_type="mlp_dl",
        status="RUNNING", progress=0.2, current_epoch=10, total_epochs=50,
    )
    db.add_all([tt_ml, tt_dl])
    await db.flush()

    p_ml = await _mk_ptask(db, "train",    f"train:{tt_ml.id}",    progress=1.0, status="SUCCESS")
    p_dl = await _mk_ptask(db, "dl_train", f"dl_train:{tt_dl.id}", progress=0.2, status="RUNNING")

    await _mk_run(db, exp.id, p_ml.id, status="SUCCESS", model_type="xgboost", trial_no=1)
    await _mk_run(db, exp.id, p_dl.id, status="RUNNING", model_type="mlp_dl",  trial_no=2)

    tree = await svc.get_progress_tree(db, modeling_task.id)
    runs = tree["experiments"][0]["runs"]

    # Sorted by rank-none then trial_no asc — xgboost first (trial_no=1)
    ml_run = next(r for r in runs if r["family"] == "ml")
    dl_run = next(r for r in runs if r["family"] == "dl")

    assert ml_run["model_type"] == "xgboost"
    assert ml_run["progress"] == 1.0
    assert ml_run["current_step"] == "已完成"

    assert dl_run["model_type"] == "mlp_dl"
    assert dl_run["progress"] == 0.2
    assert dl_run["current_step"] == "epoch 10/50"

    # Aggregate = (1.0 + 0.2) / 2 = 0.6
    assert tree["experiments"][0]["progress_aggregated"] == 0.6
    assert tree["modeling_task"]["progress_aggregated"] == 0.6


# ---------------------------------------------------------------------------
# Terminal runs count as done
# ---------------------------------------------------------------------------

async def test_failed_run_counts_as_done_in_aggregate(db, dataset, modeling_task):
    exp = await _mk_exp(db, modeling_task.id)

    tt = TrainingTask(dataset_id=dataset.id, model_type="random_forest",
                      target_column="y", status="FAILED",
                      progress=0.3, error_message="kaboom")
    db.add(tt)
    await db.flush()
    ptask = await _mk_ptask(db, "train", f"train:{tt.id}", progress=0.3, status="FAILED")
    await _mk_run(db, exp.id, ptask.id, status="FAILED", model_type="random_forest")

    tree = await svc.get_progress_tree(db, modeling_task.id)
    run = tree["experiments"][0]["runs"][0]

    assert run["progress"] == 1.0       # terminal → counted as done for aggregation
    assert run["current_step"] == "失败"
    assert tree["has_active_runs"] is False


# ---------------------------------------------------------------------------
# has_active_runs flag
# ---------------------------------------------------------------------------

async def test_has_active_runs_true_when_queued(db, dataset, modeling_task):
    exp = await _mk_exp(db, modeling_task.id)
    ptask = await _mk_ptask(db, "train", "train:ghost", progress=0.0, status="QUEUED")
    await _mk_run(db, exp.id, ptask.id, status="QUEUED", model_type="random_forest")

    tree = await svc.get_progress_tree(db, modeling_task.id)
    assert tree["has_active_runs"] is True
    assert tree["experiments"][0]["runs"][0]["current_step"] == "等待调度"


# ---------------------------------------------------------------------------
# Multiple experiments — aggregation is across every run, not experiment
# ---------------------------------------------------------------------------

async def test_modeling_task_aggregate_averages_all_runs(db, dataset, modeling_task):
    # Experiment A: 1 run at 1.0
    # Experiment B: 3 runs at 0.0, 0.5, 0.5
    # Per-run avg = (1.0 + 0.0 + 0.5 + 0.5) / 4 = 0.5
    exp_a = await _mk_exp(db, modeling_task.id, name="a")
    exp_b = await _mk_exp(db, modeling_task.id, name="b")

    tt_a = TrainingTask(dataset_id=dataset.id, model_type="m1",
                        target_column="y", status="SUCCESS", progress=1.0)
    db.add(tt_a); await db.flush()
    pa = await _mk_ptask(db, "train", f"train:{tt_a.id}", progress=1.0, status="SUCCESS")
    await _mk_run(db, exp_a.id, pa.id, status="SUCCESS", model_type="m1", trial_no=1)

    for idx, prog in enumerate([0.0, 0.5, 0.5], start=1):
        tt = TrainingTask(dataset_id=dataset.id, model_type=f"m{idx}",
                          target_column="y", status="RUNNING", progress=prog)
        db.add(tt); await db.flush()
        p = await _mk_ptask(db, "train", f"train:{tt.id}", progress=prog, status="RUNNING")
        await _mk_run(db, exp_b.id, p.id, status="RUNNING", model_type=f"m{idx}", trial_no=idx)

    tree = await svc.get_progress_tree(db, modeling_task.id)
    assert tree["modeling_task"]["progress_aggregated"] == 0.5
    # Per-experiment aggregates:
    by_name = {e["name"]: e for e in tree["experiments"]}
    assert by_name["a"]["progress_aggregated"] == 1.0
    assert by_name["b"]["progress_aggregated"] == pytest.approx(1/3, rel=1e-3)

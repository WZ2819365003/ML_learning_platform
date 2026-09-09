"""Tests for the Run Inspector aggregated endpoint."""

from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient

from app.main import create_app
from app.models.database import (
    Dataset,
    ExperimentRun,
    ModelingTask,
    PlatformExperiment,
    PlatformTask,
    TrainingLog,
    TrainingTask,
    get_db,
)


@pytest.fixture
async def inspector_fixtures(db):
    """Seed a realistic 1-task / 1-experiment / 2-run scenario."""
    ds = Dataset(name="iris", file_path="/tmp/iris.csv", file_size=1000, row_count=150)
    db.add(ds)
    await db.flush()

    tt = TrainingTask(
        dataset_id=ds.id, model_type="random_forest", name="rf_a",
        target_column="y", hyperparameters={"n_estimators": 100},
        eval_metrics=["accuracy"], status="SUCCESS", progress=1.0,
        model_path="/tmp/model.joblib",
    )
    db.add(tt)
    await db.flush()

    db.add_all([
        TrainingLog(task_id=tt.id, level="INFO", message="starting fold 1"),
        TrainingLog(task_id=tt.id, level="INFO", message="fold 1 accuracy=0.9"),
    ])

    ptask = PlatformTask(
        kind="train", status="SUCCESS", progress=1.0,
        payload_ref=f"train:{tt.id}",
        metrics_snapshot={"accuracy": 0.9},
    )
    db.add(ptask)
    await db.flush()

    exp = PlatformExperiment(
        name="exp-1",
        strategy_type="grid_search",
        objective_metric="accuracy", objective_direction="max",
        dataset_id=ds.id, status="RUNNING",
    )
    db.add(exp)
    await db.flush()

    run_main = ExperimentRun(
        experiment_id=exp.id, task_id=ptask.id,
        params={"model_type": "random_forest"},
        metrics={"accuracy": 0.9, "shap_importances": {"f1": 0.4, "f2": 0.2}},
        status="SUCCESS",
        trial_no=1, rank=1,
        search_meta={"strategy": "grid_search", "grid_index": 0},
        source_experiment_type="grid_search",
    )
    run_sib = ExperimentRun(
        experiment_id=exp.id,
        params={"model_type": "random_forest"},
        metrics={"accuracy": 0.85},
        status="SUCCESS",
        trial_no=2, rank=2,
        source_experiment_type="grid_search",
    )
    db.add_all([run_main, run_sib])
    await db.commit()

    return {"dataset": ds, "training_task": tt, "platform_task": ptask,
            "experiment": exp, "run_main": run_main, "run_sib": run_sib}


@pytest.fixture
def app_with_db(session_factory):
    """FastAPI app with get_db overridden to use the test session_factory."""
    app = create_app()

    async def _override_get_db():
        async with session_factory() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise

    app.dependency_overrides[get_db] = _override_get_db
    return app


async def test_inspector_returns_full_context(inspector_fixtures, app_with_db):
    run = inspector_fixtures["run_main"]
    async with AsyncClient(transport=ASGITransport(app=app_with_db), base_url="http://test") as client:
        resp = await client.get(f"/api/platform/runs/{run.id}/inspector")
    assert resp.status_code == 200
    data = resp.json()

    # Run block
    assert data["run"]["id"] == run.id
    assert data["run"]["trial_no"] == 1
    assert data["run"]["search_meta"]["strategy"] == "grid_search"

    # Experiment block
    assert data["experiment"]["strategy_type"] == "grid_search"
    assert data["experiment"]["objective_metric"] == "accuracy"

    # Platform task block
    assert data["platform_task"]["status"] == "SUCCESS"
    assert data["platform_task"]["payload_ref"].startswith("train:")

    # Training task + dataset
    assert data["training_task"]["model_type"] == "random_forest"
    assert data["training_task"]["dataset"]["name"] == "iris"

    # Logs oldest-first
    assert len(data["logs"]) == 2
    assert data["logs"][0]["message"] == "starting fold 1"

    # Siblings contain both runs
    sib_ids = {s["id"] for s in data["siblings"]}
    assert sib_ids == {inspector_fixtures["run_main"].id, inspector_fixtures["run_sib"].id}

    # SHAP summary
    assert data["shap"]["has_explanation"] is True
    assert data["shap"]["feature_count"] == 2
    assert data["shap"]["top_features"][0]["feature"] == "f1"


async def test_inspector_404_for_missing_run(app_with_db):
    async with AsyncClient(transport=ASGITransport(app=app_with_db), base_url="http://test") as client:
        resp = await client.get("/api/platform/runs/nonexistent/inspector")
    assert resp.status_code == 404


async def test_inspector_no_shap_when_missing(inspector_fixtures, app_with_db):
    run = inspector_fixtures["run_sib"]  # no shap_importances in metrics
    async with AsyncClient(transport=ASGITransport(app=app_with_db), base_url="http://test") as client:
        resp = await client.get(f"/api/platform/runs/{run.id}/inspector")
    assert resp.status_code == 200
    assert resp.json()["shap"]["has_explanation"] is False


async def test_inspector_log_limit(inspector_fixtures, app_with_db):
    run = inspector_fixtures["run_main"]
    async with AsyncClient(transport=ASGITransport(app=app_with_db), base_url="http://test") as client:
        resp = await client.get(f"/api/platform/runs/{run.id}/inspector", params={"log_limit": 1})
    # Only the latest log is returned (but the endpoint reverses to oldest-first,
    # so with limit=1 we get exactly one entry, the most recent one in DB terms)
    assert resp.status_code == 200
    assert len(resp.json()["logs"]) == 1


async def test_inspector_walks_id_chain_for_logs(db, app_with_db):
    """When run.id != log.task_id, the inspector must find logs via the
    PlatformTask.payload_ref legacy id (V3 runs are keyed this way)."""
    # Synthesize a V3-native run where logs live under the legacy trainer id.
    legacy_id = "legacy-id-xyz"
    ds = Dataset(name="synth", file_path="/tmp/synth.csv", file_size=1)
    db.add(ds)
    await db.flush()

    # No TrainingTask row exists for legacy_id — logs reference it anyway
    # (mirrors the production state where TrainingTask was purged).
    db.add_all([
        TrainingLog(task_id=legacy_id, level="INFO", message="legacy-log-1"),
        TrainingLog(task_id=legacy_id, level="WARNING", message="legacy-log-2"),
    ])

    ptask = PlatformTask(
        kind="train", status="SUCCESS", progress=1.0,
        payload_ref=f"train:{legacy_id}",
    )
    db.add(ptask)
    await db.flush()

    exp = PlatformExperiment(
        name="legacy-exp", strategy_type="baseline",
        objective_metric="accuracy", objective_direction="max",
        dataset_id=ds.id, status="DONE",
    )
    db.add(exp)
    await db.flush()

    run = ExperimentRun(
        experiment_id=exp.id, task_id=ptask.id,
        params={"model_type": "random_forest"},
        metrics={"accuracy": 0.77},
        status="SUCCESS", trial_no=1, rank=1,
        source_experiment_type="baseline",
    )
    db.add(run)
    await db.commit()

    async with AsyncClient(transport=ASGITransport(app=app_with_db), base_url="http://test") as client:
        resp = await client.get(f"/api/platform/runs/{run.id}/inspector")
    assert resp.status_code == 200
    data = resp.json()
    # Logs were found via the payload_ref legacy id even though no TrainingTask
    # row exists.
    assert len(data["logs"]) == 2
    messages = [lg["message"] for lg in data["logs"]]
    assert "legacy-log-1" in messages
    assert "legacy-log-2" in messages
    assert data["log_task_id"] == legacy_id


# ---------------------------------------------------------------------------
# modeling_task block — the drawer's source of truth for target column / rank
# ---------------------------------------------------------------------------


@pytest.fixture
async def modeling_task_fixtures(db):
    """A ModelingTask owning two experiments, so the task leaderboard has to
    rank across experiment boundaries (which `ExperimentRun.rank` cannot).

    Objective is `rmse` / `min`, deliberately *not* the default accuracy/max,
    so a test that accidentally re-implements the ordering instead of reusing
    `task_leaderboard` gets the winner backwards.
    """
    ds = Dataset(
        name="pv_generation.csv", file_path="/tmp/pv.csv", file_size=2048,
        row_count=8760, column_count=12,
    )
    db.add(ds)
    await db.flush()

    mt = ModelingTask(
        name="光伏出力预测",
        dataset_id=ds.id,
        dataset_name=ds.name,
        target_column="ac_power",
        task_type="regression",
        objective_metric="rmse",
        objective_direction="min",
        status="COMPLETED",
    )
    db.add(mt)
    await db.flush()

    exp_a = PlatformExperiment(
        name="baseline", strategy_type="baseline", modeling_task_id=mt.id,
        objective_metric="rmse", objective_direction="min",
        dataset_id=ds.id, status="COMPLETED",
    )
    exp_b = PlatformExperiment(
        name="grid", strategy_type="grid_search", modeling_task_id=mt.id,
        objective_metric="rmse", objective_direction="min",
        dataset_id=ds.id, status="COMPLETED",
    )
    db.add_all([exp_a, exp_b])
    await db.flush()

    # Best rmse (lowest) lives in exp_b — and is rank 2 *within* its own
    # experiment, so task rank and run.rank must disagree.
    run_a1 = ExperimentRun(
        experiment_id=exp_a.id, params={}, metrics={"rmse": 9.0},
        status="SUCCESS", trial_no=1, rank=1, source_experiment_type="baseline",
    )
    run_b1 = ExperimentRun(
        experiment_id=exp_b.id, params={}, metrics={"rmse": 7.5},
        status="SUCCESS", trial_no=1, rank=2, source_experiment_type="grid_search",
    )
    run_b2 = ExperimentRun(
        experiment_id=exp_b.id, params={}, metrics={"rmse": 4.2},
        status="SUCCESS", trial_no=2, rank=1, source_experiment_type="grid_search",
    )
    # Never finished — has no objective value, so it cannot be ranked at all.
    run_pending = ExperimentRun(
        experiment_id=exp_b.id, params={}, metrics={},
        status="RUNNING", trial_no=3, source_experiment_type="grid_search",
    )
    db.add_all([run_a1, run_b1, run_b2, run_pending])
    await db.commit()

    return {
        "dataset": ds, "modeling_task": mt,
        "exp_a": exp_a, "exp_b": exp_b,
        "run_a1": run_a1, "run_b1": run_b1, "run_b2": run_b2,
        "run_pending": run_pending,
    }


async def test_inspector_exposes_modeling_task_contract(
    modeling_task_fixtures, app_with_db
):
    """target_column / task_type / objective / dataset_name come off the
    ModelingTask, not the legacy TrainingTask (which may not even exist)."""
    run = modeling_task_fixtures["run_b2"]
    async with AsyncClient(transport=ASGITransport(app=app_with_db), base_url="http://test") as client:
        resp = await client.get(f"/api/platform/runs/{run.id}/inspector")
    assert resp.status_code == 200
    mt = resp.json()["modeling_task"]

    assert mt is not None
    assert mt["id"] == modeling_task_fixtures["modeling_task"].id
    assert mt["name"] == "光伏出力预测"
    assert mt["target_column"] == "ac_power"
    assert mt["task_type"] == "regression"
    assert mt["objective_metric"] == "rmse"
    assert mt["objective_direction"] == "min"
    assert mt["dataset_name"] == "pv_generation.csv"


async def test_inspector_rank_matches_task_leaderboard(
    modeling_task_fixtures, app_with_db
):
    """The inspector's rank must equal the /leaderboard rank for the same run —
    a second ordering that drifts from the leaderboard is worse than none."""
    task_id = modeling_task_fixtures["modeling_task"].id

    async with AsyncClient(transport=ASGITransport(app=app_with_db), base_url="http://test") as client:
        board_resp = await client.get(f"/api/v3/tasks/{task_id}/leaderboard")
        assert board_resp.status_code == 200
        board = {row["run_id"]: row["rank"] for row in board_resp.json()}

        for key in ("run_a1", "run_b1", "run_b2"):
            run = modeling_task_fixtures[key]
            resp = await client.get(f"/api/platform/runs/{run.id}/inspector")
            assert resp.status_code == 200
            assert resp.json()["modeling_task"]["rank"] == board[run.id], key

    # Sanity: the ordering really is min-rmse-first and really does cross
    # experiments, so the assertions above are not vacuous.
    assert board[modeling_task_fixtures["run_b2"].id] == 1
    assert board[modeling_task_fixtures["run_b1"].id] == 2
    assert board[modeling_task_fixtures["run_a1"].id] == 3


async def test_inspector_task_rank_differs_from_experiment_rank(
    modeling_task_fixtures, app_with_db
):
    """run.rank ranks within one experiment; modeling_task.rank ranks across
    the whole task.  Both stay in the payload — they answer different questions."""
    run = modeling_task_fixtures["run_b1"]
    async with AsyncClient(transport=ASGITransport(app=app_with_db), base_url="http://test") as client:
        resp = await client.get(f"/api/platform/runs/{run.id}/inspector")
    data = resp.json()
    assert data["run"]["rank"] == 2            # 2nd inside exp_b
    assert data["modeling_task"]["rank"] == 2  # 2nd across the task

    # ...and the run that is #1 in its experiment is only #3 on the task board.
    other = modeling_task_fixtures["run_a1"]
    async with AsyncClient(transport=ASGITransport(app=app_with_db), base_url="http://test") as client:
        other_resp = await client.get(f"/api/platform/runs/{other.id}/inspector")
    other_data = other_resp.json()
    assert other_data["run"]["rank"] == 1
    assert other_data["modeling_task"]["rank"] == 3


async def test_inspector_rank_is_none_for_unranked_run(
    modeling_task_fixtures, app_with_db
):
    """A run with no objective value is absent from the leaderboard; the drawer
    must get an explicit null (rendered as —) rather than a made-up position."""
    run = modeling_task_fixtures["run_pending"]
    async with AsyncClient(transport=ASGITransport(app=app_with_db), base_url="http://test") as client:
        resp = await client.get(f"/api/platform/runs/{run.id}/inspector")
    assert resp.status_code == 200
    mt = resp.json()["modeling_task"]
    assert mt is not None
    assert mt["target_column"] == "ac_power"  # contract still resolves
    assert mt["rank"] is None


async def test_inspector_modeling_task_is_null_when_experiment_unlinked(
    inspector_fixtures, app_with_db
):
    """The legacy fixture's experiment has no modeling_task_id — the key must
    still be present (the frontend destructures it) and simply be null."""
    run = inspector_fixtures["run_main"]
    async with AsyncClient(transport=ASGITransport(app=app_with_db), base_url="http://test") as client:
        resp = await client.get(f"/api/platform/runs/{run.id}/inspector")
    data = resp.json()
    assert "modeling_task" in data
    assert data["modeling_task"] is None


async def test_inspector_keeps_legacy_keys_alongside_modeling_task(
    inspector_fixtures, app_with_db
):
    """Regression guard for the refactor: `modeling_task` was *added*, nothing
    was removed.  training_task still owns the hyperparameters / model_path /
    arch_config that live nowhere else."""
    run = inspector_fixtures["run_main"]
    async with AsyncClient(transport=ASGITransport(app=app_with_db), base_url="http://test") as client:
        resp = await client.get(f"/api/platform/runs/{run.id}/inspector")
    data = resp.json()

    for key in ("run", "experiment", "platform_task", "training_task",
                "logs", "log_task_id", "siblings", "shap", "diagnosis",
                "modeling_task"):
        assert key in data, key

    tt = data["training_task"]
    assert tt["hyperparameters"] == {"n_estimators": 100}
    assert tt["model_path"] == "/tmp/model.joblib"
    assert tt["target_column"] == "y"
    assert tt["progress"] == 1.0
    assert data["run"]["rank"] == 1

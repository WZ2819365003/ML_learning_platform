"""Contracts for resolving, inspecting, and downloading V3 DL runs."""

from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient

from app.main import create_app
from app.models.database import (
    Dataset,
    DLTrainingLog,
    DLTrainingTask,
    ExperimentRun,
    PlatformExperiment,
    PlatformTask,
    TrainingTask,
    get_db,
)
from app.services.resolver import resolve_task_and_dataset


@pytest.fixture
async def dl_run_fixtures(db, tmp_path):
    from app.config import get_settings

    dataset_file = tmp_path / "dataset.csv"
    dataset_file.write_text("a,b,target\n1,2,0\n3,4,1\n")
    dataset = Dataset(
        name="dl-data",
        file_path=str(dataset_file),
        file_size=dataset_file.stat().st_size,
        row_count=2,
        column_count=3,
    )
    db.add(dataset)
    await db.flush()

    dl_task = DLTrainingTask(
        dataset_id=dataset.id,
        name="mlp-run",
        target_column="target",
        model_type="mlp_dl",
        task_type="classification",
        status="SUCCESS",
        progress=100.0,
        current_epoch=6,
        total_epochs=6,
        result_metrics={"accuracy": 0.8},
    )
    db.add(dl_task)
    await db.flush()
    settings = get_settings()
    settings.ensure_storage_dirs()
    model_file = settings.storage_models / f"{dl_task.id}.pt"
    model_file.write_bytes(b"pytorch-checkpoint")
    dl_task.model_path = str(model_file)
    db.add(DLTrainingLog(task_id=dl_task.id, level="INFO", message="epoch 6 complete"))

    ml_task = TrainingTask(
        dataset_id=dataset.id,
        name="rf-run",
        target_column="target",
        model_type="random_forest",
        status="SUCCESS",
        progress=100.0,
    )
    db.add(ml_task)
    await db.flush()
    ml_model_file = settings.storage_models / f"{ml_task.id}.joblib"
    ml_model_file.write_bytes(b"joblib-model")
    ml_task.model_path = str(ml_model_file)

    platform_task = PlatformTask(
        kind="dl_train",
        status="SUCCESS",
        progress=1.0,
        payload_ref=f"dl_train:{dl_task.id}",
    )
    db.add(platform_task)
    await db.flush()

    experiment = PlatformExperiment(
        name="dl-exp",
        strategy_type="baseline",
        objective_metric="accuracy",
        objective_direction="max",
        dataset_id=dataset.id,
        status="DONE",
    )
    db.add(experiment)
    await db.flush()

    run = ExperimentRun(
        experiment_id=experiment.id,
        task_id=platform_task.id,
        params={"model_type": "mlp_dl", "task_type": "classification"},
        metrics={"accuracy": 0.8},
        status="SUCCESS",
        trial_no=1,
        rank=1,
        source_experiment_type="baseline",
    )
    db.add(run)
    await db.commit()
    yield {
        "dataset": dataset,
        "dl_task": dl_task,
        "ml_task": ml_task,
        "platform_task": platform_task,
        "run": run,
    }
    model_file.unlink(missing_ok=True)
    ml_model_file.unlink(missing_ok=True)


@pytest.fixture
def dl_app_with_db(session_factory):
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


async def test_resolver_supports_direct_and_v3_dl_ids(db, dl_run_fixtures):
    expected_task = dl_run_fixtures["dl_task"]

    direct_task, direct_dataset = await resolve_task_and_dataset(expected_task.id, db)
    platform_task, platform_dataset = await resolve_task_and_dataset(
        dl_run_fixtures["platform_task"].id, db
    )
    run_task, run_dataset = await resolve_task_and_dataset(dl_run_fixtures["run"].id, db)

    assert direct_task.id == expected_task.id
    assert platform_task.id == expected_task.id
    assert run_task.id == expected_task.id
    assert direct_task.task_type == "classification"
    assert (
        direct_dataset.id
        == platform_dataset.id
        == run_dataset.id
        == dl_run_fixtures["dataset"].id
    )


async def test_inspector_serializes_dl_training_context_and_logs(
    dl_run_fixtures, dl_app_with_db
):
    run = dl_run_fixtures["run"]
    async with AsyncClient(
        transport=ASGITransport(app=dl_app_with_db), base_url="http://test"
    ) as client:
        response = await client.get(f"/api/platform/runs/{run.id}/inspector")

    assert response.status_code == 200
    body = response.json()
    assert body["training_task"]["id"] == dl_run_fixtures["dl_task"].id
    assert body["training_task"]["family"] == "dl"
    assert body["training_task"]["task_type"] == "classification"
    assert body["training_task"]["current_epoch"] == 6
    assert body["logs"][0]["message"] == "epoch 6 complete"


async def test_unified_model_download_serves_dl_checkpoint(
    dl_run_fixtures, dl_app_with_db
):
    task = dl_run_fixtures["dl_task"]
    async with AsyncClient(
        transport=ASGITransport(app=dl_app_with_db), base_url="http://test"
    ) as client:
        response = await client.get(f"/api/models/{task.id}/download")

    assert response.status_code == 200
    assert response.content == b"pytorch-checkpoint"
    assert 'filename="mlp_dl_' in response.headers["content-disposition"]
    assert response.headers["content-disposition"].endswith('.pt"')


async def test_model_detail_supports_dl_task(dl_run_fixtures, dl_app_with_db):
    task = dl_run_fixtures["dl_task"]
    async with AsyncClient(
        transport=ASGITransport(app=dl_app_with_db), base_url="http://test"
    ) as client:
        response = await client.get(f"/api/models/{task.id}/detail")

    assert response.status_code == 200
    assert response.json()["task_id"] == task.id
    assert response.json()["family"] == "dl"
    assert response.json()["test_size"] == 0.2


async def test_unified_model_download_preserves_ml_filename(
    dl_run_fixtures, dl_app_with_db
):
    task = dl_run_fixtures["ml_task"]
    async with AsyncClient(
        transport=ASGITransport(app=dl_app_with_db), base_url="http://test"
    ) as client:
        response = await client.get(f"/api/models/{task.id}/download")

    assert response.status_code == 200
    assert response.headers["content-disposition"].endswith('.joblib"')


async def test_model_download_rejects_path_outside_managed_storage(
    db, tmp_path, dl_run_fixtures, dl_app_with_db
):
    task = dl_run_fixtures["dl_task"]
    unmanaged_file = tmp_path / "unmanaged.pt"
    unmanaged_file.write_bytes(b"must-not-download")
    task.model_path = str(unmanaged_file)
    await db.commit()

    async with AsyncClient(
        transport=ASGITransport(app=dl_app_with_db), base_url="http://test"
    ) as client:
        response = await client.get(f"/api/models/{task.id}/download")

    assert response.status_code == 404


async def test_ml_only_visualization_rejects_dl_task_cleanly(
    dl_run_fixtures, dl_app_with_db
):
    task = dl_run_fixtures["dl_task"]
    async with AsyncClient(
        transport=ASGITransport(app=dl_app_with_db), base_url="http://test"
    ) as client:
        response = await client.get(f"/api/viz/{task.id}/confusion_matrix")

    assert response.status_code == 400
    assert "DL" in response.json()["detail"]

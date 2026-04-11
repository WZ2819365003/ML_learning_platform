"""Tests for unified model asset and deployment views."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest
import pytest_asyncio
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app import main as app_main
from app.main import app
from app.models.database import (
    Base,
    DLModelDeployment,
    DLTrainingTask,
    Dataset,
    ModelDeployment,
    TrainingTask,
    get_db,
)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


@pytest.fixture(scope="module")
def test_db(tmp_path_factory):
    db_file = tmp_path_factory.mktemp("db") / "unified_assets.db"
    database_url = f"sqlite+aiosqlite:///{db_file.as_posix()}"
    engine = create_async_engine(database_url, echo=False)
    sessionmaker = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    return engine, sessionmaker


@pytest_asyncio.fixture(scope="module", autouse=True)
async def setup_database(test_db):
    engine, _ = test_db
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await engine.dispose()


@pytest.fixture(scope="module")
def client(test_db):
    test_engine, test_sessionmaker = test_db

    async def override_get_db() -> AsyncSession:
        async with test_sessionmaker() as session:
            yield session

    app_main.async_engine = test_engine
    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


@pytest_asyncio.fixture(scope="module")
async def seeded_assets(test_db, tmp_path_factory):
    _, test_sessionmaker = test_db
    asset_dir = tmp_path_factory.mktemp("assets")
    dataset_path = asset_dir / "dataset.csv"
    dataset_path.write_text("f1,f2,target\n1,2,0\n3,4,1\n", encoding="utf-8")

    ml_model_path = asset_dir / "rf.joblib"
    ml_model_path.write_text("fake-ml-model", encoding="utf-8")

    dl_model_path = asset_dir / "mlp.pt"
    dl_model_path.write_text("fake-dl-model", encoding="utf-8")

    async with test_sessionmaker() as session:
        dataset = Dataset(
            name="seed-dataset.csv",
            file_path=str(dataset_path),
            file_size=dataset_path.stat().st_size,
            row_count=2,
            column_count=3,
            columns_info={"f1": {"dtype": "int"}, "f2": {"dtype": "int"}, "target": {"dtype": "int"}},
        )
        session.add(dataset)
        await session.flush()

        ml_task = TrainingTask(
            dataset_id=dataset.id,
            name="rf-seed",
            model_type="random_forest",
            hyperparameters={"n_estimators": 10},
            target_column="target",
            status="SUCCESS",
            progress=100.0,
            result_metrics={"accuracy": 0.92, "f1": 0.9},
            model_path=str(ml_model_path),
            notes="ml-note",
            tags=["ml", "prod"],
            created_at=_utcnow(),
            finished_at=_utcnow(),
        )
        dl_task = DLTrainingTask(
            dataset_id=dataset.id,
            name="mlp-seed",
            target_column="target",
            model_type="mlp_dl",
            task_type="classification",
            arch_config={"hidden_layers": [16, 8]},
            opt_config={"optimizer": "adam"},
            train_config={"epochs": 5},
            status="SUCCESS",
            progress=100.0,
            current_epoch=5,
            total_epochs=5,
            result_metrics={"val_acc": 0.88, "val_f1_macro": 0.86},
            model_path=str(dl_model_path),
            notes="dl-note",
            tags=["dl"],
            created_at=_utcnow(),
            finished_at=_utcnow(),
        )
        session.add_all([ml_task, dl_task])
        await session.flush()

        session.add(
            ModelDeployment(
                task_id=ml_task.id,
                name="rf-deploy",
                description="ml deployment",
                status="active",
                request_count=7,
            )
        )
        session.add(
            DLModelDeployment(
                dl_task_id=dl_task.id,
                name="mlp-deploy",
                description="dl deployment",
                status="paused",
                request_count=3,
            )
        )
        await session.commit()

        return {
            "dataset_id": dataset.id,
            "ml_task_id": ml_task.id,
            "dl_task_id": dl_task.id,
        }


def test_list_model_assets_returns_unified_view(client: TestClient, seeded_assets):
    response = client.get("/api/models/assets")
    assert response.status_code == 200

    payload = response.json()
    assert payload["total"] == 2
    assert len(payload["items"]) == 2

    runtime_types = {item["runtime_type"] for item in payload["items"]}
    assert runtime_types == {"ml", "dl"}

    ml_asset = next(item for item in payload["items"] if item["runtime_type"] == "ml")
    assert ml_asset["asset_id"] == f"ml:{seeded_assets['ml_task_id']}"
    assert ml_asset["deployment_count"] == 1
    assert ml_asset["notes"] == "ml-note"
    assert ml_asset["tags"] == ["ml", "prod"]
    assert ml_asset["metrics_summary"]["primary_metric_name"] == "accuracy"

    dl_asset = next(item for item in payload["items"] if item["runtime_type"] == "dl")
    assert dl_asset["asset_id"] == f"dl:{seeded_assets['dl_task_id']}"
    assert dl_asset["deployment_count"] == 1
    assert dl_asset["notes"] == "dl-note"
    assert dl_asset["metrics_summary"]["primary_metric_name"] == "val_acc"


def test_list_unified_deployments_returns_standardized_fields(client: TestClient, seeded_assets):
    response = client.get("/api/deploy/assets")
    assert response.status_code == 200

    payload = response.json()
    assert payload["total"] == 2
    assert len(payload["deployments"]) == 2

    ml_deployment = next(item for item in payload["deployments"] if item["runtime_type"] == "ml")
    assert ml_deployment["source_asset_id"] == f"ml:{seeded_assets['ml_task_id']}"
    assert ml_deployment["supports_result_polling"] is True
    assert ml_deployment["predict_url"].endswith(f"/inference/{ml_deployment['deployment_id']}/predict")
    assert ml_deployment["result_url"].endswith("/result/{job_id}")

    dl_deployment = next(item for item in payload["deployments"] if item["runtime_type"] == "dl")
    assert dl_deployment["source_asset_id"] == f"dl:{seeded_assets['dl_task_id']}"
    assert dl_deployment["supports_result_polling"] is False
    assert dl_deployment["predict_url"].endswith(f"/api/dl/deployments/{dl_deployment['deployment_id']}/predict")
    assert dl_deployment["result_url"] is None

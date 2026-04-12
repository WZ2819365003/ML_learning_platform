from __future__ import annotations

import asyncio

import pytest
import pytest_asyncio
from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app import main as app_main
from app.api.routes import timesfm as timesfm_routes
from app.main import app
from app.models.database import (
    Base,
    Dataset,
    TimeSeriesDeployment,
    TimeSeriesForecastTask,
    get_db,
)
from app.services import timeseries_service


@pytest_asyncio.fixture
async def test_db(tmp_path):
    db_file = tmp_path / "timeseries.db"
    engine = create_async_engine(f"sqlite+aiosqlite:///{db_file.as_posix()}", echo=False)
    session_factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    async with session_factory() as session:
        dataset_path = tmp_path / "ts.csv"
        dataset_path.write_text(
            "ds,value\n2024-01-01,1\n2024-01-02,2\n2024-01-03,3\n",
            encoding="utf-8",
        )
        dataset = Dataset(
            name="ts.csv",
            file_path=str(dataset_path),
            file_size=dataset_path.stat().st_size,
            row_count=3,
            column_count=2,
            columns_info={"ds": {"dtype": "string"}, "value": {"dtype": "float"}},
        )
        session.add(dataset)
        await session.commit()
        await session.refresh(dataset)

    yield engine, session_factory, dataset.id

    await engine.dispose()


@pytest.fixture
def client(test_db, monkeypatch: pytest.MonkeyPatch):
    test_engine, test_session_factory, _ = test_db

    async def override_get_db() -> AsyncSession:
        async with test_session_factory() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise

    async def fake_forecast(**_: object):
        await asyncio.sleep(0)
        return {
            "historical": [1.0, 2.0, 3.0],
            "point_forecast": [4.0, 5.0],
            "q10": [3.5, 4.5],
            "q90": [4.5, 5.5],
            "model_name": "amazon/chronos-t5-small",
        }

    monkeypatch.setattr(app_main, "async_engine", test_engine)
    monkeypatch.setattr(timesfm_routes, "is_available", lambda: True)
    monkeypatch.setattr(timeseries_service, "run_forecast_async", fake_forecast)
    monkeypatch.setattr(timeseries_service, "async_session_factory", test_session_factory)

    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


def test_create_and_list_timeseries_deployments(client: TestClient):
    create_response = client.post(
        "/api/ts/deployments",
        json={
            "name": "TimesFM Production A",
            "description": "shared forecasting endpoint",
            "backend_label": "amazon/chronos-t5-small",
        },
    )
    assert create_response.status_code == 201
    deployment = create_response.json()
    assert deployment["name"] == "TimesFM Production A"
    assert deployment["predict_url"].endswith(
        f"/api/ts/deployments/{deployment['deployment_id']}/predict"
    )

    list_response = client.get("/api/ts/deployments")
    assert list_response.status_code == 200
    payload = list_response.json()
    assert payload["total"] == 1
    assert payload["items"][0]["deployment_id"] == deployment["deployment_id"]

    status_response = client.patch(
        f"/api/ts/deployments/{deployment['deployment_id']}/status",
        params={"status": "paused"},
    )
    assert status_response.status_code == 200
    assert status_response.json()["status"] == "paused"


def test_create_timeseries_task_binds_deployment_and_old_api_stays_compatible(
    client: TestClient,
    test_db,
):
    _, _, dataset_id = test_db

    dep_response = client.post(
        "/api/ts/deployments",
        json={"name": "TimesFM Task Runner", "backend_label": "amazon/chronos-t5-small"},
    )
    deployment_id = dep_response.json()["deployment_id"]

    create_response = client.post(
        "/api/ts/tasks",
        json={
            "dataset_id": dataset_id,
            "deployment_id": deployment_id,
            "value_column": "value",
            "time_column": "ds",
            "horizon": 2,
            "frequency": "high",
        },
    )
    assert create_response.status_code == 201
    task = create_response.json()
    assert task["deployment_id"] == deployment_id
    assert task["deployment_name"] == "TimesFM Task Runner"

    for _ in range(20):
        detail_response = client.get(f"/api/ts/tasks/{task['id']}")
        assert detail_response.status_code == 200
        detail = detail_response.json()
        if detail["status"] == "SUCCESS":
            break
    else:
        raise AssertionError("background forecast did not finish")

    assert detail["result"]["point_forecast"] == [4.0, 5.0]

    legacy_response = client.post(
        "/api/timesfm/start",
        json={
            "dataset_id": dataset_id,
            "value_column": "value",
            "time_column": "ds",
            "horizon": 2,
            "frequency": "high",
            "model_name": "amazon/chronos-t5-small",
        },
    )
    assert legacy_response.status_code == 201
    legacy_task = legacy_response.json()
    assert legacy_task["deployment_id"] is not None

    legacy_list_response = client.get("/api/timesfm/list")
    assert legacy_list_response.status_code == 200
    assert legacy_list_response.json()["total"] >= 2


def test_timeseries_task_meta_update_persists_notes_and_tags(client: TestClient, test_db):
    _, _, dataset_id = test_db

    dep_response = client.post(
        "/api/ts/deployments",
        json={"name": "TimesFM Meta Runner", "backend_label": "amazon/chronos-t5-small"},
    )
    deployment_id = dep_response.json()["deployment_id"]

    create_response = client.post(
        "/api/ts/tasks",
        json={
            "dataset_id": dataset_id,
            "deployment_id": deployment_id,
            "value_column": "value",
            "time_column": "ds",
            "horizon": 2,
            "frequency": "high",
        },
    )
    assert create_response.status_code == 201
    task_id = create_response.json()["id"]

    update_response = client.patch(
        f"/api/ts/tasks/{task_id}/meta",
        json={
            "notes": "track review notes for the forecast task",
            "tags": ["forecast", "review"],
        },
    )
    assert update_response.status_code == 200
    payload = update_response.json()
    assert payload["notes"] == "track review notes for the forecast task"
    assert payload["tags"] == ["forecast", "review"]

    detail_response = client.get(f"/api/ts/tasks/{task_id}")
    assert detail_response.status_code == 200
    detail = detail_response.json()
    assert detail["notes"] == "track review notes for the forecast task"
    assert detail["tags"] == ["forecast", "review"]


def test_ts_model_alias_routes_share_status_and_preload(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        timesfm_routes,
        "get_status",
        lambda: {
            "available": True,
            "loaded": False,
            "model": "amazon/chronos-t5-small",
            "backend": "chronos-forecasting (CPU)",
        },
    )

    scheduled_models: list[str] = []

    async def fake_preload(model_name: str) -> None:
        return None

    def fake_create_task(coro):
        model_name = getattr(getattr(coro, "cr_frame", None), "f_locals", {}).get("model_name")
        scheduled_models.append(model_name)
        coro.close()
        return None

    monkeypatch.setattr(timesfm_routes, "preload_model_async", fake_preload)
    monkeypatch.setattr(timesfm_routes.asyncio, "create_task", fake_create_task)

    status_response = client.get("/api/ts/model/status")
    assert status_response.status_code == 200
    assert status_response.json()["model"] == "amazon/chronos-t5-small"

    preload_response = client.post(
        "/api/ts/model/preload",
        params={"model_name": "amazon/chronos-t5-base"},
    )
    assert preload_response.status_code == 200
    assert "Preloading" in preload_response.json()["message"]
    assert scheduled_models == ["amazon/chronos-t5-base"]


@pytest.mark.asyncio
async def test_create_ts_task_commits_before_background_schedule(
    test_db,
    monkeypatch: pytest.MonkeyPatch,
):
    _, test_session_factory, dataset_id = test_db

    async with test_session_factory() as session:
        deployment = TimeSeriesDeployment(
            name="Visibility Runner",
            backend_label="amazon/chronos-t5-small",
            status="active",
            request_count=0,
        )
        session.add(deployment)
        await session.commit()
        await session.refresh(deployment)
        deployment_id = deployment.id

    probe_result: dict[str, bool] = {}

    async def probe(task_id: str) -> None:
        async with test_session_factory() as session:
            row = await session.execute(
                select(TimeSeriesForecastTask).where(TimeSeriesForecastTask.id == task_id)
            )
            probe_result["visible"] = row.scalar_one_or_none() is not None

    def fake_schedule(task_id: str) -> None:
        asyncio.create_task(probe(task_id))

    monkeypatch.setattr(timeseries_service, "_schedule_forecast", fake_schedule)

    async with test_session_factory() as session:
        payload = await timeseries_service.create_ts_task(
            db=session,
            dataset_id=dataset_id,
            deployment_id=deployment_id,
            value_column="value",
            time_column="ds",
            horizon=2,
            frequency="high",
        )
        await asyncio.sleep(0.05)

    assert payload["deployment_id"] == deployment_id
    assert probe_result.get("visible") is True

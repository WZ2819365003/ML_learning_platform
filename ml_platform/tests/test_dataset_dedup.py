from __future__ import annotations

import io

import pandas as pd
import pytest
import pytest_asyncio
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.config import Settings
from app import main as app_main
from app.main import app
from app.models.database import Base, get_db
from app.services import data_service


@pytest_asyncio.fixture
async def test_db(tmp_path):
    db_file = tmp_path / "dataset_dedup.db"
    engine = create_async_engine(f"sqlite+aiosqlite:///{db_file.as_posix()}", echo=False)
    sessionmaker = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    yield engine, sessionmaker

    await engine.dispose()


@pytest.fixture
def client(test_db, tmp_path, monkeypatch):
    _, test_sessionmaker = test_db

    async def override_get_db() -> AsyncSession:
        async with test_sessionmaker() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise

    settings = Settings(
        database_url=f"sqlite+aiosqlite:///{(tmp_path / 'ignored.db').as_posix()}",
        storage_uploads=tmp_path / "uploads",
        storage_models=tmp_path / "models",
        storage_logs=tmp_path / "logs",
    )
    settings.ensure_storage_dirs()
    monkeypatch.setattr(data_service, "get_settings", lambda: settings)
    monkeypatch.setattr(app_main, "async_engine", test_db[0])

    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


def _csv_bytes() -> bytes:
    df = pd.DataFrame(
        {
            "feature1": [1, 2, 3],
            "feature2": [4, 5, 6],
            "target": [0, 1, 0],
        }
    )
    return df.to_csv(index=False).encode("utf-8")


def test_uploading_same_dataset_twice_reuses_existing_record(client: TestClient):
    content = _csv_bytes()

    response1 = client.post(
        "/api/data/upload",
        files={"file": ("dedup.csv", io.BytesIO(content), "text/csv")},
    )
    assert response1.status_code == 201
    dataset1 = response1.json()

    response2 = client.post(
        "/api/data/upload",
        files={"file": ("dedup-copy.csv", io.BytesIO(content), "text/csv")},
    )
    assert response2.status_code == 201
    dataset2 = response2.json()

    assert dataset2["id"] == dataset1["id"]

    list_response = client.get("/api/data/list")
    assert list_response.status_code == 200
    assert list_response.json()["total"] == 1

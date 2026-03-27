"""测试数据管理功能"""

import asyncio
import os
import tempfile
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.main import app
from app.models.database import Base, get_db
from app.models.schemas import DatasetResponse


# 创建测试数据库
TEST_DATABASE_URL = "sqlite+aiosqlite:///:memory:"
test_engine = create_async_engine(TEST_DATABASE_URL, echo=False)
test_sessionmaker = async_sessionmaker(test_engine, class_=AsyncSession, expire_on_commit=False)


async def override_get_db() -> AsyncSession:
    """覆盖数据库依赖"""
    async with test_sessionmaker() as session:
        yield session


@pytest.fixture(scope="module", autouse=True)
async def setup_database():
    """设置测试数据库"""
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await test_engine.dispose()


@pytest.fixture(scope="module")
def client():
    """创建测试客户端"""
    app.dependency_overrides[get_db] = override_get_db
    return TestClient(app)


@pytest.fixture
def test_csv_data():
    """创建测试CSV数据"""
    df = pd.DataFrame({
        "feature1": [1, 2, 3, 4, 5],
        "feature2": [6, 7, 8, 9, 10],
        "target": [0, 1, 0, 1, 0]
    })
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        df.to_csv(f, index=False)
    yield f.name
    os.unlink(f.name)


@pytest.mark.asyncio
async def test_upload_dataset(client, test_csv_data):
    """测试上传数据集"""
    with open(test_csv_data, 'rb') as f:
        response = client.post("/api/data/upload", files={"file": f})
    assert response.status_code == 201
    data = response.json()
    assert "id" in data
    assert "file_name" in data
    assert "created_at" in data


@pytest.mark.asyncio
async def test_list_datasets(client, test_csv_data):
    """测试列出数据集"""
    # 先上传一个数据集
    with open(test_csv_data, 'rb') as f:
        client.post("/api/data/upload", files={"file": f})
    
    # 测试列出数据集
    response = client.get("/api/data/list")
    assert response.status_code == 200
    data = response.json()
    assert "items" in data
    assert "total" in data
    assert "page" in data
    assert "page_size" in data


@pytest.mark.asyncio
async def test_preview_dataset(client, test_csv_data):
    """测试预览数据集"""
    # 先上传一个数据集
    with open(test_csv_data, 'rb') as f:
        upload_response = client.post("/api/data/upload", files={"file": f})
    dataset_id = upload_response.json()["id"]
    
    # 测试预览数据集
    response = client.get(f"/api/data/{dataset_id}/preview")
    assert response.status_code == 200
    data = response.json()
    assert "columns" in data
    assert "rows" in data
    assert len(data["rows"]) <= 100


@pytest.mark.asyncio
async def test_delete_dataset(client, test_csv_data):
    """测试删除数据集"""
    # 先上传一个数据集
    with open(test_csv_data, 'rb') as f:
        upload_response = client.post("/api/data/upload", files={"file": f})
    dataset_id = upload_response.json()["id"]
    
    # 测试删除数据集
    response = client.delete(f"/api/data/{dataset_id}")
    assert response.status_code == 200
    data = response.json()
    assert data["message"] == "Dataset deleted"
    assert data["id"] == dataset_id

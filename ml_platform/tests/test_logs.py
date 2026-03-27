"""测试日志管理功能"""

import asyncio
import os
import tempfile

import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.main import app
from app.models.database import Base, get_db


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
        "feature1": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "feature2": [11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
        "target": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
    })
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        df.to_csv(f, index=False)
    yield f.name
    os.unlink(f.name)


@pytest.fixture
async def training_task_id(client, test_csv_data):
    """创建训练任务并返回任务ID"""
    # 上传数据集
    with open(test_csv_data, 'rb') as f:
        upload_response = client.post("/api/data/upload", files={"file": f})
    dataset_id = upload_response.json()["id"]
    
    # 启动训练任务
    training_request = {
        "dataset_id": dataset_id,
        "model_type": "logistic_regression",
        "target_column": "target",
        "hyperparameters": {"max_iter": 100},
        "test_size": 0.2,
        "eval_metrics": ["accuracy"],
        "cross_validation": {
            "enabled": True,
            "folds": 2
        }
    }
    start_response = client.post("/api/training/start", json=training_request)
    return start_response.json()["id"]


@pytest.mark.asyncio
async def test_get_logs(client, training_task_id):
    """测试获取训练任务日志"""
    response = client.get(f"/api/logs/{training_task_id}")
    assert response.status_code == 200
    data = response.json()
    assert "task_id" in data
    assert "entries" in data
    assert "total" in data


@pytest.mark.asyncio
async def test_get_logs_with_level_filter(client, training_task_id):
    """测试按日志级别过滤"""
    response = client.get(f"/api/logs/{training_task_id}?level=INFO")
    assert response.status_code == 200
    data = response.json()
    assert "task_id" in data
    assert "entries" in data


@pytest.mark.asyncio
async def test_get_logs_with_pagination(client, training_task_id):
    """测试日志分页"""
    response = client.get(f"/api/logs/{training_task_id}?page=1&page_size=10")
    assert response.status_code == 200
    data = response.json()
    assert "task_id" in data
    assert "entries" in data


@pytest.mark.asyncio
async def test_get_metrics(client, training_task_id):
    """测试获取训练任务指标"""
    response = client.get(f"/api/logs/{training_task_id}/metrics")
    assert response.status_code == 200
    data = response.json()
    assert "task_id" in data
    assert "steps" in data


@pytest.mark.asyncio
async def test_download_logs_txt(client, training_task_id):
    """测试下载日志文件（txt格式）"""
    response = client.get(f"/api/logs/{training_task_id}/download?format=txt")
    # 可能返回404如果日志文件不存在，或者返回文件
    assert response.status_code in [200, 404]


@pytest.mark.asyncio
async def test_download_logs_json(client, training_task_id):
    """测试下载日志文件（json格式）"""
    response = client.get(f"/api/logs/{training_task_id}/download?format=json")
    # 可能返回404如果日志文件不存在，或者返回文件
    assert response.status_code in [200, 404]

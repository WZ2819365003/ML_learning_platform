"""测试训练功能"""

import asyncio
import os
import tempfile
import time

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
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


@pytest.fixture(scope="module", autouse=True)
async def setup_database():
    """设置测试数据库"""
    from app.scheduler.scheduler import set_scheduler

    class NoopScheduler:
        async def submit(self, platform_task_id):
            return None

    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    set_scheduler(NoopScheduler())
    yield
    set_scheduler(None)
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
    offset = time.time_ns() % 1_000_000
    df = pd.DataFrame({
        "feature1": [offset + i for i in range(1, 11)],
        "feature2": [11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
        "target": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
    })
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        df.to_csv(f, index=False)
    yield f.name
    os.unlink(f.name)


@pytest.fixture
async def uploaded_dataset_id(client, test_csv_data):
    """上传测试数据集并返回ID"""
    with open(test_csv_data, 'rb') as f:
        response = client.post("/api/data/upload", files={"file": f})
    return response.json()["id"]


@pytest.mark.asyncio
async def test_get_available_models(client):
    """测试获取可用模型列表"""
    response = client.get("/api/training/models")
    assert response.status_code == 200
    payload = response.json()
    models = payload["models"] if isinstance(payload, dict) else payload
    assert isinstance(models, list)
    assert len(models) > 0
    model_ids = {item["id"] if isinstance(item, dict) else item for item in models}
    assert "random_forest" in model_ids
    assert "xgboost" in model_ids
    assert "lightgbm" in model_ids
    assert "logistic_regression" in model_ids
    assert "svm" in model_ids
    assert "mlp" in model_ids


@pytest.mark.asyncio
async def test_start_training(client, uploaded_dataset_id):
    """测试启动训练任务"""
    training_request = {
        "dataset_id": uploaded_dataset_id,
        "model_type": "random_forest",
        "target_column": "target",
        "hyperparameters": {
            "n_estimators": 10,
            "max_depth": 3
        },
        "test_size": 0.2,
        "eval_metrics": ["accuracy", "f1"],
        "cross_validation": {
            "enabled": True,
            "folds": 2
        }
    }
    
    response = client.post("/api/training/start", json=training_request)
    assert response.status_code == 201, response.text
    data = response.json()
    assert "id" in data
    assert data["status"] == "PENDING"
    assert data["model_type"] == "random_forest"
    assert data["target_column"] == "target"


@pytest.mark.asyncio
async def test_list_training_tasks(client, uploaded_dataset_id):
    """测试列出训练任务"""
    # 先启动一个训练任务
    training_request = {
        "dataset_id": uploaded_dataset_id,
        "model_type": "logistic_regression",
        "target_column": "target"
    }
    client.post("/api/training/start", json=training_request)
    
    # 测试列出训练任务
    response = client.get("/api/training/list")
    assert response.status_code == 200
    data = response.json()
    assert "items" in data
    assert "total" in data
    assert "page" in data
    assert "page_size" in data


@pytest.mark.asyncio
async def test_get_training_status(client, uploaded_dataset_id):
    """测试获取训练任务状态"""
    # 先启动一个训练任务
    training_request = {
        "dataset_id": uploaded_dataset_id,
        "model_type": "logistic_regression",
        "target_column": "target"
    }
    start_response = client.post("/api/training/start", json=training_request)
    assert start_response.status_code == 201, start_response.text
    task_id = start_response.json()["id"]
    
    # 测试获取训练任务状态
    response = client.get(f"/api/training/{task_id}/status")
    assert response.status_code == 200
    data = response.json()
    assert data["id"] == task_id

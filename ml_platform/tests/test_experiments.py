"""测试实验管理功能"""

import pytest
from fastapi.testclient import TestClient
from app.main import app


@pytest.fixture(scope="module")
def client():
    """创建测试客户端"""
    return TestClient(app)


@pytest.mark.asyncio
async def test_list_experiments(client):
    """测试列出实验"""
    response = client.get("/api/experiments/list")
    # 可能返回503如果MLflow不可用，或者返回实验列表
    assert response.status_code in [200, 503]
    if response.status_code == 200:
        data = response.json()
        assert isinstance(data, list)


@pytest.mark.asyncio
async def test_list_runs(client):
    """测试列出运行"""
    response = client.get("/api/experiments/runs")
    # 可能返回503如果MLflow不可用，或者返回运行列表
    assert response.status_code in [200, 503]
    if response.status_code == 200:
        data = response.json()
        assert isinstance(data, list)


@pytest.mark.asyncio
async def test_list_runs_with_experiment_name(client):
    """测试按实验名称列出运行"""
    response = client.get("/api/experiments/runs?experiment_name=ml_platform")
    # 可能返回503如果MLflow不可用，或者返回运行列表
    assert response.status_code in [200, 503]
    if response.status_code == 200:
        data = response.json()
        assert isinstance(data, list)


@pytest.mark.asyncio
async def test_list_runs_with_max_results(client):
    """测试限制返回结果数量"""
    response = client.get("/api/experiments/runs?max_results=10")
    # 可能返回503如果MLflow不可用，或者返回运行列表
    assert response.status_code in [200, 503]
    if response.status_code == 200:
        data = response.json()
        assert isinstance(data, list)
        assert len(data) <= 10


@pytest.mark.asyncio
async def test_get_run_detail(client):
    """测试获取运行详情"""
    # 使用一个可能存在的运行ID
    response = client.get("/api/experiments/runs/nonexistent_run_id")
    # 可能返回404如果运行不存在，或者返回503如果MLflow不可用
    assert response.status_code in [404, 503]


@pytest.mark.asyncio
async def test_get_run_detail_with_valid_id(client):
    """测试获取有效运行ID的详情"""
    # 先获取运行列表
    list_response = client.get("/api/experiments/runs?max_results=1")
    if list_response.status_code == 200 and len(list_response.json()) > 0:
        run_id = list_response.json()[0]["run_id"]
        response = client.get(f"/api/experiments/runs/{run_id}")
        assert response.status_code == 200
        data = response.json()
        assert "run_id" in data
        assert "run_name" in data
        assert "status" in data
        assert "params" in data
        assert "metrics" in data

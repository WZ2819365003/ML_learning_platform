"""Tests for the 代码配置 executor (config_exec_service)."""
import pytest

from app.services.config_exec_service import execute_config_code


def test_returns_config_dict():
    cfg = execute_config_code("config = {'selected_models': ['random_forest', 'xgboost'], 'model_family': 'ml'}")
    assert cfg["selected_models"] == ["random_forest", "xgboost"]
    assert cfg["model_family"] == "ml"


def test_supports_comprehensions_and_arithmetic():
    cfg = execute_config_code(
        "ml = ['rf', 'xgb']\n"
        "config = {'selected_models': ml + ['mlp_dl'], 'search_space': {m: {'n': 100} for m in ['rf']}}"
    )
    assert cfg["selected_models"] == ["rf", "xgb", "mlp_dl"]
    assert cfg["search_space"] == {"rf": {"n": 100}}


def test_rejects_empty():
    with pytest.raises(ValueError):
        execute_config_code("   ")


def test_rejects_missing_config():
    with pytest.raises(ValueError):
        execute_config_code("x = 1")


def test_rejects_missing_selected_models():
    with pytest.raises(ValueError):
        execute_config_code("config = {'name': 'x'}")


def test_blocks_import():
    with pytest.raises(ValueError):
        execute_config_code("import os\nconfig = {'selected_models': ['a']}")

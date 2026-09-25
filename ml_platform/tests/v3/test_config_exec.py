"""Tests for the 代码配置 executor (config_exec_service) — A3 subprocess sandbox."""
import time

import pytest

from app.core.user_code_executor import UserCodeTimeout
from app.services.config_exec_service import execute_config_code


async def test_returns_config_dict():
    cfg = await execute_config_code("config = {'selected_models': ['random_forest', 'xgboost'], 'model_family': 'ml'}")
    assert cfg["selected_models"] == ["random_forest", "xgboost"]
    assert cfg["model_family"] == "ml"


async def test_supports_comprehensions_and_arithmetic():
    cfg = await execute_config_code(
        "ml = ['rf', 'xgb']\n"
        "config = {'selected_models': ml + ['mlp_dl'], 'search_space': {m: {'n': 100} for m in ['rf']}}"
    )
    assert cfg["selected_models"] == ["rf", "xgb", "mlp_dl"]
    assert cfg["search_space"] == {"rf": {"n": 100}}


async def test_rejects_empty():
    with pytest.raises(ValueError):
        await execute_config_code("   ")


async def test_rejects_missing_config():
    with pytest.raises(ValueError):
        await execute_config_code("x = 1")


async def test_rejects_missing_selected_models():
    with pytest.raises(ValueError):
        await execute_config_code("config = {'name': 'x'}")


async def test_blocks_import():
    with pytest.raises(ValueError):
        await execute_config_code("import os\nconfig = {'selected_models': ['a']}")


async def test_infinite_loop_killed_by_timeout():
    """A3 acceptance: ``while True`` must not block the API — the sandbox
    subprocess is SIGKILLed at the wall-clock limit and control returns."""
    start = time.monotonic()
    with pytest.raises(UserCodeTimeout):
        await execute_config_code("while True: pass", timeout_s=2)
    elapsed = time.monotonic() - start
    assert elapsed < 8, f"timeout kill took {elapsed:.1f}s — event loop was blocked"


async def test_rejects_non_json_serializable_config():
    with pytest.raises(ValueError):
        await execute_config_code(
            "config = {'selected_models': ['a'], 'bad': lambda: 1}"
        )

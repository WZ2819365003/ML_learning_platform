"""registry/training_plans/*.json 里的方案定义必须能被派发端接受。

方案创建接口（training_plan_service.create_plan）只校验顶层形状，不看
search_space 的内部结构——一份格式写错的方案可以成功入库，要等到用户在调参策略里
套用、点提交时才 422。所以这里直接拿派发端的校验函数过一遍。
"""
from __future__ import annotations

import json
import pathlib

import pytest

from app.core.model_registry import resolve_model_family
from app.services.modeling_task_service import load_tuning_spaces
from app.services.tuning_service import (
    _budget_starvation_detail,
    _planned_trials_per_model,
    _tpe_startup_trials,
    _validate_search_space,
)

PLAN_DIR = pathlib.Path(__file__).resolve().parents[1] / "registry" / "training_plans"
PLAN_FILES = sorted(PLAN_DIR.glob("*.json"))


def _load(path: pathlib.Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_seed_plan_directory_is_not_empty():
    assert PLAN_FILES, f"{PLAN_DIR} 下没有方案定义"


@pytest.mark.parametrize("path", PLAN_FILES, ids=lambda p: p.stem)
def test_search_space_passes_dispatch_validation(path):
    plan = _load(path)
    # 不抛 HTTPException 即通过
    _validate_search_space(plan["strategy_type"], plan["search_space"], plan["selected_models"])


@pytest.mark.parametrize("path", PLAN_FILES, ids=lambda p: p.stem)
def test_every_model_is_a_known_ml_model_for_the_declared_family(path):
    plan = _load(path)
    for token in plan["selected_models"]:
        assert resolve_model_family(token) == plan["model_family"], token
    # search_space 里不应有没选中的模型——派发端会静默忽略，方案就会跟看上去不一样
    assert set(plan["search_space"]) <= set(plan["selected_models"])


@pytest.mark.parametrize("path", PLAN_FILES, ids=lambda p: p.stem)
def test_budget_does_not_starve_any_model(path):
    """max_trials 是整个批次的总上限、按模型顺序消耗。

    这条测试最初按「每个模型的上限」写的，信了 tuning_service 顶部一句错误的文档，
    于是方案里 max_trials 设成了单个模型的次数——线上贝叶斯三个模型只跑了第一个。
    现在直接用派发端的判定函数，方案和派发永远是同一套规则。
    """
    plan = _load(path)
    planned = _planned_trials_per_model(
        plan["strategy_type"], plan["selected_models"], load_tuning_spaces(plan["task_type"]),
        plan["search_space"], plan["budget_config"],
    )
    assert set(planned) == set(plan["selected_models"]), "有模型不会展开任何试验"
    assert _budget_starvation_detail(planned, plan["budget_config"].get("max_trials")) is None


@pytest.mark.parametrize("path", PLAN_FILES, ids=lambda p: p.stem)
def test_bayesian_budget_leaves_room_for_tpe_guided_trials(path):
    """试验次数不超过随机启动次数时，「贝叶斯」实际上全程是随机搜索。"""
    plan = _load(path)
    if plan["strategy_type"] != "bayesian_search":
        pytest.skip("只检查贝叶斯")
    n = plan["budget_config"]["n_trials_per_model"]
    startup = _tpe_startup_trials(n, plan["budget_config"])
    assert n - startup >= n // 2, f"{n} 次试验里有 {startup} 次是随机启动，TPE 引导的太少"

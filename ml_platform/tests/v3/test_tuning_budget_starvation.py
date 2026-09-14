"""max_trials 是整个批次的总上限，按模型先来后到消耗——不能让它悄悄吃掉整个模型。

线上现象：贝叶斯搜索选了 xgboost / lightgbm / 随机森林三个模型，每模型 20 次，
「最大 Trial 数」是界面默认的 20。第一个模型跑满 20 次就把总预算用光，另外两个
模型一次都没训练，实验却显示「已完成」，没有任何提示。网格搜索也一样：
_expand_grid_search 按模型顺序展开组合，到上限就 return。

规则：选了 ≥2 个会实际展开试验的模型、且总上限小于计划总次数时，派发前直接 422，
说清楚每个模型计划多少次、哪些会被截掉、需要把上限调到多少。只选 1 个模型时维持
原来「安全阀」语义，允许截断。
"""
from __future__ import annotations

import pytest
from fastapi import HTTPException
from sqlalchemy import select

from app.models.database import Dataset, ModelingTask
from app.services import tuning_service
from app.services.modeling_task_service import load_tuning_spaces

MODELS = ["xgboost_regressor", "lightgbm_regressor", "random_forest_regressor"]


async def _regression_task(db):
    ds = Dataset(name="load.csv", file_path="/tmp/load.csv", file_size=1, row_count=500)
    db.add(ds)
    await db.flush()
    mt = ModelingTask(
        name="load", dataset_id=ds.id, target_column="load_kw",
        task_type="regression", objective_metric="rmse", objective_direction="min",
    )
    db.add(mt)
    await db.flush()
    await db.commit()
    return (await db.execute(select(ModelingTask).where(ModelingTask.id == mt.id))).scalar_one()


def _preflight(task, **kw):
    return tuning_service._preflight_batch(task, search_space=kw.pop("search_space", {}), **kw)


# ── 纯函数：每个模型计划跑多少次 ────────────────────────────────────────────

def test_planned_trials_follow_registry_grid_and_per_model_budget():
    defaults = load_tuning_spaces("regression")
    grid = tuning_service._planned_trials_per_model("grid_search", MODELS, defaults, {}, {})
    assert grid == {"xgboost_regressor": 12, "lightgbm_regressor": 8, "random_forest_regressor": 9}
    bayes = tuning_service._planned_trials_per_model(
        "bayesian_search", MODELS, defaults, {}, {"n_trials_per_model": 20})
    assert bayes == {m: 20 for m in MODELS}


# ── 贝叶斯 ──────────────────────────────────────────────────────────────────

async def test_bayesian_rejects_a_cap_that_would_skip_whole_models(db):
    task = await _regression_task(db)
    with pytest.raises(HTTPException) as exc:
        _preflight(task, strategy_type="bayesian_search", selected_models=MODELS,
                   budget_config={"n_trials_per_model": 20, "max_trials": 20})
    detail = exc.value.detail
    assert exc.value.status_code == 422
    assert "60" in detail and "20" in detail
    assert "lightgbm_regressor" in detail and "random_forest_regressor" in detail


async def test_bayesian_accepts_a_cap_that_covers_every_model(db):
    task = await _regression_task(db)
    pf = _preflight(task, strategy_type="bayesian_search", selected_models=MODELS,
                    budget_config={"n_trials_per_model": 20, "max_trials": 60})
    assert pf.total_trials == 60


async def test_bayesian_without_cap_runs_every_model(db):
    task = await _regression_task(db)
    pf = _preflight(task, strategy_type="bayesian_search", selected_models=MODELS,
                    budget_config={"n_trials_per_model": 20})
    assert pf.total_trials == 60


# ── 网格 ────────────────────────────────────────────────────────────────────

async def test_grid_rejects_a_cap_that_would_skip_whole_models(db):
    task = await _regression_task(db)
    with pytest.raises(HTTPException) as exc:
        _preflight(task, strategy_type="grid_search", selected_models=MODELS,
                   budget_config={"max_trials": 12})
    detail = exc.value.detail
    assert "29" in detail
    assert "lightgbm_regressor" in detail and "random_forest_regressor" in detail


async def test_grid_accepts_a_cap_that_covers_every_combination(db):
    task = await _regression_task(db)
    pf = _preflight(task, strategy_type="grid_search", selected_models=MODELS,
                    budget_config={"max_trials": 29})
    assert pf.total_trials == 29
    assert {t["model_type"] for t in pf.trials} == set(MODELS)


async def test_single_model_keeps_safety_valve_truncation(db):
    """只有一个模型时截断不会饿死别的模型，保持原语义。"""
    task = await _regression_task(db)
    pf = _preflight(task, strategy_type="grid_search", selected_models=["xgboost_regressor"],
                    budget_config={"max_trials": 5})
    assert pf.total_trials == 5

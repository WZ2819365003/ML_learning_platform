"""分报告名额按 机器学习基线 / 深度学习基线 / 调优 分摊。"""
from __future__ import annotations

from collections import Counter

from app.services.report_run_selection import (
    DL_BASELINE, ML_BASELINE, TUNED,
    bucket_of, describe_selection, select_report_runs,
)


def run(rank, model, strategy="baseline", family="ml", score=None, hp=None):
    return {
        "rank": rank, "run_id": f"r{rank}", "model_type": model, "strategy_type": strategy,
        "family": family, "selection_value": score if score is not None else 20 + rank,
        "params": {"hyperparameters": hp if hp is not None else {"i": rank}},
    }


def virtual_plant_task():
    """线上「虚拟电厂楼宇超短期负荷预测」的真实构成：32 个 Run，前 9 名全是随机森林网格。"""
    pool = [run(i, "random_forest_regressor", "grid_search") for i in range(1, 10)]
    pool += [run(i, "xgboost_regressor", "bayesian_search") for i in range(10, 30)]
    pool += [run(30, "lightgbm_regressor"), run(31, "lstm", family="dl"), run(32, "transformer", family="dl")]
    return pool


def test_old_rule_would_have_written_eight_random_forests():
    """对照组：原来「前 8 名」的结果。"""
    top8 = sorted(virtual_plant_task(), key=lambda r: r["rank"])[:8]
    assert {r["model_type"] for r in top8} == {"random_forest_regressor"}


def test_every_category_and_model_gets_covered_on_the_real_task():
    chosen = select_report_runs(virtual_plant_task())
    assert len(chosen) == 8
    by_bucket = Counter(r["report_bucket"] for r in chosen)
    assert by_bucket == {ML_BASELINE: 1, DL_BASELINE: 2, TUNED: 5}
    models = Counter(r["model_type"] for r in chosen)
    assert models == {
        "random_forest_regressor": 3, "xgboost_regressor": 2,
        "lightgbm_regressor": 1, "lstm": 1, "transformer": 1,
    }


def test_champion_is_always_included_and_output_is_rank_ordered():
    chosen = select_report_runs(virtual_plant_task())
    assert chosen[0]["rank"] == 1
    assert [r["rank"] for r in chosen] == sorted(r["rank"] for r in chosen)


def test_within_a_bucket_models_alternate_before_going_deeper():
    """调优类 5 个名额：随机森林#1、xgboost#1、随机森林#2、xgboost#2、随机森林#3。"""
    tuned = [r for r in select_report_runs(virtual_plant_task()) if r["report_bucket"] == TUNED]
    assert [r["rank"] for r in tuned] == [1, 2, 3, 10, 11]


def test_even_split_when_every_category_is_deep():
    pool = [run(i, "rf", "grid_search") for i in range(1, 11)]
    pool += [run(i, "xgb") for i in range(11, 21)]
    pool += [run(i, "lstm", family="dl") for i in range(21, 31)]
    counts = Counter(r["report_bucket"] for r in select_report_runs(pool))
    # 8 = 3 + 3 + 2，余数给最好成绩更靠前的类
    assert counts == {TUNED: 3, ML_BASELINE: 3, DL_BASELINE: 2}


def test_unused_quota_flows_to_categories_that_still_have_runs():
    pool = [run(1, "lstm", family="dl")] + [run(i, "rf", "grid_search") for i in range(2, 20)]
    counts = Counter(r["report_bucket"] for r in select_report_runs(pool))
    assert counts == {DL_BASELINE: 1, TUNED: 7}


def test_fewer_runs_than_slots_reports_them_all():
    pool = [run(1, "rf"), run(2, "lstm", family="dl")]
    assert len(select_report_runs(pool)) == 2


def test_duplicates_need_same_model_params_and_score():
    """线上真实情况：5 个 xgboost 超参数都为空，分数分成 72.4673 和 95.5468 两组。"""
    pool = [
        run(1, "xgb", score=72.4673, hp={}), run(2, "xgb", score=72.4673, hp={}),
        run(3, "xgb", score=95.5468, hp={}), run(4, "xgb", score=95.5468, hp={}),
        run(5, "xgb", score=95.5468, hp={}),
    ]
    chosen = select_report_runs(pool)
    assert [r["rank"] for r in chosen] == [1, 3]


def test_dl_runs_are_dl_baseline_even_inside_a_tuning_batch():
    assert bucket_of(run(1, "lstm", "grid_search", family="dl")) == DL_BASELINE
    assert bucket_of(run(1, "xgb", "automl")) == TUNED
    assert bucket_of(run(1, "xgb", "baseline")) == ML_BASELINE


def test_family_falls_back_to_params_when_payload_ref_is_missing():
    entry = run(1, "lstm")
    entry["family"] = None
    entry["params"] = {"family": "dl", "hyperparameters": {}}
    assert bucket_of(entry) == DL_BASELINE


def test_describe_selection_counts_unique_runs_per_category():
    pool = virtual_plant_task()
    summary = describe_selection(pool, select_report_runs(pool))
    assert summary[TUNED] == {"label": "调优", "runs": 29, "reported": 5}
    assert summary[DL_BASELINE]["reported"] == 2

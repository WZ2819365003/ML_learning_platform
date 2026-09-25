"""挑哪些 Run 写分报告。

以前是「排行榜前 8 名」。一个任务里网格搜索跑了 9 组随机森林、排在第 1–9 名，
8 篇分报告就全是同一个模型的参数变体，RMSE 只差零点几；贝叶斯调出来的 xgboost、
lightgbm 基线、两个深度学习模型一篇都没有。

现在按三类分摊名额：机器学习基线 / 深度学习基线 / 调优（网格、贝叶斯、AutoML）。

  1. 去重：同模型、同超参数、同得分的 Run 只留排名最靠前的一个。只比参数不够——
     验证方式不记在 params 里，同样的空超参数可以跑出 72.47 和 95.55 两种分数
  2. 分名额：名额在「有 Run 的类别」之间一个一个轮流发，按各类最好成绩排序先发；
     某类的 Run 发完了就跳过它。相当于平均分，余数给成绩好的类，分不完的名额自动
     让给还有 Run 的类
  3. 类内挑 Run：在该类的各个模型之间轮转，每轮给每个模型它的下一个最好 Run。
     保证先覆盖不同模型，再加深同一个模型
  4. 冠军一定在内：它是所在类、所在模型的第一个

纯函数，不碰数据库，便于测试。
"""
from __future__ import annotations

import json
from typing import Any, Iterable

MAX_RUN_REPORTS = 8

ML_BASELINE = "ml_baseline"
DL_BASELINE = "dl_baseline"
TUNED = "tuned"

BUCKET_LABELS = {
    ML_BASELINE: "机器学习基线",
    DL_BASELINE: "深度学习基线",
    TUNED: "调优",
}

_TUNED_STRATEGIES = {"grid_search", "bayesian_search", "automl"}


def _family(entry: dict[str, Any]) -> str:
    family = str(entry.get("family") or "").lower()
    if family in ("ml", "dl"):
        return family
    params = entry.get("params") or {}
    if isinstance(params, dict) and str(params.get("family") or "").lower() in ("ml", "dl"):
        return str(params["family"]).lower()
    # 历史 Run 可能既没有 payload_ref 也没有 params.family，退回注册表判断
    try:
        from app.core.model_registry import resolve_model_family
        return resolve_model_family(str(entry.get("model_type") or "")) or "ml"
    except Exception:  # noqa: BLE001 — 选报告不能因为注册表导入失败而整体报错
        return "ml"


def bucket_of(entry: dict[str, Any]) -> str:
    # 深度学习在网格/贝叶斯里会被降级成基线（tuning_service._expand_dl_baseline），
    # 所以 DL 一律算深度学习基线，不看批次的策略
    if _family(entry) == "dl":
        return DL_BASELINE
    strategy = str(entry.get("strategy_type") or "baseline").lower()
    return TUNED if strategy in _TUNED_STRATEGIES else ML_BASELINE


def _score(entry: dict[str, Any]) -> Any:
    for key in ("selection_value", "objective_value"):
        value = entry.get(key)
        if isinstance(value, (int, float)):
            return round(float(value), 6)
    return None


def _hyperparameters(entry: dict[str, Any]) -> Any:
    params = entry.get("params") or {}
    if not isinstance(params, dict):
        return params
    return params.get("hyperparameters", params)


def duplicate_key(entry: dict[str, Any]) -> str:
    return json.dumps(
        [entry.get("model_type"), _hyperparameters(entry), _score(entry)],
        sort_keys=True, ensure_ascii=False, default=str,
    )


def _ranked(entries: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(entries, key=lambda e: (e.get("rank") is None, e.get("rank") or 0))


def select_report_runs(
    pool: Iterable[dict[str, Any]],
    max_reports: int = MAX_RUN_REPORTS,
) -> list[dict[str, Any]]:
    """从已排序的成功 Run 里挑出要写分报告的，按全局排名返回，并标上 report_bucket。"""
    ranked = _ranked(e for e in pool if str(e.get("status", "SUCCESS")).upper() == "SUCCESS")

    seen: set[str] = set()
    unique: list[dict[str, Any]] = []
    for entry in ranked:
        key = duplicate_key(entry)
        if key in seen:
            continue
        seen.add(key)
        unique.append(entry)

    # 类 → 模型 → Run（都保持排名顺序；dict 保留插入顺序，即各自最好成绩的顺序）
    buckets: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for entry in unique:
        by_model = buckets.setdefault(bucket_of(entry), {})
        by_model.setdefault(str(entry.get("model_type")), []).append(entry)

    capacity = {b: sum(len(v) for v in models.values()) for b, models in buckets.items()}
    quota = {b: 0 for b in buckets}
    slots = min(max_reports, sum(capacity.values()))
    while sum(quota.values()) < slots:
        for b in buckets:
            if sum(quota.values()) >= slots:
                break
            if quota[b] < capacity[b]:
                quota[b] += 1

    chosen: list[dict[str, Any]] = []
    for b, models in buckets.items():
        queues = [list(runs) for runs in models.values()]
        picked = 0
        while picked < quota[b]:
            for q in queues:
                if picked >= quota[b]:
                    break
                if q:
                    chosen.append({**q.pop(0), "report_bucket": b})
                    picked += 1

    return _ranked(chosen)


def describe_selection(pool: Iterable[dict[str, Any]], chosen: list[dict[str, Any]]) -> dict[str, Any]:
    """给报告元数据用：每类共有多少个（去重后）、写了几篇。"""
    unique_counts: dict[str, int] = {}
    seen: set[str] = set()
    for entry in _ranked(pool):
        key = duplicate_key(entry)
        if key in seen:
            continue
        seen.add(key)
        b = bucket_of(entry)
        unique_counts[b] = unique_counts.get(b, 0) + 1
    written: dict[str, int] = {}
    for entry in chosen:
        written[entry["report_bucket"]] = written.get(entry["report_bucket"], 0) + 1
    return {
        b: {"label": BUCKET_LABELS[b], "runs": unique_counts.get(b, 0), "reported": written.get(b, 0)}
        for b in (ML_BASELINE, DL_BASELINE, TUNED)
        if unique_counts.get(b)
    }

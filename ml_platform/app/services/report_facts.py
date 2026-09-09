"""Every number and verdict a report asserts, computed from the context.

Nothing here is left to the model. That includes the judgements — whether a gap
between two models is meaningful, whether cross-fold spread is stable, whether
an error distribution has a tail — because each is a comparison the code can do
exactly and the model can only estimate.

Numbers in prose are rounded to what a reader needs (72.5, 0.8%); the exact
values live in the chart rows, where hovering reveals them.
"""

from __future__ import annotations

import re
from typing import Any

from app.core.validation_split import has_temporal_features

# Names a feature-engineering step produces, as opposed to a column that came
# with the data. Grouped so the dataset section can describe composition rather
# than list 35 column names.
_FEATURE_GROUPS: tuple[tuple[str, str, str], ...] = (
    ("周期三角变换", r"(_sin|_cos)$", "把环形的时间关系显式交给模型"),
    ("滞后项", r"_lag_\d+$", "引入历史时刻的取值"),
    ("滚动统计", r"_roll_", "描述近期水平与波动"),
    ("气象与交互衍生", r"(_x_|_degree$|_index$|_depression$|_spread_|_sq$|^days_since_)", "由原始观测量派生"),
)

BASE_GROUP_NAME = "原始采集与日历"

_ERROR_METRICS = ("rmse", "mae", "mse", "mape")
_SCORE_METRICS = ("r2", "accuracy", "f1", "roc_auc", "auc", "precision", "recall")

# RMSE/MAE for a normal error distribution. The ratio is always ≥ 1, so the
# number alone says nothing; only the distance from this baseline does.
_NORMAL_RMSE_MAE = 1.2533

# A gap this many fold-standard-deviations wide is not noise, whatever the
# validation scheme on either side of it.
_DECISIVE_GAP_IN_STD = 3.0

_METRIC_NAMES = {"r2": "R²", "rmse": "RMSE", "mae": "MAE", "mse": "MSE",
                 "mape": "MAPE", "accuracy": "准确率", "f1": "F1",
                 "roc_auc": "ROC-AUC", "auc": "AUC"}

_TIME_SAFE_STRATEGIES = {"time_series_expanding", "chronological_holdout"}

_CN_DIGITS = "零一两三四五六七八九十"


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def _fmt(value: Any, digits: int = 4) -> str:
    """Round to `digits`, then drop only trailing decimal zeros.

    ":g" switches to six significant figures, which silently turned 14274.15
    into 14274.1 — a report is not the place to lose a digit of a stated range.
    Zeros are stripped only after a decimal point: "130" must stay "130".
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return "—"
    text = f"{float(value):.{digits}f}"
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text or "0"


def readable(value: Any, metric: str | None = None) -> str:
    """The precision a sentence needs: 72.5, 0.3, 132, 0.997.

    Scores on a 0–1 scale keep three decimals because 0.778 and 0.758 are
    different models; everything else keeps one significant decimal past what
    the magnitude already says. Exact values belong in chart rows.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return "—"
    v = abs(float(value))
    if metric and str(metric).lower() in _SCORE_METRICS:
        return _fmt(value, 3)
    if v >= 100:
        return _fmt(value, 0)
    if v >= 0.1:
        return _fmt(value, 1)
    return _fmt(value, 2)


def pct_text(value: Any, base: Any, *, exact: bool = False) -> str:
    """`value` as a share of `base`; one decimal for prose, two for rows."""
    if not isinstance(value, (int, float)) or not isinstance(base, (int, float)) or not base:
        return "—"
    pct = abs(value) / abs(base) * 100
    if exact:
        return f"{_fmt(pct, 2)}%"
    if pct >= 10:
        return f"{_fmt(pct, 0)}%"
    if pct >= 0.1:
        return f"{_fmt(pct, 1)}%"
    return f"{_fmt(pct, 2)}%"


def _pct(value: float, base: float, digits: int = 2) -> str:
    return f"{round(abs(value) / abs(base) * 100, digits):g}%"


def cn_count(n: int) -> str:
    """Small counts in Chinese numerals, so prose is not a wall of digits."""
    if isinstance(n, bool) or not isinstance(n, int) or n < 0:
        return str(n)
    if n <= 10:
        return _CN_DIGITS[n]
    return str(n)


def metric_name(key: str | None) -> str:
    return _METRIC_NAMES.get(str(key or "").lower(), str(key or "score").upper())


def is_error_metric(key: str | None) -> bool:
    return str(key or "").lower() in _ERROR_METRICS


def short_model_name(model: Any) -> str:
    """xgboost_regressor → xgboost, for the one-line cover verdict only."""
    text = str(model or "")
    return re.sub(r"_(regressor|classifier|dl)$", "", text) or text


# ---------------------------------------------------------------------------
# Columns
# ---------------------------------------------------------------------------

def classify_columns(columns: list[str], target: str | None) -> dict[str, Any]:
    """Split the dataset's columns into what was collected and what was built."""
    engineered: dict[str, list[str]] = {}
    base: list[str] = []
    for col in columns:
        for name, pattern, _ in _FEATURE_GROUPS:
            if re.search(pattern, col):
                engineered.setdefault(name, []).append(col)
                break
        else:
            base.append(col)
    groups = [
        {"name": name, "columns": engineered[name], "purpose": purpose}
        for name, _, purpose in _FEATURE_GROUPS
        if engineered.get(name)
    ]
    eng_count = sum(len(g["columns"]) for g in groups)
    return {
        "base": base,
        "base_count": len(base),
        "groups": groups,
        "eng_count": eng_count,
        "eng_pct": _pct(eng_count, len(columns), 0) if columns else "—",
        "target": target,
    }


def dataset_columns(context: dict[str, Any]) -> list[str]:
    """All column names. columns_info is capped at sixteen keys; column_names is not."""
    dataset = context.get("dataset") or {}
    columns = dataset.get("column_names") or list((dataset.get("columns_info") or {}).keys())
    return [str(c) for c in columns]


# ---------------------------------------------------------------------------
# Verdict primitives
# ---------------------------------------------------------------------------

def gap_verdict(gap: float | None, noise: float | None,
                noise_name: str = "折间波动") -> dict[str, Any]:
    """Is the distance between two models bigger than one model's own wobble?

    A leaderboard prints differences to four decimals and says nothing about
    which of them mean anything. Comparing the gap to the champion's cross-fold
    standard deviation is the cheapest honest answer, and it is arithmetic, not
    judgement — so the model never has to guess it.
    """
    if not isinstance(gap, (int, float)) or not isinstance(noise, (int, float)) or noise <= 0:
        return {"known": False}
    within = abs(gap) < abs(noise)
    return {
        "known": True,
        "within_noise": within,
        "short": "分不出高下" if within else "差距是真实的",
        "long": (
            f"差距落在{noise_name}之内，两者分不出高下"
            if within else f"差距超出{noise_name}，落差是真实的"
        ),
    }


def spread_verdict(values: list[float]) -> dict[str, Any]:
    """Is one fold an outlier, or is the whole thing wobbling?

    Different problems with different fixes: a single bad fold means the split
    is uneven, general spread means the model is unstable. The worst fold is
    measured against the spread of the *other* folds: a z-score over all folds
    cannot exceed sqrt(n-1), which for five folds is exactly 2, so a "more than
    two standard deviations" rule could never fire on a five-fold run — one fold
    at 90 among four at 70 read as "no outlier".
    """
    if len(values) < 3:
        return {"known": False}
    mean = sum(values) / len(values)
    var = sum((v - mean) ** 2 for v in values) / len(values)
    std = var ** 0.5
    worst = max(values, key=lambda v: abs(v - mean))
    others = list(values)
    others.remove(worst)
    others_mean = sum(others) / len(others)
    others_std = (sum((v - others_mean) ** 2 for v in others) / len(others)) ** 0.5
    deviation = abs(worst - others_mean)
    tolerance = 1e-9 * max(1.0, abs(others_mean))
    outlier = (deviation > 3 * others_std) if others_std > tolerance else (deviation > tolerance)
    return {
        "known": True,
        "mean": mean,
        "std": std,
        "range": max(values) - min(values),
        "outlier": outlier,
        "note": (
            "存在单折离群，说明数据划分不均"
            if outlier else "无量级差异，亦未出现单折离群"
        ),
    }


def error_shape(rmse: float | None, mae: float | None) -> dict[str, Any]:
    """What the RMSE/MAE ratio says about the tail of the error distribution."""
    if not isinstance(rmse, (int, float)) or not isinstance(mae, (int, float)) or mae <= 0:
        return {}
    ratio = rmse / mae
    if ratio < _NORMAL_RMSE_MAE * 1.05:
        note = "接近正态误差分布下的 1.25，误差分布无明显长尾"
    elif ratio < _NORMAL_RMSE_MAE * 1.4:
        note = "略高于正态误差分布下的 1.25，存在一定量的大偏差样本，但不构成长尾"
    else:
        note = "显著高于正态误差分布下的 1.25，说明存在少量极端偏差样本"
    return {
        "ratio": round(ratio, 2),
        "sentence": f"RMSE/MAE 比值为 {round(ratio, 2):g}，{note}。",
    }


# ---------------------------------------------------------------------------
# Leaderboard reading
# ---------------------------------------------------------------------------

def entry_metric(entry: dict[str, Any], key: str) -> Any:
    metrics = entry.get("metrics") or {}
    for candidate in (f"cv_avg_{key}", f"selection_cv_mean_{key}", key):
        value = metrics.get(candidate)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return value
    return entry.get("objective_value")


def entry_std(entry: dict[str, Any], key: str) -> float | None:
    metrics = entry.get("metrics") or {}
    for candidate in (f"cv_std_{key}", f"selection_cv_std_{key}"):
        value = metrics.get(candidate)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
    return None


def entry_r2(entry: dict[str, Any]) -> float | None:
    metrics = entry.get("metrics") or {}
    for candidate in ("cv_avg_r2", "selection_cv_mean_r2", "val_r2", "selection_val_r2", "r2"):
        value = metrics.get(candidate)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
    return None


def is_cv(entry: dict[str, Any]) -> bool:
    metrics = entry.get("metrics") or {}
    return any(
        isinstance(value, (int, float))
        for key, value in metrics.items()
        if key.startswith(("cv_avg_", "selection_cv_mean_"))
    )


# Older names, still imported by other modules.
_entry_metric = entry_metric
_is_cv = is_cv


def validation_scheme(entry: dict[str, Any]) -> str:
    """Return the measurement cohort an entry can honestly be compared in."""
    strategy = str((entry.get("metrics") or {}).get("validation_strategy") or "")
    if strategy == "time_series_expanding":
        return "时间序列交叉验证"
    if strategy == "chronological_holdout":
        return "时间顺序留出验证"
    if is_cv(entry):
        return "交叉验证"
    metrics = entry.get("metrics") or {}
    if any(key.startswith("selection_val_") for key in metrics):
        return "留出验证"
    if isinstance(metrics.get("history"), list) and metrics.get("history"):
        return "留出验证"
    return "验证结果"


def _cohorts(board: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for entry in board:
        grouped.setdefault(validation_scheme(entry), []).append(entry)
    return grouped


def _metric_label(key: str | None, scheme: str) -> str:
    return f"{scheme} {metric_name(key)}"


def display_labels(board: list[dict[str, Any]]) -> dict[str, str]:
    """One label per entry; a rerun of the same model gets a numbered suffix.

    Keyed by run_id when there is one, else by position, because a chart
    category must be unique and two xgboost bars with the same name are one
    bar to a tooltip.
    """
    labels: dict[str, str] = {}
    seen: dict[str, int] = {}
    for index, entry in enumerate(board):
        model = str(entry.get("model_type") or entry.get("run_id") or f"run-{index + 1}")
        seen[model] = seen.get(model, 0) + 1
        labels[_entry_key(entry, index)] = model if seen[model] == 1 else f"{model} #{seen[model]}"
    return labels


def _entry_key(entry: dict[str, Any], index: int) -> str:
    return str(entry.get("run_id") or f"#{index}")


def rank_summary(context: dict[str, Any]) -> dict[str, Any] | None:
    """Who won, by how much, and whether the margin means anything.

    The leaderboard is sorted on the objective across every validation scheme,
    so its first row is the best number recorded. Whether that number beats a
    run measured differently is settled by size: a gap wider than three
    fold-standard-deviations is not noise under any scheme, and a report that
    refuses to rank a 72 against a 132 because one is a fold mean and the other
    a hold-out score is being coy, not careful. Inside one scheme the bar is
    lower — one standard deviation — because those numbers are directly
    comparable.
    """
    task = context.get("task") or {}
    board = context.get("leaderboard") or []
    if not board:
        return None
    metric = str(task.get("objective_metric") or "score").lower()
    stats = context.get("_target_stats") or {}
    mean = stats.get("mean") if isinstance(stats.get("mean"), (int, float)) else None
    error = is_error_metric(metric)

    cohorts = _cohorts(board)
    best = board[0]
    best_scheme = validation_scheme(best)
    best_value = entry_metric(best, metric)
    best_std = entry_std(best, metric)
    cohort = cohorts[best_scheme]

    # Two runs of the same model with identical scores read as a two-horse race
    # until someone notices they are the same horse.
    dup = [e for e in cohort[1:] if e.get("model_type") == best.get("model_type")
           and entry_metric(e, metric) == best_value]
    distinct = [best] + [e for e in cohort[1:] if e not in dup]
    runner = distinct[1] if len(distinct) > 1 else None
    runner_value = entry_metric(runner, metric) if runner else None
    gap = (abs(runner_value - best_value)
           if isinstance(runner_value, (int, float)) and isinstance(best_value, (int, float))
           else None)
    noise = best_std if best_std else (entry_std(runner, metric) if runner else None)
    verdict = gap_verdict(gap, noise)

    def _gap(entry: dict[str, Any]) -> float | None:
        value = entry_metric(entry, metric)
        if isinstance(value, (int, float)) and isinstance(best_value, (int, float)):
            return abs(value - best_value)
        return None

    beyond = [e for e in distinct[1:] if noise and (_gap(e) or 0) >= noise]
    within = [e for e in distinct[1:] if noise and (_gap(e) or 0) < noise]

    others: list[dict[str, Any]] = []
    for scheme, entries in cohorts.items():
        if scheme == best_scheme:
            continue
        leader = entries[0]
        leader_value = entry_metric(leader, metric)
        values = [entry_metric(e, metric) for e in entries]
        values = [v for v in values if isinstance(v, (int, float))]
        cross_noise = best_std or entry_std(leader, metric)
        cross_gap = (abs(leader_value - best_value)
                     if isinstance(leader_value, (int, float)) and isinstance(best_value, (int, float))
                     else None)
        decided = (cross_gap is not None and cross_noise is not None and cross_noise > 0
                   and cross_gap > _DECISIVE_GAP_IN_STD * cross_noise) or None
        if decided is None and cross_gap is not None and cross_noise is not None and cross_noise > 0:
            decided = False
        ratios = ([v / best_value for v in values]
                  if error and isinstance(best_value, (int, float)) and best_value else [])
        others.append({
            "scheme": scheme,
            "entries": entries,
            "leader": leader,
            "leader_value": leader_value,
            "gap": cross_gap,
            "noise": cross_noise,
            "decided": decided,
            "ratio_lo": min(ratios) if ratios else None,
            "ratio_hi": max(ratios) if ratios else None,
            "gap_lo": (min(abs(v - best_value) for v in values)
                       if values and isinstance(best_value, (int, float)) else None),
        })
    others.sort(key=lambda o: -len(o["entries"]))

    return {
        "metric": metric,
        "metric_name": metric_name(metric),
        "error_metric": error,
        "mean": mean,
        "target": task.get("target_column"),
        "board": board,
        "cohorts": cohorts,
        "labels": display_labels(board),
        "best": best,
        "best_value": best_value,
        "best_std": best_std,
        "best_scheme": best_scheme,
        "best_pct": pct_text(best_value, mean) if error and mean else None,
        "dup": dup,
        "distinct": distinct,
        "runner": runner,
        "runner_value": runner_value,
        "gap": gap,
        "noise": noise,
        "verdict": verdict,
        "beyond": beyond,
        "within": within,
        "others": others,
        "mixed": len(cohorts) > 1,
    }


# ---------------------------------------------------------------------------
# Overview facts
# ---------------------------------------------------------------------------

def _has_final_evaluation(task: dict[str, Any], board: list[dict[str, Any]]) -> bool:
    final_state = task.get("final_evaluation") or {}
    return bool(
        task.get("final_test_value") is not None
        or final_state.get("state") == "FINALIZED"
        or final_state.get("final_metrics")
        or any(entry.get("final_test_value") is not None for entry in board)
    )


def _leak_state(columns: list[str], board: list[dict[str, Any]]) -> str | None:
    """None, "all" or "some": how many runs were validated on shuffled rows."""
    if not has_temporal_features(columns):
        return None
    recorded = [str((e.get("metrics") or {}).get("validation_strategy") or "") for e in board]
    unsafe = [s for s in recorded if s not in _TIME_SAFE_STRATEGIES]
    if not unsafe:
        return None
    return "all" if len(unsafe) == len(recorded) else "some"


def top_risk(context: dict[str, Any], summary: dict[str, Any] | None) -> dict[str, Any]:
    """The single most serious caveat, by a fixed order: leak > no final > rerun."""
    task = context.get("task") or {}
    board = context.get("leaderboard") or []
    leak = _leak_state(dataset_columns(context), board)
    if leak:
        return {"key": "leak", "leak": leak}
    if summary and isinstance(summary.get("best_value"), (int, float)) \
            and not _has_final_evaluation(task, board):
        return {"key": "final"}
    if summary and summary.get("dup"):
        return {"key": "dup"}
    return {"key": None}


def _risk_sentence(risk: dict[str, Any], summary: dict[str, Any] | None) -> str:
    key = risk.get("key")
    if key == "leak":
        opener = ("这批 Run 用的是随机切分。" if risk.get("leak") == "all"
                  else "这批 Run 里有一部分用的是随机切分。")
        return opener + "数据里有滞后或周期特征，随机切分会让“未来”混进训练，所以上面的分数比真实上线时乐观。"
    if key == "final":
        return "这些分数来自模型选择阶段，封存测试集上的最终评估还没做。选择阶段的数据参与过挑选，分数天然偏乐观。"
    if key == "dup" and summary:
        n = cn_count(len(summary["dup"]) + 1)
        return (f"排名前{n}位是同一个 {summary['best'].get('model_type')} 的重复训练，结果一致。"
                f"它们不是{n}个候选，比较时只算一个。")
    return "没有发现影响结论的验证风险。"


def _headline(summary: dict[str, Any], risk: dict[str, Any]) -> str:
    best = short_model_name(summary["best"].get("model_type"))
    if summary["error_metric"] and summary["mean"] and isinstance(summary["best_value"], (int, float)):
        pct = abs(summary["best_value"]) / abs(summary["mean"]) * 100
        target = summary.get("target") or "目标"
        error = (f"误差不到 {target} 均值的 1%" if pct < 1
                 else f"误差约为 {target} 均值的 {_fmt(pct, 1)}%")
    else:
        error = f"{summary['metric_name']} {readable(summary['best_value'], summary['metric'])}"
    key = risk.get("key")
    if key == "leak":
        tail = "但这批模型是随机切分训练的，分数偏乐观"
    elif key == "final":
        tail = "但还没在封存测试集上做最终评估"
    elif key == "dup":
        tail = f"但前{cn_count(len(summary['dup']) + 1)}名是同一模型的重复训练"
    else:
        tail = "各项验证检查已通过"
    return f"{best} 胜出，{error}；{tail}。"


def _verdict_sentences(summary: dict[str, Any]) -> dict[str, str]:
    metric = summary["metric"]
    name = summary["metric_name"]
    best = summary["best"].get("model_type")
    value = readable(summary["best_value"], metric)
    scheme = summary["best_scheme"]
    if summary["best_pct"]:
        target = summary.get("target") or "目标列"
        verdict = (f"{best} 表现最好，{scheme} {name} {value}，"
                   f"相当于把 {target} 预测偏了均值的 {summary['best_pct']}。")
    else:
        verdict = f"{best} 表现最好，{scheme} {name} {value}。"

    runner = ""
    if summary["runner"] is not None and isinstance(summary["gap"], (int, float)):
        who = summary["runner"].get("model_type")
        gap = readable(summary["gap"], metric)
        verdict_gap = summary["verdict"]
        if verdict_gap.get("known") and verdict_gap["within_noise"]:
            runner = f"{who} 只差 {gap}——这个差距比模型自身的折间波动还小，两者分不出高下。"
        elif verdict_gap.get("known"):
            runner = (f"{who} 落后 {gap}，大于折间波动 {readable(summary['noise'], metric)}，"
                      "这个落差是真实的。")
        else:
            runner = f"{who} 落后 {gap}。"

    others = ""
    if summary["others"]:
        other = summary["others"][0]
        n = cn_count(len(other["entries"]))
        if other["decided"]:
            if summary["error_metric"] and other["ratio_lo"]:
                others = (f"另外{n}个{other['scheme']}模型的误差都在它的 "
                          f"{_fmt(other['ratio_lo'], 1)} 倍以上，明显落后。")
            else:
                others = (f"另外{n}个{other['scheme']}模型都比它差 "
                          f"{readable(other['gap_lo'], metric)} 以上，明显落后。")
        else:
            others = (f"{other['leader'].get('model_type')}（{other['scheme']} "
                      f"{readable(other['leader_value'], metric)}）与它接近，口径不同，排不出先后。")
    elif len(summary["distinct"]) > 2:
        rest = summary["distinct"][2:]
        gaps = [abs(entry_metric(e, metric) - summary["best_value"]) for e in rest
                if isinstance(entry_metric(e, metric), (int, float))]
        if gaps:
            others = f"其余{cn_count(len(rest))}个模型落后 {readable(min(gaps), metric)} 以上。"
    return {"verdict": verdict, "runner": runner, "others": others}


def _gaps_sentence(summary: dict[str, Any]) -> str:
    metric = summary["metric"]
    scheme = summary["best_scheme"]
    distinct = summary["distinct"]
    if len(distinct) < 2:
        return f"{scheme}组只有 {summary['best'].get('model_type')} 一个模型，没有可比较的差距。"
    if not summary["noise"]:
        return f"{scheme}组没有记录折间波动，模型之间的差距无法与噪声比较。"
    beyond, within = summary["beyond"], summary["within"]
    if beyond:
        first = beyond[0]
        gap = readable(abs(entry_metric(first, metric) - summary["best_value"]), metric)
        if len(beyond) == 1:
            text = f"{first.get('model_type')} 落后 {gap}，是{scheme}组里唯一能确认的差距。"
        else:
            names = "、".join(str(e.get("model_type")) for e in beyond)
            text = f"{names} 落后 {gap} 以上，差距能够确认。"
        if within:
            text += f"其余{cn_count(len(within) + 1)}个用哪个都行。"
        return text
    return f"{scheme}组内{cn_count(len(distinct))}个模型的差距都小于折间波动，用哪个都行。"


def _shap_lead(best: dict[str, Any]) -> str:
    shap = ((best.get("metrics") or {}).get("top_shap_importances") or [])[:2]
    if len(shap) < 2 or not isinstance(shap[0].get("mean_abs_shap"), (int, float)):
        return ""
    top, second = shap[0], shap[1]
    if not second.get("mean_abs_shap"):
        return f"模型主要在用 {top.get('feature')} 做预测。"
    ratio = abs(top["mean_abs_shap"]) / abs(second["mean_abs_shap"])
    if ratio >= 3:
        return f"模型主要在用 {top.get('feature')} 做预测，其余特征只是修正。"
    if ratio >= 1.5:
        return f"{top.get('feature')} 贡献最大，{second.get('feature')} 次之，两者差 {_fmt(ratio, 1)} 倍。"
    return f"{top.get('feature')} 与 {second.get('feature')} 的贡献接近，没有单一主导特征。"


def _target_missing(context: dict[str, Any]) -> int | None:
    task = context.get("task") or {}
    info = ((context.get("dataset") or {}).get("columns_info") or {}).get(task.get("target_column"))
    if isinstance(info, dict):
        for key in ("missing_count", "null_count"):
            if isinstance(info.get(key), (int, float)):
                return int(info[key])
    return None


def build_overview_facts(context: dict[str, Any]) -> dict[str, Any]:
    task = context.get("task") or {}
    dataset = context.get("dataset") or {}
    board = context.get("leaderboard") or []
    summary = rank_summary(context)
    if summary is None:
        return {"task": {"name": task.get("name") or "建模任务"}}

    risk = top_risk(context, summary)
    columns = dataset_columns(context)
    fields = classify_columns(columns, task.get("target_column"))
    missing = _target_missing(context)
    rows = dataset.get("row_count") or None
    cols = dataset.get("column_count") or (len(columns) or None)
    shape = f"{rows if rows is not None else '—'} 行、{cols if cols is not None else '—'} 列。"
    if task.get("target_column"):
        if missing == 0:
            shape += f"目标列 {task['target_column']} 无缺失。"
        elif isinstance(missing, int):
            shape += f"目标列 {task['target_column']} 缺失 {missing} 个。"
        else:
            shape += f"目标列是 {task['target_column']}。"

    counts = context.get("run_status_counts") or {}
    facts: dict[str, Any] = {
        "task": {"name": task.get("name") or "建模任务"},
        "headline": {"sentence": _headline(summary, risk)},
        "conclusion": _verdict_sentences(summary),
        "risk": {"key": risk.get("key"), "sentence": _risk_sentence(risk, summary)},
        "gaps": {"sentence": _gaps_sentence(summary)},
        "best": {
            "model": summary["best"].get("model_type"),
            "metric_label": _metric_label(summary["metric"], summary["best_scheme"]),
            "value": readable(summary["best_value"], summary["metric"]),
            "pct_of_mean": summary["best_pct"] or "—",
            "fold_std": readable(summary["best_std"], summary["metric"]),
        },
        "runs": {
            "total": sum(counts.values()) or len(board),
            "models": len({e.get("model_type") for e in board}),
        },
        "ds": {"shape_sentence": shape},
        "fields": {
            "has_groups": bool(fields["groups"]),
            "eng_count": fields["eng_count"],
            "eng_pct": fields["eng_pct"],
        },
    }
    lead = _shap_lead(summary["best"])
    if lead:
        facts["shap"] = {"lead_sentence": lead}
    return facts


# ---------------------------------------------------------------------------
# Run facts
# ---------------------------------------------------------------------------

_FOLD_LABEL = "第 {n} 折"


def build_run_facts(
    run: dict[str, Any],
    context: dict[str, Any],
    best: dict[str, Any] | None = None,
) -> tuple[str, dict[str, Any]]:
    """Facts for one model's sub-report, plus which template to render.

    The template choice is the family: a tree model has no epochs, so asking it
    for a convergence section produced a heading with "上下文中没有这项数据"
    under it. It has cross-fold behaviour instead, which is just as much a
    training process.
    """
    task = context.get("task") or {}
    metric = str(task.get("objective_metric") or "score").lower()
    name = metric_name(metric)
    stats = context.get("_target_stats") or {}
    mean = stats.get("mean")
    target = task.get("target_column") or "目标列"
    metrics = run.get("metrics") or {}
    folds = metrics.get("cv_folds") if isinstance(metrics.get("cv_folds"), list) else []
    history = metrics.get("history") if isinstance(metrics.get("history"), list) else []

    value = entry_metric(run, metric)
    board = context.get("leaderboard") or []
    overall_best = best or (board[0] if board else run)
    scheme = validation_scheme(run)
    comparable = [entry for entry in board if validation_scheme(entry) == scheme]
    best = comparable[0] if comparable else run
    best_value = entry_metric(best, metric)
    best_std = entry_std(best, metric)
    own_std = entry_std(run, metric)
    gap = (abs(value - best_value)
           if isinstance(value, (int, float)) and isinstance(best_value, (int, float))
           else None)
    is_best = _same_run(run, best)
    # A rerun of the winning model scores identically, and comparing it to the
    # champion produced "与最优模型 xgboost_regressor（72.4673）相差 0，相对差 0%"
    # — the model measured against itself under another name.
    is_duplicate = (not is_best
                    and run.get("model_type") == best.get("model_type")
                    and value == best_value)
    verdict = gap_verdict(gap, own_std or best_std)

    metrics_sentence = _metrics_sentence(run, metric, target, mean)

    facts: dict[str, Any] = {
        "run": {
            "model": run.get("model_type"),
            "strategy": run.get("strategy_type") or "baseline",
            "params_note": _params_note(run),
        },
        "headline": {"sentence": (
            # is_cv, not `folds`: the rank-1 run has a cross-validated mean but
            # no per-fold detail persisted, and was labelled 留出验证 for it.
            (f"{_metric_label(metric, scheme)} {readable(value, metric)}"
             + (f"，占 {target} 均值的 {pct_text(value, mean)}。"
                if mean and is_error_metric(metric) else "。"))
            if isinstance(value, (int, float)) else ""
        )},
        "metrics": {"sentence": metrics_sentence},
        "error_shape": error_shape(
            metrics.get("cv_avg_rmse") or metrics.get("selection_cv_mean_rmse"),
            metrics.get("cv_avg_mae") or metrics.get("selection_cv_mean_mae"),
        ),
    }

    if is_best:
        facts["gap"] = {"sentence": "本模型即本次最优。"}
    elif is_duplicate:
        facts["gap"] = {"sentence": (
            f"本次结果与排名第一的 {best.get('model_type')} 完全一致，"
            "为同一模型的重复训练，不构成独立的对比项。"
        )}
    elif verdict.get("known"):
        facts["gap"] = {"sentence": (
            f"与最优的 {best.get('model_type')} 相差 {readable(gap, metric)}，"
            f"小于折间标准差 {readable(own_std or best_std, metric)}，两者分不出高下。"
            if verdict["within_noise"] else
            f"与最优的 {best.get('model_type')} 相差 {readable(gap, metric)}，"
            f"相对差 {pct_text(gap, best_value)}，超过折间波动，这个落差是真实的。"
        )}
    elif isinstance(gap, (int, float)):
        facts["gap"] = {"sentence": (
            f"与最优的 {best.get('model_type')}（{readable(best_value, metric)}）相差 {readable(gap, metric)}。"
        )}

    # A run measured under a different scheme than the overall winner is still
    # ranked when the gap dwarfs the fold noise; only a close call is left open.
    if validation_scheme(overall_best) != scheme and not _same_run(run, overall_best):
        facts.setdefault("gap", {})["caveat"] = _cross_scheme_sentence(
            run, overall_best, metric, is_error_metric(metric),
        )
        if is_best:
            facts["gap"]["sentence"] = f"本模型是{scheme}组里最好的。"

    if is_cv(run) and not folds:
        facts["validation"] = {"summary_sentence": (
            "该 Run 记录了交叉验证汇总指标，但未保存逐折明细；"
            "因此可以用于同口径组内排序，不能据此分析折间离群或稳定性。"
        )}
    elif not folds:
        facts["validation"] = {"summary_sentence": (
            f"该 Run 采用{scheme}，未提供可展示的逐折训练记录。"
        )}

    top_shap = metrics.get("top_shap_importances") or []
    if len(top_shap) >= 2:
        first, second = top_shap[0], top_shap[1]
        ratio = (abs(first.get("mean_abs_shap", 0)) / abs(second["mean_abs_shap"])
                 if second.get("mean_abs_shap") else None)
        facts["shap"] = {
            "top_feature": first.get("feature"),
            "concentration_sentence": (
                f"特征贡献高度集中：首位 {first.get('feature')} 为 "
                f"{_fmt(first.get('mean_abs_shap'), 1)}，是次位 {second.get('feature')} 的 "
                f"{round(ratio, 1)} 倍。"
                if ratio else ""
            ),
        }
        best_top = ((best.get("metrics") or {}).get("top_shap_importances") or [{}])[0]
        if not is_best and best_top.get("feature") == first.get("feature"):
            facts["shap"]["vs_best_sentence"] = (
                f"与 {best.get('model_type')} 的特征结构基本一致，同样由 {first.get('feature')} 主导，"
                "故二者的性能差异更可能来自拟合细节，而非对特征的利用方式不同。"
            )

    if folds:
        return "run_ml", _merge(facts, _fold_facts(folds, metric))
    if history:
        return "run_dl", _merge(facts, _history_facts(history, metric, run))
    return "run_ml", facts


def _run_metric(metrics: dict[str, Any], key: str, cv: bool) -> float | None:
    """One run's own value for `key`, from the keys its validation scheme writes.

    Cross-validated runs record cv_avg_* / selection_cv_mean_*; hold-out runs
    record val_* / selection_val_*. Reading only the first pair left every
    deep-learning sub-report with a single number and no MAE or R².
    """
    names = ((f"cv_avg_{key}", f"selection_cv_mean_{key}") if cv
             else (f"val_{key}", f"selection_val_{key}"))
    for name in names:
        value = metrics.get(name)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
    return None


def _metrics_sentence(run: dict[str, Any], metric: str, target: str, mean: Any) -> str:
    """The scores in the mock's shape: "RMSE a、MAE b，分别占 T 均值的 p 和 q；R² c。"

    Errors first, with their share of the target mean where there is one;
    scores after a semicolon. Prefixed once with the validation scheme rather
    than once per number.
    """
    metrics = run.get("metrics") or {}
    cv = is_cv(run)
    errors = [(k, _run_metric(metrics, k, cv)) for k in ("rmse", "mae")]
    errors = [(k, v) for k, v in errors if v is not None]
    scores = [(k, _run_metric(metrics, k, cv)) for k in ("r2", "accuracy", "f1", "roc_auc")]
    scores = [(k, v) for k, v in scores if v is not None]
    if not errors and not scores:
        value = entry_metric(run, metric)
        if not isinstance(value, (int, float)):
            return ""
        (errors if is_error_metric(metric) else scores).append((metric, float(value)))

    parts: list[str] = []
    if errors:
        names = "、".join(f"{metric_name(k)} {readable(v, k)}" for k, v in errors)
        if mean:
            shares = " 和 ".join(pct_text(v, mean) for _, v in errors)
            names += f"，{'分别' if len(errors) > 1 else ''}占 {target} 均值的 {shares}"
        parts.append(names)
    if scores:
        parts.append("、".join(f"{metric_name(k)} {readable(v, k)}" for k, v in scores))
    prefix = "交叉验证 " if cv else "验证 "
    return prefix + "；".join(parts) + "。"


def _same_run(a: dict[str, Any], b: dict[str, Any]) -> bool:
    """The same run, by identity or by a shared run_id.

    Comparing `a.get("run_id") == b.get("run_id")` made any two entries
    without an id the same run — None equals None — so a hold-out run assembled
    without an id was declared the overall best.
    """
    if a is b:
        return True
    run_id = a.get("run_id")
    return run_id is not None and run_id == b.get("run_id")


def _cross_scheme_sentence(run: dict[str, Any], overall_best: dict[str, Any],
                           metric: str, error: bool) -> str:
    value = entry_metric(run, metric)
    best_value = entry_metric(overall_best, metric)
    best_scheme = validation_scheme(overall_best)
    name = overall_best.get("model_type")
    noise = entry_std(overall_best, metric) or entry_std(run, metric)
    if not isinstance(value, (int, float)) or not isinstance(best_value, (int, float)):
        return (f"注意口径不同：本模型采用{validation_scheme(run)}，总榜首 {name} 采用{best_scheme}；"
                "两种口径不直接比较。")
    gap = abs(value - best_value)
    if noise and gap > _DECISIVE_GAP_IN_STD * noise:
        if error and best_value:
            size = f"本模型的误差是它的 {_fmt(value / best_value, 1)} 倍"
        else:
            size = f"本模型落后 {readable(gap, metric)}"
        return (f"总榜首 {name} 的{best_scheme}成绩是 {readable(best_value, metric)}，{size}；"
                "口径不同，但差距远超折间波动，落后是确定的。")
    return (f"注意口径不同：本模型采用{validation_scheme(run)}，总榜首 {name} 采用{best_scheme}；"
            "差距落在折间波动之内，两种口径不直接比较，排不出先后。")


def _merge(base: dict[str, Any], extra: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in extra.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = {**out[key], **value}
        else:
            out[key] = value
    return out


def _params_note(run: dict[str, Any]) -> str:
    """A clause: "使用默认超参数" or "参数设置为 max_depth=12、…"."""
    params = (run.get("params") or {}).get("hyperparameters") or {}
    if not params:
        return "使用默认超参数"
    return "参数设置为 " + "、".join(f"{k}={v}" for k, v in list(params.items())[:6])


def fold_values(folds: list[dict[str, Any]], metric: str) -> tuple[str | None, list[float]]:
    """The objective metric's per-fold values, or the first numeric key's."""
    if not folds:
        return None, []
    numeric = [k for k in (folds[0] or {})
               if k != "fold" and isinstance(folds[0][k], (int, float))]
    key = metric if metric in numeric else (numeric[0] if numeric else None)
    if key is None:
        return None, []
    return key, [f.get(key) for f in folds if isinstance(f.get(key), (int, float))]


def _fold_facts(folds: list[dict[str, Any]], metric: str) -> dict[str, Any]:
    key, values = fold_values(folds, metric)
    if not values or len(values) < 3:
        return {}
    spread = spread_verdict(values)
    order = sorted(range(len(values)), key=lambda i: values[i])
    # For an error metric the smallest value is the best fold; for a score it is
    # the largest. Getting this backwards labels the worst fold "最好".
    best_i, worst_i = (order[0], order[-1]) if key in _ERROR_METRICS else (order[-1], order[0])
    mean = spread["mean"]
    label = metric_name(key)
    n = cn_count(len(folds))
    return {
        "cv": {
            "scheme": f"{len(folds)} 折交叉验证",
            "metric": label,
            "range": readable(spread["range"], key),
            "range_pct": pct_text(spread["range"], mean),
            "cv_pct": pct_text(spread["std"], mean),
            "best_fold": _FOLD_LABEL.format(n=folds[best_i].get("fold")),
            "best_value": readable(values[best_i], key),
            "worst_fold": _FOLD_LABEL.format(n=folds[worst_i].get("fold")),
            "worst_value": readable(values[worst_i], key),
            "spread_note": spread["note"],
            # The fold range is the chart caption's line; this paragraph adds
            # the spread numbers and names the two extreme folds, and stays at
            # two sentences so the model's one fits under the paragraph cap.
            "summary_sentence": (
                f"折间标准差 {readable(spread['std'], key)}，变异系数 {pct_text(spread['std'], mean)}。"
                f"最差的{_FOLD_LABEL.format(n=folds[worst_i].get('fold'))}与最好的"
                f"{_FOLD_LABEL.format(n=folds[best_i].get('fold'))}之间{spread['note']}。"
            ),
            "verdict_sentence": (
                "折间波动为整体性，无单折离群，划分稳定性良好。"
                if not spread["outlier"] else
                "存在单折离群，提示数据划分不均，需检查切分方式。"
            ),
        },
    }


def history_series(history: list[dict[str, Any]], metric: str) -> dict[str, Any]:
    """The validation curve, the training curve, and where the best epoch is."""
    def _series(*names) -> tuple[str, list[tuple[int, float]]]:
        for name in names:
            values = [(i + 1, r.get(name)) for i, r in enumerate(history)
                      if isinstance(r.get(name), (int, float))]
            if values:
                return name, values
        return "", []

    val_name, val = _series(f"val_{metric}", "val_loss", "valid_loss")
    _, train = _series("train_loss", "loss")
    if not val:
        return {}
    best_epoch, best_value = min(val, key=lambda p: p[1])
    # Two questions about the epochs after the best one: is the training loss
    # still coming down, and did the validation loss stay above its minimum?
    # Both together are the two curves parting company. Comparing the first and
    # last validation values *after* the best epoch missed the common shape
    # where validation jumps once and then sits flat while training keeps
    # falling — the gap widens every epoch and the old rule called it a plateau.
    after_train = [v for e, v in train if e > best_epoch]
    after_val = [v for e, v in val if e > best_epoch]
    train_falling = bool(after_train and after_train[-1] < after_train[0] * (1 - 0.01))
    val_stayed_up = bool(after_val and after_val[-1] > best_value + abs(best_value) * 0.01)
    return {
        "val_name": val_name,
        "val": val,
        "train": train,
        "best_epoch": best_epoch,
        "best_value": best_value,
        "ran": val[-1][0],
        "train_falling": train_falling,
        "overfit": train_falling and val_stayed_up,
    }


def _history_facts(history: list[dict[str, Any]], metric: str,
                   run: dict[str, Any] | None = None) -> dict[str, Any]:
    series = history_series(history, metric)
    if not series:
        return {}

    # The curve is usually the raw loss, not the objective metric — labelling
    # it "验证 RMSE 17435.13" put a squared quantity next to an RMSE of 132.04
    # and invited the reader to think the model was a thousand times worse than
    # it is.
    label = metric_name(metric) if series["val_name"].endswith(metric) else "损失"
    # "最优验证RMSE" needs the spaces a Chinese word does not.
    phrase = f" {label} " if label.isascii() else label

    best_epoch, best_value, ran = series["best_epoch"], series["best_value"], series["ran"]
    planned = _planned_epochs(run)
    patience = _as_int(_train_config(run).get("early_stopping_patience"))
    # "计划训练 38 轮，实际在第 38 轮触发早停" — planned was just len(history),
    # so it always equalled the actual count and the sentence said nothing.
    stopped_early = bool(planned and ran < planned)

    return {
        "run": {"arch_note": _arch_note(run)},
        "train": {
            "plan_note": (f"计划训练 {planned} 轮，" if planned else ""),
            "actual_epochs": ran,
            # With no configured epoch count there is no way to tell an early
            # stop from a completed run, and "训练结束" asserted the wrong one
            # for a run whose best epoch was ten short of its last.
            "stop_reason": (
                f"触发早停（早停耐心 {patience} 轮）" if stopped_early and patience
                else "触发早停" if stopped_early
                else "训练结束" if planned else "结束"
            ),
            "metric": label,
            "metric_phrase": phrase,
            "best_epoch": best_epoch,
            # Prose precision: 132, not 132.0422; the exact value is in the
            # chart rows.
            "best_value": readable(best_value, metric if label != "损失" else None),
            "patience_used": ran - best_epoch,
            "overfit_note": (
                "训练损失在此之后继续下降而验证损失回升，两条线分开的位置即过拟合起点。"
                if series["overfit"] else ""
            ),
            "verdict_sentence": (
                f"训练在第 {ran} 轮结束，最优验证{phrase}出现在第 {best_epoch} 轮。"
            ),
        },
    }


def _as_int(value: Any) -> int | None:
    """Nested config arrives stringified.

    _compact_value turns every leaf past depth three into str(), so the epoch
    budget reaches here as "50" and an isinstance(int) check silently declines
    it — which is how a 38-of-50 early stop was reported as a completed run.
    """
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    try:
        return int(float(str(value).strip()))
    except (TypeError, ValueError):
        return None


def _as_numbers(value: Any) -> list[int]:
    """The layer widths, whether they arrive as a list or as "[256, 128]"."""
    if isinstance(value, (list, tuple)):
        return [n for n in (_as_int(v) for v in value) if n is not None]
    return [int(n) for n in re.findall(r"\d+", str(value or ""))]


def _train_config(run: dict[str, Any] | None) -> dict[str, Any]:
    hyper = ((run or {}).get("params") or {}).get("hyperparameters") or {}
    return hyper.get("train_config") or {}


def _arch_note(run: dict[str, Any] | None) -> str:
    """What this network actually is, from the recorded architecture."""
    hyper = ((run or {}).get("params") or {}).get("hyperparameters") or {}
    arch = hyper.get("arch_config") or {}
    bits: list[str] = []
    layers = _as_int(arch.get("num_layers"))
    if layers:
        bits.append(f"{layers} 层")
    hidden = _as_int(arch.get("hidden_size"))
    if hidden:
        bits.append(f"隐藏层 {hidden} 维")
    # "×".join over the *string* "[256, 128]" produced "[×2×5×6×,× ×1×2×8×]".
    widths = _as_numbers(arch.get("hidden_layers"))
    if widths:
        bits.append("隐藏层 " + "×".join(str(n) for n in widths))
    if arch.get("dropout"):
        bits.append(f"dropout {arch['dropout']}")
    batch = _as_int(_train_config(run).get("batch_size"))
    if batch:
        bits.append(f"批量 {batch}")
    return "采用 " + "、".join(bits) + " 的配置，" if bits else ""


def _planned_epochs(run: dict[str, Any] | None) -> int | None:
    """The configured epoch budget, which is not the number of rows in history."""
    params = ((run or {}).get("params") or {})
    hyper = params.get("hyperparameters") or {}
    # The DL trainers nest it under train_config; searching only the flat level
    # found nothing, so a 38-of-50 early stop was reported as a completed run.
    for source in (hyper.get("train_config") or {}, hyper, params):
        for key in ("epochs", "max_epochs", "n_epochs", "num_epochs"):
            value = _as_int(source.get(key))
            if value and value > 0:
                return value
    return None

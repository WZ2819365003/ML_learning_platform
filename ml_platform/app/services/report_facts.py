"""Every number and verdict a report asserts, computed from the context.

Nothing here is left to the model. That includes the judgements — whether a gap
between two models is meaningful, whether cross-fold spread is stable, whether
an error distribution has a tail — because each is a comparison the code can do
exactly and the model can only estimate.
"""

from __future__ import annotations

import re
from typing import Any

# Names a feature-engineering step produces, as opposed to a column that came
# with the data. Grouped so the dataset section can describe composition rather
# than list 35 column names.
_FEATURE_GROUPS: tuple[tuple[str, str, str], ...] = (
    ("周期三角变换", r"(_sin|_cos)$", "把环形的时间关系显式交给模型"),
    ("滞后项", r"_lag_\d+$", "引入历史时刻的取值"),
    ("滚动统计", r"_roll_", "描述近期水平与波动"),
    ("气象与交互衍生", r"(_x_|_degree$|_index$|_depression$|_spread_|_sq$|^days_since_)", "由原始观测量派生"),
)

_ERROR_METRICS = ("rmse", "mae", "mse", "mape")

# RMSE/MAE for a normal error distribution. The ratio is always ≥ 1, so the
# number alone says nothing; only the distance from this baseline does.
_NORMAL_RMSE_MAE = 1.2533

_CHECK_NAMES = {
    "final_evaluation": "最终评估",
    "cross_fold_stability": "跨折稳定性",
    "run_success": "Run 全部成功",
}

_METRIC_NAMES = {"r2": "R²", "rmse": "RMSE", "mae": "MAE", "mse": "MSE",
                 "mape": "MAPE", "accuracy": "准确率", "f1": "F1"}


def _fmt(value: Any, digits: int = 4) -> str:
    """Round to `digits`, then drop only trailing zeros.

    ":g" switches to six significant figures, which silently turned 14274.15
    into 14274.1 — a report is not the place to lose a digit of a stated range.
    """
    if not isinstance(value, (int, float)):
        return "—"
    text = f"{float(value):.{digits}f}".rstrip("0").rstrip(".")
    return text or "0"


def _pct(value: float, base: float, digits: int = 2) -> str:
    return f"{round(abs(value) / abs(base) * 100, digits):g}%"


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


def md_table(headers: list[str], rows: list[list[str]], align: list[str] | None = None) -> str:
    align = align or ["---"] * len(headers)
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join(align) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


def gap_verdict(gap: float | None, noise: float | None,
                noise_name: str = "交叉验证噪声") -> dict[str, Any]:
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
        "short": "差距不具备统计意义" if within else "差距可辨识",
        "long": (
            f"差距落在{noise_name}之内，当前数据不足以判定二者优劣"
            if within else f"差距超出{noise_name}，可以判定优劣"
        ),
    }


def spread_verdict(values: list[float]) -> dict[str, Any]:
    """Is one fold an outlier, or is the whole thing wobbling?

    Different problems with different fixes: a single bad fold means the split
    is uneven, general spread means the model is unstable. The test is whether
    the worst fold sits more than two standard deviations from the mean.
    """
    if len(values) < 3:
        return {"known": False}
    mean = sum(values) / len(values)
    var = sum((v - mean) ** 2 for v in values) / len(values)
    std = var ** 0.5
    worst = max(values, key=lambda v: abs(v - mean))
    outlier = std > 0 and abs(worst - mean) > 2 * std
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
# Fact bundles, one per template
# ---------------------------------------------------------------------------

_FAMILY_ML = "ml"


def _metric_label(key: str | None, scheme: str) -> str:
    return f"{scheme} {_METRIC_NAMES.get(str(key or '').lower(), str(key or 'score').upper())}"


def _entry_metric(entry: dict[str, Any], key: str) -> Any:
    metrics = entry.get("metrics") or {}
    for candidate in (f"cv_avg_{key}", f"selection_cv_mean_{key}", key):
        value = metrics.get(candidate)
        if isinstance(value, (int, float)):
            return value
    return entry.get("objective_value")


def _is_cv(entry: dict[str, Any]) -> bool:
    return isinstance((entry.get("metrics") or {}).get("cv_avg_r2"), (int, float)) or isinstance(
        (entry.get("metrics") or {}).get("cv_avg_rmse"), (int, float)
    )


def _runs_summary(total: int, batches: int, counts: dict[str, Any],
                  cv_n: int, other_n: int) -> str:
    failed = counts.get("FAILED", 0) or 0
    outcome = "全部成功" if not failed else f"其中失败 {failed} 个"
    text = f"{total} 个 Run 分 {batches} 批执行，{outcome}。"
    if cv_n and other_n:
        text += f"其中 {cv_n} 个采用交叉验证，{other_n} 个采用留出验证。"
    return text


def build_overview_facts(context: dict[str, Any]) -> dict[str, Any]:
    task = context.get("task") or {}
    dataset = context.get("dataset") or {}
    board = context.get("leaderboard") or []
    metric = str(task.get("objective_metric") or "score").lower()
    stats = context.get("_target_stats") or {}
    mean = stats.get("mean")

    if not board:
        return {"task": {"name": task.get("name") or "建模任务"}}

    best = board[0]
    best_value = _entry_metric(best, metric)
    best_std = (best.get("metrics") or {}).get(f"cv_std_{metric}")

    # Two runs of the same model with identical scores read as a two-horse race
    # until someone notices they are the same horse.
    dup = [e for e in board[1:] if e.get("model_type") == best.get("model_type")
           and _entry_metric(e, metric) == best_value]
    # Rank order still has the duplicates in it, so board[2] is not the third
    # *model* — it was the runner-up again, reported as "落后 0.3183" from itself.
    distinct = [best] + [e for e in board[1:] if e not in dup]
    runner = distinct[1] if len(distinct) > 1 else None
    runner_value = _entry_metric(runner, metric) if runner else None
    gap = (abs(runner_value - best_value)
           if isinstance(runner_value, (int, float)) and isinstance(best_value, (int, float))
           else None)
    verdict = gap_verdict(gap, best_std)

    cv_rows = [e for e in board if _is_cv(e)]
    other = [e for e in board if not _is_cv(e)]

    rows = []
    for e in board:
        v = _entry_metric(e, metric)
        rows.append([
            str(e.get("rank") or "—"),
            str(e.get("model_type") or "—"),
            _fmt(v),
            _pct(v, mean) if isinstance(v, (int, float)) and mean else "—",
            _fmt((e.get("metrics") or {}).get("cv_avg_r2")),
            _fmt((e.get("metrics") or {}).get(f"cv_std_{metric}")),
            "交叉验证" if _is_cv(e) else "留出验证",
        ])

    counts = context.get("run_status_counts") or {}
    total = sum(counts.values()) or len(board)
    batches = len(context.get("experiments") or [])

    third = distinct[2] if len(distinct) > 2 else None
    third_value = _entry_metric(third, metric) if third else None

    # column_names carries all of them; columns_info is capped at sixteen.
    columns = (dataset.get("column_names")
               or list((dataset.get("columns_info") or {}).keys()))
    fields = classify_columns(columns, task.get("target_column"))
    group_text = "；".join(
        f"{g['name']} {len(g['columns'])} 列（{'、'.join(g['columns'])}）"
        for g in fields["groups"]
    )

    shap = ((best.get("metrics") or {}).get("top_shap_importances") or [])[:2]
    readiness = context.get("_readiness") or {}
    failed = [c for c in (readiness.get("checks") or []) if not c.get("passed")]

    facts: dict[str, Any] = {
        "task": {"name": task.get("name") or "建模任务"},
        "best": {
            "model": best.get("model_type"),
            "metric_label": _metric_label(metric, "交叉验证" if _is_cv(best) else "留出验证"),
            "value": _fmt(best_value),
            "pct_of_mean": _pct(best_value, mean) if mean else "—",
            "fold_std": _fmt(best_std),
        },
        "runs": {"summary": _runs_summary(total, batches, counts, len(cv_rows), len(other))},
        "readiness": {
            "score": readiness.get("score"),
            "gap_note": ("未达成项：" + "、".join(
                _CHECK_NAMES.get(c.get("key"), c.get("label", "")) for c in failed)
                if failed else "各项均已达成"),
            "rubric": (
                "就绪评分满分 100，由"
                + "、".join(
                    _CHECK_NAMES.get(c.get("key"), c.get("label", "")) + f"（{c.get('weight')}）"
                    for c in (readiness.get("checks") or []))
                + "构成。" if readiness.get("checks") else ""
            ),
        },
        "tables": {
            "leaderboard": md_table(
                ["排名", "模型", metric.upper(), "占均值", "R²", "折间标准差", "口径"],
                rows,
                ["---:", "---", "---:", "---:", "---:", "---:", "---"],
            ),
            "fields": md_table(
                ["类别", "列数", "字段"],
                [["原始采集与日历", str(fields["base_count"]), "、".join(fields["base"])]]
                + [[g["name"], str(len(g["columns"])), "、".join(g["columns"])]
                   for g in fields["groups"]],
                ["---", "---:", "---"],
            ),
        },
        "ds": {
            "shape_sentence": (
                f"{dataset.get('row_count')} 行 × {dataset.get('column_count')} 列，"
                f"目标列 {task.get('target_column')}；"
                f"均值 {_fmt(mean, 2)}，取值范围 {_fmt(stats.get('min'), 2)}–{_fmt(stats.get('max'), 2)}。"
            ),
        },
        "fields": {
            "summary_sentence": (
                f"{len(columns)} 列中 {fields['base_count']} 列为原始采集与日历字段，"
                f"{fields['eng_count']} 列为训练流程构造的特征，构造特征占 {fields['eng_pct']}：{group_text}。"
                if fields["groups"] else ""
            ),
        },
    }

    if isinstance(best_value, (int, float)) and task.get("final_test_value") is None:
        facts["final_eval"] = {
            "sentence": "该成绩取自模型选择阶段，封存测试集上的最终评估尚未执行，泛化能力未经确认；"
        }
    else:
        facts["final_eval"] = {"sentence": ""}

    if dup and runner:
        facts["duplicates"] = {
            "note": f"排名前 {len(dup) + 1} 位为同一模型 {best.get('model_type')} 的多次独立训练，结果一致"
        }
        facts["runner_up"] = {"model": runner.get("model_type")}
        facts["gap"] = {"verdict_short": verdict.get("short", "")}

    if verdict.get("known") and runner:
        facts.setdefault("gap", {})["sentence"] = (
            f"{runner.get('model_type')} 与 {best.get('model_type')} 相差 {_fmt(gap)}，"
            f"而后者自身的折间标准差为 {_fmt(best_std)}——{verdict['long']}。"
        )
        if third and isinstance(third_value, (int, float)):
            facts["third"] = {"sentence": (
                f"{third.get('model_type')} 落后 {_fmt(abs(third_value - best_value))}。"
            )}

    if cv_rows and other:
        facts["families"] = {"caveat": (
            "两族分数口径不同：交叉验证为多折均值，留出验证为单次结果，"
            "二者不宜直接横向比较，表中排名仅供参考。"
        )}

    if len(shap) >= 2 and isinstance(shap[0].get("mean_abs_shap"), (int, float)):
        top, second = shap[0], shap[1]
        ratio = (abs(top["mean_abs_shap"]) / abs(second["mean_abs_shap"])
                 if second.get("mean_abs_shap") else None)
        facts["shap"] = {"evidence_sentence": (
            f"最优模型的 SHAP 首位为 {top.get('feature')}，平均绝对贡献 {_fmt(top['mean_abs_shap'], 1)}，"
            + (f"是次位 {second.get('feature')}（{_fmt(second['mean_abs_shap'], 1)}）的 "
               f"{round(ratio, 1)} 倍。" if ratio else "。")
        )}
    return facts


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
    stats = context.get("_target_stats") or {}
    mean = stats.get("mean")
    metrics = run.get("metrics") or {}
    folds = metrics.get("cv_folds") if isinstance(metrics.get("cv_folds"), list) else []
    history = metrics.get("history") if isinstance(metrics.get("history"), list) else []

    value = _entry_metric(run, metric)
    best = best or ((context.get("leaderboard") or [{}])[0])
    best_value = _entry_metric(best, metric)
    best_std = (best.get("metrics") or {}).get(f"cv_std_{metric}")
    own_std = metrics.get(f"cv_std_{metric}")
    gap = (abs(value - best_value)
           if isinstance(value, (int, float)) and isinstance(best_value, (int, float))
           else None)
    is_best = run.get("run_id") == best.get("run_id")
    # A rerun of the winning model scores identically, and comparing it to the
    # champion produced "与最优模型 xgboost_regressor（72.4673）相差 0，相对差 0%"
    # — the model measured against itself under another name.
    is_duplicate = (not is_best
                    and run.get("model_type") == best.get("model_type")
                    and value == best_value)
    noise_name = "交叉验证噪声" if _is_cv(run) else "最优模型的折间波动"
    verdict = gap_verdict(gap, own_std or best_std, noise_name)

    headline = []
    for key in ("rmse", "mae", "r2", "accuracy", "f1"):
        v = metrics.get(f"cv_avg_{key}") or metrics.get(f"selection_cv_mean_{key}")
        if not isinstance(v, (int, float)):
            continue
        name = _METRIC_NAMES.get(key, key.upper())
        label = f"交叉验证 {name}" if folds else name
        if key in _ERROR_METRICS and mean:
            headline.append(f"{label} {_fmt(v)}（占目标列均值 {_pct(v, mean)}）")
        else:
            headline.append(f"{label} {_fmt(v)}")
    if not headline and isinstance(value, (int, float)):
        headline.append(
            f"验证 {metric.upper()} {_fmt(value)}"
            + (f"（占目标列均值 {_pct(value, mean)}）" if mean else "")
        )

    facts: dict[str, Any] = {
        "run": {
            "model": run.get("model_type"),
            "strategy": run.get("strategy_type") or "baseline",
            "params_note": _params_note(run),
        },
        "headline": {"sentence": (
            # _is_cv, not `folds`: the rank-1 run has a cross-validated mean but
            # no per-fold detail persisted, and was labelled 留出验证 for it.
            (f"{_metric_label(metric, '交叉验证' if _is_cv(run) else '留出验证')} {_fmt(value)}，"
             f"列第 {run.get('rank')}。") if isinstance(value, (int, float)) else ""
        )},
        "metrics": {"sentence": "，".join(headline) + "。" if headline else ""},
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
            f"与最优模型 {best.get('model_type')}（{_fmt(best_value)}）相差 {_fmt(gap)}，"
            f"相对差 {_pct(gap, best_value)}；该差距小于本模型自身的折间标准差 "
            f"{_fmt(own_std or best_std)}，{verdict['long']}。"
            if verdict["within_noise"] else
            f"与最优模型 {best.get('model_type')}（{_fmt(best_value)}）相差 {_fmt(gap)}，"
            f"相对差 {_pct(gap, best_value)}，{verdict['long']}。"
        )}
    elif isinstance(gap, (int, float)):
        facts["gap"] = {"sentence": (
            f"与最优模型 {best.get('model_type')}（{_fmt(best_value)}）相差 {_fmt(gap)}。"
        )}

    if not folds and (best.get("metrics") or {}).get(f"cv_std_{metric}") is not None:
        facts.setdefault("gap", {})["caveat"] = (
            "注意口径不同：本模型的分数来自留出验证集单次结果，最优模型的来自多折交叉验证均值。"
        )

    top_shap = metrics.get("top_shap_importances") or []
    if len(top_shap) >= 2:
        first, second = top_shap[0], top_shap[1]
        ratio = (abs(first.get("mean_abs_shap", 0)) / abs(second["mean_abs_shap"])
                 if second.get("mean_abs_shap") else None)
        facts["shap"] = {
            "top_feature": first.get("feature"),
            "concentration_sentence": (
                f"特征贡献高度集中：首位 {first.get('feature')} 为 "
                f"{_fmt(first.get('mean_abs_shap'), 1)}，次位 {second.get('feature')} "
                f"{_fmt(second.get('mean_abs_shap'), 1)}，相差 {round(ratio, 1)} 倍。"
                if ratio else ""
            ),
        }
        facts["tables"] = {"shap": md_table(
            ["特征", "平均绝对 SHAP"],
            [[str(f.get("feature")), _fmt(f.get("mean_abs_shap"), 1)] for f in top_shap[:6]],
            ["---", "---:"],
        )}
        best_top = ((best.get("metrics") or {}).get("top_shap_importances") or [{}])[0]
        if not is_best and best_top.get("feature") == first.get("feature"):
            facts["shap"]["vs_best_sentence"] = (
                f"与 {best.get('model_type')} 的特征结构基本一致，同样由 {first.get('feature')} 主导，"
                "故二者的性能差异更可能来自拟合细节，而非对特征的利用方式不同。"
            )

    # A shallow merge drops facts["tables"]["shap"] the moment the family
    # bundle also carries a "tables" key — the SHAP table just stops rendering.
    if folds:
        return "run_ml", _merge(facts, _fold_facts(folds, metric))
    if history:
        return "run_dl", _merge(facts, _history_facts(history, metric, run))
    return "run_ml", facts


def _merge(base: dict[str, Any], extra: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in extra.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = {**out[key], **value}
        else:
            out[key] = value
    return out


def _params_note(run: dict[str, Any]) -> str:
    params = (run.get("params") or {}).get("hyperparameters") or {}
    if not params:
        return "默认超参数"
    return "、".join(f"{k}={v}" for k, v in list(params.items())[:6])


def _fold_facts(folds: list[dict[str, Any]], metric: str) -> dict[str, Any]:
    values = [f.get(metric) for f in folds if isinstance(f.get(metric), (int, float))]
    if not values:
        return {}
    spread = spread_verdict(values)
    order = sorted(range(len(values)), key=lambda i: values[i])
    # For an error metric the smallest value is the best fold; for a score it is
    # the largest. Getting this backwards labels the worst fold "最好".
    best_i, worst_i = (order[0], order[-1]) if metric in _ERROR_METRICS else (order[-1], order[0])
    keys = [k for k in (folds[0] or {}) if k != "fold" and isinstance(folds[0][k], (int, float))]
    # The objective metric leads: it is the one the ranking and the prose use.
    keys.sort(key=lambda k: (k != metric, k))
    rows = [[_FOLD_LABEL.format(n=f.get("fold"))] + [_fmt(f.get(k)) for k in keys] for f in folds]
    heads = [_METRIC_NAMES.get(k, k.upper()) for k in keys]
    mean = spread["mean"]
    rows.append(["**均值**"] + [
        "**" + _fmt(sum(f[k] for f in folds if isinstance(f.get(k), (int, float))) / len(folds)) + "**"
        for k in keys
    ])
    return {
        "cv": {
            "scheme": f"{len(folds)} 折交叉验证",
            "metric": metric.upper(),
            "range": _fmt(spread["range"]),
            "range_pct": _pct(spread["range"], mean),
            "cv_pct": _pct(spread["std"], mean),
            "best_fold": _FOLD_LABEL.format(n=folds[best_i].get("fold")),
            "best_value": _fmt(values[best_i]),
            "worst_fold": _FOLD_LABEL.format(n=folds[worst_i].get("fold")),
            "worst_value": _fmt(values[worst_i]),
            "spread_note": spread["note"],
            "verdict_sentence": (
                "折间波动为整体性，无单折离群，划分稳定性良好。"
                if not spread["outlier"] else
                "存在单折离群，提示数据划分不均，需检查切分方式。"
            ),
        },
        "tables": {"folds": md_table(["折"] + heads, rows,
                                     ["---:"] + ["---:"] * len(keys))},
    }


def _history_facts(history: list[dict[str, Any]], metric: str,
                   run: dict[str, Any] | None = None) -> dict[str, Any]:
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

    # The curve is usually the raw loss, not the objective metric — labelling
    # it "验证 RMSE 17435.13" put a squared quantity next to an RMSE of 132.04
    # and invited the reader to think the model was a thousand times worse than
    # it is.
    label = _METRIC_NAMES.get(metric, metric.upper()) if val_name.endswith(metric) else "损失"

    best_epoch, best_value = min(val, key=lambda p: p[1])
    ran = val[-1][0]
    planned = _planned_epochs(run)
    patience = _as_int(_train_config(run).get("early_stopping_patience"))
    # "计划训练 38 轮，实际在第 38 轮触发早停" — planned was just len(history),
    # so it always equalled the actual count and the sentence said nothing.
    stopped_early = bool(planned and ran < planned)

    overfit = ""
    if train and len(train) > best_epoch:
        after_train = [v for e, v in train if e > best_epoch]
        after_val = [v for e, v in val if e > best_epoch]
        if after_train and after_val and after_train[-1] < after_train[0] and after_val[-1] > after_val[0]:
            overfit = "训练损失在此之后继续下降而验证损失回升，两条线分开的位置即过拟合起点。"

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
            "best_epoch": best_epoch,
            "best_value": _fmt(best_value),
            "patience_used": ran - best_epoch,
            "overfit_note": overfit,
            "verdict_sentence": (
                f"训练在第 {ran} 轮结束，最优验证{label}出现在第 {best_epoch} 轮。"
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
    return "采用" + "、".join(bits) + "的配置，" if bits else ""


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

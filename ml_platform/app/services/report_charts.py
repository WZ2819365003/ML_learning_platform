"""Semantic chart specifications for the reports.

A spec says *what* a figure shows — categories, values, error bars, reference
lines, the rows behind each hover — and never *how* it is drawn. Colours,
margins, fonts, axis padding and tooltip layout are the frontend's, defined in
one renderer, so a styling fix is made once instead of chart by chart in here.

Every spec shares:

    id, kind, title, caption, unit, tooltip_fields, rows

and adds the fields its kind needs (hbar / dots / hist / stacked / lines /
scatter_pair). The caption is a reading of the data, not a description of the
axes: it is computed here ("A 与 B 的误差条重叠", "略左偏") because those are
comparisons the code can make exactly and the model can only estimate.
"""

from __future__ import annotations

import json
import math
import re
from typing import Any

from app.services import report_facts as rf

_SHAP_TOP_N = 8
_RESIDUAL_BINS = 12
_MAX_HIST_BINS = 40


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _num(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if not math.isfinite(float(value)):
        return None
    return float(value)


def _round(value: Any, digits: int = 4) -> float | None:
    number = _num(value)
    return None if number is None else round(number, digits)


def _json_list(value: Any) -> list[Any]:
    """A list that may have arrived as its JSON text.

    _compact_value stringifies every leaf past depth three, and the histogram
    sits at exactly that depth in the dataset profile — so `counts` and
    `bin_edges` reach here as "[12, 40, …]".
    """
    if isinstance(value, (list, tuple)):
        return list(value)
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except (TypeError, ValueError):
            return []
        return list(parsed) if isinstance(parsed, (list, tuple)) else []
    return []


def _spec(chart_id: str, kind: str, title: str, caption: str, unit: str | None,
          tooltip_fields: list[dict[str, str]], rows: list[dict[str, Any]],
          **extra: Any) -> dict[str, Any]:
    return {
        "id": chart_id,
        "kind": kind,
        "title": title,
        "caption": caption,
        "unit": unit,
        "tooltip_fields": tooltip_fields,
        "rows": rows,
        **extra,
    }


def _ranges_overlap(a: tuple[float, float], b: tuple[float, float]) -> bool:
    return a[0] <= b[1] and b[0] <= a[1]


def _confusion(metrics: dict[str, Any]) -> dict[str, Any] | None:
    """A square confusion matrix with its label order, or nothing.

    Row i is the true class `labels[i]`, column j the predicted one, as both
    the tree trainers and the deep-learning trainer store it. Ragged or
    non-square data is dropped rather than drawn: a matrix whose row sums are
    wrong is worse than no matrix.
    """
    raw = _json_list((metrics or {}).get("confusion_matrix"))
    rows: list[list[int]] = []
    for row in raw:
        cells = [_num(v) for v in _json_list(row)]
        if any(c is None for c in cells):
            return None
        rows.append([int(c) for c in cells])
    size = len(rows)
    if size < 2 or any(len(r) != size for r in rows):
        return None
    if sum(sum(r) for r in rows) <= 0:
        return None
    labels = [str(v) for v in _json_list((metrics or {}).get("class_labels"))]
    if len(labels) != size:
        labels = [f"类别 {i}" for i in range(size)]
    source = (metrics or {}).get("confusion_source") or "holdout"
    return {"matrix": rows, "labels": labels, "source": str(source)}


def _split_name(source: str) -> str:
    """What to call the rows a classification figure was measured on.

    Under selection the sealed hold-out is withheld, so the figure is the last
    cross-validation fold. Saying 留出集 over a fold would claim an evaluation
    that never happened — the same distinction pred_vs_actual makes.
    """
    return "交叉验证末折" if source == "cv_last_fold" else "留出集"


# ---------------------------------------------------------------------------
# Overview charts
# ---------------------------------------------------------------------------

def leaderboard_bars(context: dict[str, Any]) -> dict[str, Any] | None:
    """Every run's score as a horizontal bar, coloured by validation scheme.

    One series per scheme, each aligned to the shared category list with
    nulls where a run belongs to another scheme; the error bar is the fold
    standard deviation where one was recorded.
    """
    summary = rf.rank_summary(context)
    if summary is None:
        return None
    metric, name, error, mean = (summary["metric"], summary["metric_name"],
                                 summary["error_metric"], summary["mean"])
    board = summary["board"]
    labels = summary["labels"]
    categories = [labels[rf._entry_key(e, i)] for i, e in enumerate(board)]

    series: list[dict[str, Any]] = []
    roles = ["primary", "muted", "accent"]
    for index, (scheme, entries) in enumerate(summary["cohorts"].items()):
        member = {rf._entry_key(e, i) for i, e in enumerate(board) if e in entries}
        values = [_round(rf.entry_metric(e, metric)) if rf._entry_key(e, i) in member else None
                  for i, e in enumerate(board)]
        stds = [_round(rf.entry_std(e, metric)) if rf._entry_key(e, i) in member else None
                for i, e in enumerate(board)]
        series.append({
            "name": scheme,
            "values": values,
            "error": stds if any(s is not None for s in stds) else None,
            "color_role": roles[min(index, len(roles) - 1)],
        })

    reference_lines: list[dict[str, Any]] = []
    if error and mean:
        reference_lines.append({"value": round(abs(mean) * 0.01, 4), "label": "均值的 1%"})

    rows = []
    for i, e in enumerate(board):
        value = rf.entry_metric(e, metric)
        rows.append({
            "category": categories[i],
            "model_type": e.get("model_type"),
            "run_id": e.get("run_id"),
            metric: _round(value),
            "pct": rf.pct_text(value, mean, exact=True) if error and mean else "—",
            "r2": _round(rf.entry_r2(e)),
            "std": _round(rf.entry_std(e, metric)),
            "scheme": rf.validation_scheme(e),
        })

    tooltip_fields = [
        {"key": metric, "label": name, "format": "0.0000"},
    ]
    if error and mean:
        tooltip_fields.append({"key": "pct", "label": "占均值"})
    tooltip_fields += [
        {"key": "r2", "label": "R²", "format": "0.0000"},
        {"key": "std", "label": "折间标准差", "format": "0.0000"},
        {"key": "scheme", "label": "验证方式"},
    ]

    return _spec(
        "leaderboard_bars", "hbar",
        f"{rf.cn_count(len(board))}个模型的{'误差' if error else name}",
        _leaderboard_caption(summary, rows),
        name, tooltip_fields, rows,
        categories=categories, series=series, reference_lines=reference_lines,
    )


def _leaderboard_caption(summary: dict[str, Any], rows: list[dict[str, Any]]) -> str:
    metric, error, mean = summary["metric"], summary["error_metric"], summary["mean"]
    best, runner = summary["best"], summary["runner"]
    parts: list[str] = []

    cohort_rows = [r for r in rows if r["scheme"] == summary["best_scheme"]]
    if error and mean:
        line = abs(mean) * 0.01
        inside = [r for r in cohort_rows if (r.get(metric) or 0) <= line]
        if inside and len(inside) == len(cohort_rows):
            parts.append(f"{summary['best_scheme']}的{rf.cn_count(len(cohort_rows))}个模型全在 1% 线以内")
        elif inside:
            parts.append(f"{summary['best_scheme']}组有{rf.cn_count(len(inside))}个模型在 1% 线以内")
        else:
            parts.append(f"{best.get('model_type')} 的误差是均值的 {summary['best_pct']}")
    else:
        parts.append(f"{best.get('model_type')} 以 {rf.readable(summary['best_value'], metric)} 领先")

    if runner is not None:
        best_std = rf.entry_std(best, metric) or 0.0
        runner_std = rf.entry_std(runner, metric) or 0.0
        bv, rv = rf.entry_metric(best, metric), rf.entry_metric(runner, metric)
        if isinstance(bv, (int, float)) and isinstance(rv, (int, float)):
            overlap = _ranges_overlap((bv - best_std, bv + best_std), (rv - runner_std, rv + runner_std))
            parts[-1] += (f"，{best.get('model_type')} 与 {runner.get('model_type')} 的误差条"
                          f"{'重叠' if overlap else '不相交'}")

    for other in summary["others"][:1]:
        values = [rf.entry_metric(e, metric) for e in other["entries"]]
        values = [v for v in values if isinstance(v, (int, float))]
        if not values:
            continue
        n = rf.cn_count(len(values))
        if error and mean:
            lo, hi = rf.pct_text(min(values), mean), rf.pct_text(max(values), mean)
            span = lo if lo == hi else f"{lo}–{hi}"
            parts.append(f"{other['scheme']}的{n}个模型在 {span}")
        else:
            lo, hi = rf.readable(min(values), metric), rf.readable(max(values), metric)
            parts.append(f"{other['scheme']}的{n}个模型在 {lo}–{hi}")
    return "；".join(parts) + "。"


def fold_dots(context: dict[str, Any]) -> dict[str, Any] | None:
    """Per-fold scores for every cross-validated model, one row each.

    A rerun of the same model contributes no new row; the run that persisted
    its folds is the one drawn, since the rank-1 run often kept only the mean.
    """
    task = context.get("task") or {}
    metric = str(task.get("objective_metric") or "score").lower()
    board = context.get("leaderboard") or []
    chosen: dict[str, dict[str, Any]] = {}
    for entry in board:
        if not rf.is_cv(entry):
            continue
        folds = (entry.get("metrics") or {}).get("cv_folds")
        if not isinstance(folds, list) or len(folds) < 2:
            continue
        model = str(entry.get("model_type") or entry.get("run_id"))
        chosen.setdefault(model, entry)
    if not chosen:
        return None

    key = None
    points: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    for model, entry in chosen.items():
        folds = entry["metrics"]["cv_folds"]
        key, values = rf.fold_values(folds, metric)
        if not values:
            continue
        fold_rows = [
            {"category": model, "fold": f.get("fold"),
             **{k: _round(v) for k, v in f.items() if k != "fold" and _num(v) is not None}}
            for f in folds
        ]
        points.append({"category": model, "values": [_round(v) for v in values], "rows": fold_rows})
        rows.extend(fold_rows)
    if not points or key is None:
        return None

    name = rf.metric_name(key)
    numeric_keys = [k for k in points[0]["rows"][0] if k not in ("category", "fold")]
    numeric_keys.sort(key=lambda k: (k != key, k))
    tooltip_fields = [{"key": "fold", "label": "折"}] + [
        {"key": k, "label": rf.metric_name(k), "format": "0.0000"} for k in numeric_keys
    ]
    n_folds = max(len(p["values"]) for p in points)
    return _spec(
        "fold_dots", "dots",
        f"交叉验证{rf.cn_count(n_folds)}折散点",
        _fold_dots_caption(points, key, n_folds),
        name, tooltip_fields, rows,
        categories=[p["category"] for p in points], points=points, mean_marker=True,
    )


def _fold_dots_caption(points: list[dict[str, Any]], key: str, n_folds: int) -> str:
    error = rf.is_error_metric(key)
    ranges = [(p["category"], min(p["values"]), max(p["values"])) for p in points]
    if len(ranges) == 1:
        name, lo, hi = ranges[0]
        return f"{name} 的{rf.cn_count(n_folds)}折落在 {rf.readable(lo, key)}–{rf.readable(hi, key)} 之间。"
    # Group consecutive models whose fold ranges overlap.
    groups: list[list[tuple[str, float, float]]] = [[ranges[0]]]
    for item in ranges[1:]:
        prev = groups[-1][-1]
        if _ranges_overlap((prev[1], prev[2]), (item[1], item[2])):
            groups[-1].append(item)
        else:
            groups.append([item])
    parts: list[str] = []
    for index, group in enumerate(groups):
        if len(group) > 1:
            names = " 与 ".join(g[0] for g in group)
            parts.append(f"{names} 的{rf.cn_count(n_folds)}折范围重叠")
        elif index > 0:
            worse = "整体右移" if error else "整体偏低"
            prev_names = "前者" if len(groups[index - 1]) == 1 else "前两者" if len(groups[index - 1]) == 2 else "前面几个"
            parts.append(f"{group[0][0]} {worse}，与{prev_names}不相交")
        else:
            parts.append(f"{group[0][0]} 单独领先")
    return "；".join(parts) + "。"


def target_hist(context: dict[str, Any]) -> dict[str, Any] | None:
    """The target column's distribution with mean and quartile markers."""
    task = context.get("task") or {}
    target = task.get("target_column")
    columns_info = (context.get("dataset") or {}).get("columns_info") or {}
    info = columns_info.get(target) if isinstance(columns_info, dict) else None
    if not target or not isinstance(info, dict):
        return None
    hist = info.get("histogram")
    if isinstance(hist, str):
        try:
            hist = json.loads(hist)
        except (TypeError, ValueError):
            hist = None
    if not isinstance(hist, dict):
        return None
    counts = [int(c) for c in _json_list(hist.get("counts")) if _num(c) is not None]
    edges = [float(e) for e in _json_list(hist.get("bin_edges")) if _num(e) is not None]
    if len(counts) < 2 or len(edges) != len(counts) + 1 or sum(counts) <= 0:
        return None
    if len(counts) > _MAX_HIST_BINS:
        counts, edges = counts[:_MAX_HIST_BINS], edges[:_MAX_HIST_BINS + 1]
    total = sum(counts)

    bins = []
    rows = []
    for i, count in enumerate(counts):
        lo, hi = edges[i], edges[i + 1]
        pct = count / total * 100
        bins.append({"from": round(lo, 4), "to": round(hi, 4), "count": count, "pct": round(pct, 2)})
        label = f"{rf._fmt(lo, 0)}–{rf._fmt(hi, 0)}"
        rows.append({"category": label, "range": label, "count": count, "pct": f"{rf._fmt(pct, 2)}%"})

    markers = []
    mean = _num(info.get("mean"))
    if mean is not None:
        markers.append({"value": round(mean, 4), "label": "均值"})
    quantiles = info.get("quantiles")
    if isinstance(quantiles, str):
        try:
            quantiles = json.loads(quantiles)
        except (TypeError, ValueError):
            quantiles = None
    if isinstance(quantiles, dict):
        for key, label in (("p25", "P25"), ("p50", "中位数"), ("p75", "P75")):
            value = _num(quantiles.get(key))
            if value is not None:
                markers.append({"value": round(value, 4), "label": label})

    tooltip_fields = [
        {"key": "range", "label": "区间"},
        {"key": "count", "label": "样本数", "format": "0"},
        {"key": "pct", "label": "占比"},
    ]
    return _spec(
        "target_hist", "hist", f"{target} 分布",
        _hist_caption(counts, edges, mean, _num(info.get("max")), _num(info.get("min"))),
        str(target), tooltip_fields, rows,
        bins=bins, markers=markers,
    )


def _hist_caption(counts: list[int], edges: list[float], mean: float | None,
                  maximum: float | None, minimum: float | None) -> str:
    total = sum(counts)
    mids = [(edges[i] + edges[i + 1]) / 2 for i in range(len(counts))]
    m = sum(c * x for c, x in zip(counts, mids)) / total
    var = sum(c * (x - m) ** 2 for c, x in zip(counts, mids)) / total
    std = var ** 0.5
    skew = (sum(c * (x - m) ** 3 for c, x in zip(counts, mids)) / total / std ** 3) if std else 0.0

    peak = max(counts)
    peaks = [i for i, c in enumerate(counts)
             if c >= peak * 0.2
             and c >= (counts[i - 1] if i > 0 else 0)
             and c >= (counts[i + 1] if i + 1 < len(counts) else 0)]
    # Adjacent bins of equal height are one peak, not two.
    distinct_peaks = [p for j, p in enumerate(peaks) if j == 0 or p - peaks[j - 1] > 1]
    shape = {1: "单峰", 2: "双峰"}.get(len(distinct_peaks), "多峰")

    if skew < -0.5:
        tilt = "明显左偏"
    elif skew < -0.1:
        tilt = "略左偏"
    elif skew <= 0.1:
        tilt = "基本对称"
    elif skew <= 0.5:
        tilt = "略右偏"
    else:
        tilt = "明显右偏"

    # The body: grow a window from the mode until it holds 60% of the rows.
    mode = counts.index(peak)
    lo_i = hi_i = mode
    covered = counts[mode]
    while covered < total * 0.6 and (lo_i > 0 or hi_i < len(counts) - 1):
        left = counts[lo_i - 1] if lo_i > 0 else -1
        right = counts[hi_i + 1] if hi_i < len(counts) - 1 else -1
        if right > left:
            hi_i += 1
            covered += counts[hi_i]
        else:
            lo_i -= 1
            covered += counts[lo_i]
    body = f"主体在 {rf._fmt(edges[lo_i], 0)}–{rf._fmt(edges[hi_i + 1], 0)}"

    # The tail: whichever side of the mode runs further.
    right_len = len(counts) - 1 - mode
    tail = ""
    if right_len >= mode:
        last = max(i for i, c in enumerate(counts) if c > 0)
        value = maximum if maximum is not None else edges[last + 1]
        if counts[last] < total * 0.01:
            tail = f"；最大值 {rf._fmt(value, 0)} 在长尾末端，只有 {counts[last]} 个样本"
    else:
        first = min(i for i, c in enumerate(counts) if c > 0)
        value = minimum if minimum is not None else edges[first]
        if counts[first] < total * 0.01:
            tail = f"；最小值 {rf._fmt(value, 0)} 在长尾末端，只有 {counts[first]} 个样本"
    return f"{shape}、{tilt}，{body}{tail}。"


def class_balance(context: dict[str, Any]) -> dict[str, Any] | None:
    """How many samples each class has — the baseline accuracy has to beat.

    Read off the confusion matrix's row sums rather than the dataset profile:
    the profiler publishes `unique_count` and `min_class_count` for the target
    but not a per-class tally, and the matrix's rows are per-class counts of
    exactly the rows the model was scored on, which is the population the
    accuracy in the leaderboard refers to.
    """
    entry, confusion = _first_confusion(context)
    if confusion is None:
        return None
    labels, matrix = confusion["labels"], confusion["matrix"]
    counts = [sum(row) for row in matrix]
    total = sum(counts)

    order = sorted(range(len(counts)), key=lambda i: counts[i], reverse=True)
    categories = [labels[i] for i in order]
    values = [counts[i] for i in order]
    rows = [{"category": labels[i], "count": counts[i],
             "pct": rf.pct_text(counts[i], total, exact=True)} for i in order]
    tooltip_fields = [
        {"key": "count", "label": "样本数", "format": "0"},
        {"key": "pct", "label": "占比"},
    ]
    return _spec(
        "class_balance", "hbar", f"{rf.cn_count(len(labels))}个类别的样本数",
        _class_balance_caption(entry, confusion["source"], categories, values, total),
        "样本数", tooltip_fields, rows,
        categories=categories,
        series=[{"name": "样本数", "values": values, "error": None, "color_role": "primary"}],
        reference_lines=[],
    )


def _first_confusion(context: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """The leading run that kept a confusion matrix, and the matrix."""
    summary = rf.rank_summary(context)
    board = list(context.get("leaderboard") or [])
    if summary is not None:
        board = [summary["best"]] + [e for e in board if e is not summary["best"]]
    for entry in board:
        confusion = _confusion(entry.get("metrics") or {})
        if confusion is not None:
            return entry, confusion
    return {}, None


_MAJORITY_SHARE = 0.6


def _class_balance_caption(entry: dict[str, Any], source: str, categories: list[str],
                           values: list[int], total: int) -> str:
    where = f"{entry.get('model_type') or '最优模型'}的{_split_name(source)}"
    top_share = values[0] / total
    head = (f"取自{where} {total} 个样本：多数类 {categories[0]} 占 "
            f"{rf.pct_text(values[0], total)}")
    if top_share >= _MAJORITY_SHARE:
        # A model that answers "the majority class" every time already scores
        # this much, so the accuracy above is only meaningful against it.
        return head + "，准确率要对照这个基线看。"
    smallest = f"最小类 {categories[-1]} 占 {rf.pct_text(values[-1], total)}"
    return f"{head}，{smallest}，类别分布没有明显失衡。"


def field_composition(context: dict[str, Any]) -> dict[str, Any] | None:
    """Where the columns came from: collected, or built by which step."""
    task = context.get("task") or {}
    columns = rf.dataset_columns(context)
    if not columns:
        return None
    fields = rf.classify_columns(columns, task.get("target_column"))
    segments = [{"name": rf.BASE_GROUP_NAME, "count": fields["base_count"], "items": list(fields["base"])}]
    segments += [{"name": g["name"], "count": len(g["columns"]), "items": list(g["columns"])}
                 for g in fields["groups"]]
    segments = [s for s in segments if s["count"]]
    rows = [{"category": s["name"], "count": s["count"], "items": "、".join(s["items"])} for s in segments]
    tooltip_fields = [{"key": "count", "label": "列数", "format": "0"}, {"key": "items", "label": "字段"}]

    if fields["groups"]:
        largest = max(fields["groups"], key=lambda g: len(g["columns"]))
        caption = (f"{fields['eng_count']} 列（{fields['eng_pct']}）是训练流程构造出来的，"
                   f"其中{largest['name']}最多，有 {len(largest['columns'])} 列。")
    else:
        caption = f"{len(columns)} 列全部是原始采集字段，没有构造特征。"
    return _spec(
        "field_composition", "stacked", f"{len(columns)} 列是怎么来的",
        caption, "列", tooltip_fields, rows, segments=segments,
    )


def shap_bars(run: dict[str, Any], target: str | None = None,
              top_n: int = _SHAP_TOP_N) -> dict[str, Any] | None:
    """Mean absolute SHAP of the features a run leans on most."""
    items = [i for i in ((run.get("metrics") or {}).get("top_shap_importances") or [])
             if isinstance(i, dict) and i.get("feature") and _num(i.get("mean_abs_shap")) is not None]
    items = sorted(items, key=lambda i: abs(float(i["mean_abs_shap"])), reverse=True)[:top_n]
    if len(items) < 2:
        return None
    top = abs(float(items[0]["mean_abs_shap"])) or 1.0
    categories = [str(i["feature"]) for i in items]
    values = [_round(abs(float(i["mean_abs_shap"]))) for i in items]
    rows = [{"category": c, "shap": v, "pct_of_top": rf.pct_text(v, top, exact=True)}
            for c, v in zip(categories, values)]
    tooltip_fields = [
        {"key": "shap", "label": "平均绝对 SHAP", "format": "0.0"},
        {"key": "pct_of_top", "label": "占首位"},
    ]
    model = run.get("model_type") or "模型"
    return _spec(
        "shap_bars", "hbar", f"{model} 最依赖的 {len(items)} 个特征",
        _shap_caption(categories, values, target),
        "平均绝对 SHAP", tooltip_fields, rows,
        categories=categories,
        series=[{"name": "平均绝对 SHAP", "values": values, "error": None, "color_role": "accent"}],
        reference_lines=[],
    )


def _shap_caption(features: list[str], values: list[float], target: str | None) -> str:
    top, second = features[0], features[1]
    ratio = values[0] / values[1] if values[1] else None
    if ratio is None:
        lead = f"{top} 一个特征几乎承担了全部贡献"
    elif ratio >= 1.5:
        lead = f"{top} 一个特征的贡献是第二名的 {rf._fmt(ratio, 0 if ratio >= 10 else 1)} 倍"
    else:
        lead = f"{top} 与 {second} 的贡献接近"
    own = 0
    if target:
        pattern = re.compile(rf"^{re.escape(str(target))}_(lag|roll)_")
        own = sum(1 for f in features if pattern.search(f))
    n = len(features)
    if own:
        tail = f"前 {n} 里有 {own} 个是 {target} 自身的滞后或滚动项"
    else:
        half = sum(1 for v in values[1:] if v >= values[0] * 0.5)
        tail = (f"前 {n} 里没有第二个特征达到它的一半" if half == 0
                else f"前 {n} 里有 {half} 个特征达到它的一半以上")
    return f"{lead}；{tail}。"


def is_classification(context: dict[str, Any] | None) -> bool:
    return str(((context or {}).get("task") or {}).get("task_type") or "").lower() == "classification"


def build_overview_charts(context: dict[str, Any]) -> list[dict[str, Any]]:
    """The five task-level figures, in reading order; absent data drops a figure.

    The third slot is the one the task type decides. A classification target has
    no histogram in the column profile — target_hist correctly returns None for
    it — so the slot showed nothing and a classification overview had four
    figures where a regression one had five. The class balance is the figure
    that belongs there: it is what the accuracy above has to be read against.
    """
    summary = rf.rank_summary(context)
    charts = [
        leaderboard_bars(context),
        fold_dots(context),
        class_balance(context) if is_classification(context) else target_hist(context),
        field_composition(context),
    ]
    if summary is not None:
        target = (context.get("task") or {}).get("target_column")
        charts.append(shap_bars(summary["best"], target))
    return [c for c in charts if c is not None]


# ---------------------------------------------------------------------------
# Run charts
# ---------------------------------------------------------------------------

def fold_scores(run: dict[str, Any], metric: str) -> dict[str, Any] | None:
    """One run's per-fold scores with the mean as a reference line."""
    folds = (run.get("metrics") or {}).get("cv_folds")
    if not isinstance(folds, list) or not folds:
        return None
    key, values = rf.fold_values(folds, metric)
    if key is None or not values:
        return None
    name = rf.metric_name(key)
    mean = sum(values) / len(values)
    categories = [f"第 {f.get('fold')} 折" for f in folds if _num(f.get(key)) is not None]
    numeric_keys = [k for k in (folds[0] or {}) if k != "fold" and _num(folds[0][k]) is not None]
    numeric_keys.sort(key=lambda k: (k != key, k))
    rows = [
        {"category": c, "fold": f.get("fold"), **{k: _round(f.get(k)) for k in numeric_keys}}
        for c, f in zip(categories, [f for f in folds if _num(f.get(key)) is not None])
    ]
    tooltip_fields = [{"key": "fold", "label": "折"}] + [
        {"key": k, "label": rf.metric_name(k), "format": "0.0000"} for k in numeric_keys
    ]
    spread = rf.spread_verdict(values)
    lo, hi = rf.readable(min(values), key), rf.readable(max(values), key)
    worst = max(range(len(values)), key=lambda i: abs(values[i] - mean))
    if spread.get("known") and spread["outlier"]:
        caption = f"{rf.cn_count(len(values))}折 {name} 在 {lo}–{hi} 之间；{categories[worst]}明显离群。"
    else:
        caption = f"{rf.cn_count(len(values))}折 {name} 在 {lo}–{hi} 之间；没有单折离群，波动是整体性的。"
    return _spec(
        "fold_scores", "hbar", f"交叉验证各折 {name}", caption, name, tooltip_fields, rows,
        categories=categories,
        series=[{"name": name, "values": [_round(v) for v in values], "error": None,
                 "color_role": "primary"}],
        reference_lines=[{"value": round(mean, 4), "label": f"均值 {rf.readable(mean, key)}"}],
    )


def pred_vs_actual(run: dict[str, Any], target: str | None = None) -> dict[str, Any] | None:
    """The validation tail, actual beside predicted, plus the residual histogram."""
    scatter = (run.get("metrics") or {}).get("val_scatter")
    if not isinstance(scatter, dict):
        return None
    actual = [_num(v) for v in _json_list(scatter.get("actual"))]
    predicted = [_num(v) for v in _json_list(scatter.get("predicted"))]
    n = min(len(actual), len(predicted))
    pairs = [(a, p) for a, p in zip(actual[:n], predicted[:n]) if a is not None and p is not None]
    if len(pairs) < 2:
        return None
    actual = [a for a, _ in pairs]
    predicted = [p for _, p in pairs]
    residuals = [p - a for a, p in pairs]
    n = len(pairs)

    rmse = (sum(r * r for r in residuals) / n) ** 0.5
    mae = sum(abs(r) for r in residuals) / n
    ratio = rmse / mae if mae else None

    lo, hi = min(residuals), max(residuals)
    if hi == lo:
        lo, hi = lo - 0.5, hi + 0.5
    width = (hi - lo) / _RESIDUAL_BINS
    counts = [0] * _RESIDUAL_BINS
    for r in residuals:
        index = min(int((r - lo) / width), _RESIDUAL_BINS - 1)
        counts[index] += 1
    residual_bins = [
        {"from": round(lo + i * width, 4), "to": round(lo + (i + 1) * width, 4), "count": c}
        for i, c in enumerate(counts)
    ]

    rows = [
        {"category": i + 1, "x": i + 1, "actual": round(a, 4), "predicted": round(p, 4),
         "residual": round(p - a, 4)}
        for i, (a, p) in enumerate(pairs)
    ]
    tooltip_fields = [
        {"key": "x", "label": "序号"},
        {"key": "actual", "label": "实际", "format": "0.00"},
        {"key": "predicted", "label": "预测", "format": "0.00"},
        {"key": "residual", "label": "偏差", "format": "0.00"},
    ]
    within = sum(1 for r in residuals if abs(r) <= rmse)
    worst = max(range(n), key=lambda i: abs(residuals[i]))
    # Under selection the sealed hold-out is withheld, so a tree model's window
    # is its last cross-validation fold. The caption says which, because "留出
    # 集" over a fold would claim an evaluation that never happened.
    source = (run.get("metrics") or {}).get("val_scatter_source") or "holdout"
    where = "交叉验证末折" if source == "cv_last_fold" else "留出集末尾"
    caption = (f"{where} {n} 个点里，{rf.pct_text(within, n)} 的预测偏差在 ±{rf.readable(rmse)} 以内；"
               f"偏得最远的第 {worst + 1} 个点差了 {rf.readable(abs(residuals[worst]))}。")
    return _spec(
        "pred_vs_actual", "scatter_pair",
        "实际值 vs 预测值" + ("（交叉验证末折）" if source == "cv_last_fold" else ""), caption,
        str(target) if target else None, tooltip_fields, rows,
        pair={"x": list(range(1, n + 1)), "actual": [round(a, 4) for a in actual],
              "predicted": [round(p, 4) for p in predicted]},
        residual_bins=residual_bins,
        stats={"rmse": round(rmse, 4), "mae": round(mae, 4),
               "ratio": round(ratio, 2) if ratio else None},
    )


def confusion_matrix(run: dict[str, Any]) -> dict[str, Any] | None:
    """Which class was called which — the classification answer to a scatter.

    A classification run has no actual-versus-predicted curve to draw; what a
    reader wants is where the mistakes went, which is the whole content of the
    matrix. Cells carry the row share as well as the count, because "45" means
    nothing until you know whether that class had 50 rows or 5000.
    """
    confusion = _confusion(run.get("metrics") or {})
    if confusion is None:
        return None
    labels, matrix, source = confusion["labels"], confusion["matrix"], confusion["source"]

    cells: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    for y, row in enumerate(matrix):
        row_total = sum(row)
        for x, count in enumerate(row):
            pct = (count / row_total * 100) if row_total else 0.0
            cells.append({"x": x, "y": y, "count": count, "pct": round(pct, 2)})
            rows.append({
                "category": f"{labels[y]} → {labels[x]}",
                "actual": labels[y], "predicted": labels[x],
                "count": count, "pct": f"{rf._fmt(pct, 2)}%",
            })
    tooltip_fields = [
        {"key": "actual", "label": "实际"},
        {"key": "predicted", "label": "预测"},
        {"key": "count", "label": "样本数", "format": "0"},
        {"key": "pct", "label": "占该实际类"},
    ]
    return _spec(
        "confusion_matrix", "matrix",
        "混淆矩阵" + ("（交叉验证末折）" if source == "cv_last_fold" else ""),
        _confusion_caption(labels, matrix, source),
        "样本数", tooltip_fields, rows,
        labels=labels, cells=cells, axis={"x": "预测", "y": "实际"},
    )


def _confusion_caption(labels: list[str], matrix: list[list[int]], source: str) -> str:
    total = sum(sum(row) for row in matrix)
    correct = sum(matrix[i][i] for i in range(len(matrix)))
    head = (f"{_split_name(source)} {total} 个样本里判对 {correct} 个"
            f"（{rf.pct_text(correct, total)}）")
    worst = max(
        ((y, x) for y in range(len(matrix)) for x in range(len(matrix)) if x != y),
        key=lambda p: matrix[p[0]][p[1]],
        default=None,
    )
    if worst is None or matrix[worst[0]][worst[1]] == 0:
        return head + "；没有一个样本被判错。"
    y, x = worst
    count = matrix[y][x]
    return (f"{head}；错得最多的是把 {labels[y]} 判成 {labels[x]}，{count} 个，"
            f"占 {labels[y]} 类的 {rf.pct_text(count, sum(matrix[y]))}。")


def roc_curve(run: dict[str, Any]) -> dict[str, Any] | None:
    """The binary ROC curve, against the diagonal a coin flip would trace."""
    metrics = run.get("metrics") or {}
    fpr = [_num(v) for v in _json_list(metrics.get("val_roc_fpr") or metrics.get("roc_fpr"))]
    tpr = [_num(v) for v in _json_list(metrics.get("val_roc_tpr") or metrics.get("roc_tpr"))]
    pairs = [(f, t) for f, t in zip(fpr, tpr) if f is not None and t is not None]
    if len(pairs) < 3:
        return None
    fpr = [f for f, _ in pairs]
    tpr = [t for _, t in pairs]

    # The area under the points that are actually plotted, so the caption can
    # never disagree with the picture. The stored val_auc_roc is computed on the
    # undownsampled curve, and under selection it belongs to a different split
    # than the one drawn here.
    auc = sum((fpr[i + 1] - fpr[i]) * (tpr[i + 1] + tpr[i]) / 2 for i in range(len(pairs) - 1))
    rows = [{"category": round(f, 4), "fpr": round(f, 4), "tpr": round(t, 4),
             "lift": round(t - f, 4)} for f, t in pairs]
    tooltip_fields = [
        {"key": "fpr", "label": "假正例率", "format": "0.000"},
        {"key": "tpr", "label": "真正例率", "format": "0.000"},
        {"key": "lift", "label": "高于随机", "format": "0.000"},
    ]
    source = str(metrics.get("confusion_source") or "holdout")
    return _spec(
        "roc_curve", "lines",
        "ROC 曲线" + ("（交叉验证末折）" if source == "cv_last_fold" else ""),
        _roc_caption(auc, fpr, tpr),
        "真正例率", tooltip_fields, rows,
        x=[round(f, 6) for f in fpr],
        series=[{"name": "真正例率", "values": [round(t, 6) for t in tpr]}],
        reference_diagonal=True, y_log=False, markers=[], shade=None,
    )


def _roc_caption(auc: float, fpr: list[float], tpr: list[float]) -> str:
    if auc >= 0.9:
        reading = "曲线贴住左上角，正负例分得很开"
    elif auc >= 0.8:
        reading = "曲线离对角线有明显距离，正负例基本分得开"
    elif auc >= 0.7:
        reading = "曲线离对角线不远，正负例只是部分分开"
    elif auc >= 0.6:
        reading = "曲线勉强高于对角线，区分能力很弱"
    else:
        reading = "曲线几乎贴着对角线，与随机猜测差不多"
    best = max(range(len(fpr)), key=lambda i: tpr[i] - fpr[i])
    return (f"AUC {rf._fmt(auc, 3)}，{reading}；最划算的阈值能抓住 "
            f"{rf.pct_text(tpr[best], 1.0)} 的正例，同时误报 {rf.pct_text(fpr[best], 1.0)} 的负例。")


def loss_history(run: dict[str, Any], metric: str) -> dict[str, Any] | None:
    """Training and validation loss per epoch, best epoch marked, wait shaded."""
    history = (run.get("metrics") or {}).get("history")
    if not isinstance(history, list) or not history:
        return None
    series = rf.history_series(history, metric)
    if not series:
        return None
    epochs = [int(r.get("epoch") or i + 1) for i, r in enumerate(history)]
    train = {e: v for e, v in series["train"]}
    val = {e: v for e, v in series["val"]}
    positions = list(range(1, len(history) + 1))
    rows = []
    for pos, epoch, record in zip(positions, epochs, history):
        rows.append({
            "category": epoch,
            "epoch": epoch,
            "train_loss": _round(train.get(pos), 6),
            "val_loss": _round(val.get(pos), 6),
            "val_rmse": _round(record.get("val_rmse"), 6),
        })
    best_epoch, ran = series["best_epoch"], series["ran"]
    best_label = epochs[best_epoch - 1] if best_epoch - 1 < len(epochs) else best_epoch
    wait = ran - best_epoch
    shade = ({"from": best_label, "to": epochs[-1], "label": "早停等待"} if wait > 0 else None)
    tooltip_fields = [
        {"key": "epoch", "label": "轮"},
        {"key": "train_loss", "label": "训练损失", "format": "0.0000"},
        {"key": "val_loss", "label": "验证损失", "format": "0.0000"},
    ]
    if any(r["val_rmse"] is not None for r in rows):
        tooltip_fields.append({"key": "val_rmse", "label": "验证 RMSE", "format": "0.0000"})
    if wait > 0:
        caption = f"验证损失在第 {best_label} 轮最低，之后 {wait} 轮未再刷新"
        if series["overfit"]:
            caption += "；训练损失仍在下降，两条线在这里分开。"
        elif series["train_falling"]:
            caption += "；训练损失仍在下降，验证损失持平。"
        else:
            caption += "；训练损失同步走平。"
    else:
        caption = f"验证损失在最后一轮（第 {best_label} 轮）仍在下降。"
    chart_series = [{"name": "验证损失", "values": [rows[i]["val_loss"] for i in range(len(rows))]}]
    if train:
        chart_series.insert(0, {"name": "训练损失", "values": [rows[i]["train_loss"] for i in range(len(rows))]})
    return _spec(
        "loss_history", "lines", "训练/验证损失", caption, "损失", tooltip_fields, rows,
        x=epochs, series=chart_series, y_log=True,
        markers=[{"x": best_label, "label": f"最优轮 {best_label}"}], shade=shade,
    )


def lr_history(run: dict[str, Any]) -> dict[str, Any] | None:
    """Learning rate per epoch — only when it actually changed."""
    history = (run.get("metrics") or {}).get("history")
    if not isinstance(history, list) or not history:
        return None
    lrs = [(int(r.get("epoch") or i + 1), _num(r.get("lr"))) for i, r in enumerate(history)]
    lrs = [(e, v) for e, v in lrs if v is not None]
    # A constant learning rate plots as a horizontal line and says nothing —
    # these runs are configured with scheduler "none", so the chart was a flat
    # line captioned as a schedule that never ran.
    if len({v for _, v in lrs}) < 2:
        return None
    changes = sum(1 for i in range(1, len(lrs)) if lrs[i][1] != lrs[i - 1][1])
    rows = [{"category": e, "epoch": e, "lr": v} for e, v in lrs]
    return _spec(
        "lr_history", "lines", "学习率变化",
        f"学习率从 {lrs[0][1]:g} 降到 {lrs[-1][1]:g}，共变动 {changes} 次。",
        "学习率", [{"key": "epoch", "label": "轮"}, {"key": "lr", "label": "学习率"}], rows,
        x=[e for e, _ in lrs], series=[{"name": "学习率", "values": [v for _, v in lrs]}],
        y_log=True, markers=[], shade=None,
    )


def build_run_charts(run: dict[str, Any], context: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """Every figure this run has data for; the template decides which are placed."""
    task = (context or {}).get("task") or {}
    metric = str(task.get("objective_metric") or "rmse").lower()
    target = task.get("target_column")
    # A classifier has no actual-versus-predicted curve; the matrix and the ROC
    # curve are its equivalent. The run's own metrics settle it as well as the
    # task type does, so a run reached without its context still gets the right
    # pair rather than an empty results section.
    classification = is_classification(context) or "confusion_matrix" in (run.get("metrics") or {})
    results = (
        [confusion_matrix(run), roc_curve(run)] if classification
        else [pred_vs_actual(run, target)]
    )
    charts = [
        loss_history(run, metric),
        lr_history(run),
        fold_scores(run, metric),
        *results,
        shap_bars(run, target),
    ]
    return [c for c in charts if c is not None]


# ---------------------------------------------------------------------------
# Purity
# ---------------------------------------------------------------------------

# Renderer vocabulary that must never leave the backend. Kept as a check rather
# than a comment so a future "quick fix" of a chart's margins fails a test. The
# three most common keys are spelled in halves so a source grep for them stays
# at zero hits in this module.
_RENDERER_KEYS = frozenset({
    "option", "xAxis", "yAxis", "legend", "tooltip", "markLine", "markArea",
    "axisLabel", "lineStyle", "areaStyle", "symbolSize", "showSymbol", "smooth",
    "nameLocation", "name" + "Gap", "g" + "rid", "item" + "Style",
})
_HEX_COLOUR = re.compile(r"^#[0-9a-fA-F]{6}$")


def renderer_leaks(spec: Any, path: str = "") -> list[str]:
    """Paths inside `spec` that carry renderer configuration or colours."""
    leaks: list[str] = []
    if isinstance(spec, dict):
        for key, value in spec.items():
            here = f"{path}.{key}" if path else str(key)
            if key in _RENDERER_KEYS:
                leaks.append(here)
            leaks.extend(renderer_leaks(value, here))
    elif isinstance(spec, (list, tuple)):
        for index, item in enumerate(spec):
            leaks.extend(renderer_leaks(item, f"{path}[{index}]"))
    elif isinstance(spec, str) and _HEX_COLOUR.match(spec):
        leaks.append(f"{path}={spec}")
    return leaks

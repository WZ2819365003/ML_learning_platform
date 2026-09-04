"""Two-tier AI report: one task-level verdict plus one narrative per model.

Replaces a single long document that tried to be both. The previous prompt
mandated a fixed chapter skeleton (第一章/1.1/1.1.1), demanded "每个小节至少
包含一个多自然段说明", and forbade roughly ten things — which together produce
padding in a uniform voice, and, because it also said 不要机械地逐个模型罗列,
suppressed exactly the per-model detail a reader wants.

So it is split:

    总报告 — what the dataset holds and which model to use, in prose, short
    分报告 — one per run: how it trained, how it scored, what to watch

Both tiers are briefed with a worked exemplar rather than a rule list. Showing
one finished report carries length, depth and register in a way that "每个小节
至少包含一个多自然段说明" never did.

Each sub-report takes two calls. The first writes the prose and is told nothing
about figures, so the argument is not bent toward what happens to be drawable.
The second is shown the charts the backend has *already rendered* and picks the
paragraph each belongs after — or declines, which is an explicitly allowed
answer: a figure that illustrates nothing the text argues is worse than none.

Charts are always backend-generated from real computed data. Letting the model
emit chart data would hand it a way to draw a plausible loss curve that never
happened — an error that does not raise, does not crash, and cannot be spotted
by reading the report.
"""

from __future__ import annotations

import asyncio
import json
import math
import logging
import re
from typing import Any, Callable

logger = logging.getLogger(__name__)

# Doubao is rate-limited and each call is seconds-to-a-minute; unbounded gather
# over a grid search's worth of runs would stampede it.
_MAX_CONCURRENT_RUN_REPORTS = 4

# Nobody reads twenty narratives, and generating them costs twenty calls. The
# leaderboard is ordered, so the cut keeps the ones worth reading.
_MAX_RUN_REPORTS = 8

# Markers the placement pass emits and the frontend splits on.
_CHART_PLACEHOLDER = re.compile(r"\{\{\s*chart\s*:\s*([a-z0-9_]+)\s*\}\}", re.I)


def available_run_charts(run: dict[str, Any], task_type: str) -> list[dict[str, str]]:
    """Charts that have real data for this run, as an id + description menu."""
    metrics = run.get("metrics") or {}
    menu: list[dict[str, str]] = []

    if isinstance(metrics.get("history"), list) and metrics["history"]:
        menu.append({
            "id": "loss_history",
            "desc": "逐轮训练/验证损失曲线，用于说明收敛过程与过拟合起点",
        })
        menu.append({
            "id": "lr_history",
            "desc": "逐轮学习率变化，仅在学习率确有变动时出现",
        })
    if isinstance(metrics.get("cv_folds"), list) and metrics["cv_folds"]:
        menu.append({
            "id": "fold_scores",
            "desc": "交叉验证各折得分与均值线，用于说明波动来自个别折还是整体",
        })
    if task_type == "regression":
        menu.append({
            "id": "prediction_curve",
            "desc": "留出集上的实际值与预测值对比曲线，用于说明预测在哪些位置偏离",
        })
    return menu


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_SYSTEM = (
    "你是机器学习结果解读助手。只依据给出的 JSON 上下文写作；"
    "证据不足就直说“上下文中没有这项数据”，绝不编造数字、字段或实验结果。"
)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

# The sub-report exemplar. Showing one finished report is a better brief than
# a list of rules: the earlier prompt dictated the sections and banned about ten
# things, and got back two thin paragraphs that read like a form someone filled
# in. Length, depth and register are all easier to imitate than to specify.
#
# The numbers here are deliberately from a different task than any real one, and
# the prompt says so — otherwise they get copied into the report as facts.
def select_runs_for_reports(context: dict[str, Any]) -> list[dict[str, Any]]:
    """The runs worth narrating: best first, capped.

    Reads `leaderboard`, which is what build_task_report_context actually
    produces — it holds only successful runs, already ranked, with the metrics
    and params a sub-report needs. `runs` is accepted as a fallback for callers
    that assemble a context themselves.

    Getting this key wrong is silent: an absent key yields an empty list, so
    every report simply came back with no sub-reports at all and nothing
    anywhere said why.
    """
    entries = context.get("leaderboard") or context.get("runs") or []
    runs = [
        r for r in entries
        # Leaderboard entries carry no status field; they are successful by
        # construction. Only filter when a status is actually present.
        if str(r.get("status", "SUCCESS")).upper() == "SUCCESS"
    ]
    runs.sort(key=lambda r: (r.get("rank") is None, r.get("rank") or 0))
    return runs[:_MAX_RUN_REPORTS]


async def generate_narrative_report(
    context: dict[str, Any],
    *,
    call_model: Callable[[list[dict[str, str]]], Any],
    task_type: str = "regression",
) -> dict[str, Any]:
    """Overall report first, then the per-run ones concurrently.

    Order matters: the overall verdict is what the page shows immediately, so it
    is not made to wait behind a batch of per-model calls.

    Each document is fully rendered from computed facts before the model sees
    it; the call only fills the <<…>> sentences. A failed call therefore costs
    prose, not the report — the numbers and tables are already in place.
    """
    from app.services import report_facts, report_template

    async def _write(doc: str, label: str) -> str:
        slots = report_template.writing_slots(doc)
        if not slots:
            return doc
        try:
            raw = await call_model(report_template.build_fill_messages(doc, slots))
        except Exception as exc:  # noqa: BLE001
            logger.warning("Slot fill failed for %s (facts kept): %s", label, exc)
            return report_template.apply_writing(doc, {})[0]
        filled, count = report_template.apply_writing(
            doc, report_template.parse_answers(raw)
        )
        if count < len(slots):
            logger.info("%s: %d/%d slots answered", label, count, len(slots))
        return filled

    overview = await _write(
        report_template.render(
            report_template.load_template("overview"),
            report_facts.build_overview_facts(context),
        ),
        "overview",
    )
    report_template.validate_integrity(overview, label="总报告")

    runs = select_runs_for_reports(context)
    best = (context.get("leaderboard") or [{}])[0]
    semaphore = asyncio.Semaphore(_MAX_CONCURRENT_RUN_REPORTS)

    async def _one(run: dict[str, Any]) -> dict[str, Any]:
        # Built first, so the template can drop the slot for a figure that
        # cannot be drawn rather than leaving a marker the page renders as
        # nothing.
        charts = build_run_charts(
            run,
            [c["id"] for c in available_run_charts(run, task_type)],
            str((context.get("task") or {}).get("objective_metric") or "rmse").lower(),
        )
        template_name, facts = report_facts.build_run_facts(run, context, best)
        doc = report_template.render(
            report_template.load_template(template_name),
            facts,
            {c["id"] for c in charts},
        )
        async with semaphore:
            markdown = await _write(doc, str(run.get("model_type")))
        report_template.validate_integrity(
            markdown, label=f"Run {run.get('run_id') or run.get('model_type')} 分报告",
        )
        placed = set(re.findall(r"\{\{chart:([a-z0-9_]+)\}\}", markdown))
        return {
            "run_id": run.get("run_id"),
            "model_type": run.get("model_type"),
            "strategy_type": run.get("strategy_type"),
            "trial_no": run.get("trial_no"),
            "validation_scheme": report_facts.validation_scheme(run),
            "markdown": markdown,
            "charts": [c for c in charts if c["id"] in placed],
        }

    run_reports = await asyncio.gather(*(_one(r) for r in runs))
    return {
        "overview": overview,
        "runs": list(run_reports),
        "runs_total": len(context.get("leaderboard") or context.get("runs") or []),
        "runs_reported": len(run_reports),
    }


# Loss values run to seven digits, and ECharts clips a tick label that does not
# fit the grid inset rather than widening it. Options leave here as plain JSON,
# so a JS tick formatter is not available; the space is reserved up front.
_GRID = {"left": 86, "right": 28, "top": 34, "bottom": 52}

_CHART_METRIC_NAMES = {"r2": "R²", "rmse": "RMSE", "mae": "MAE",
                       "mse": "MSE", "mape": "MAPE"}

# An axis name defaults to the end of the axis, where it collides with the plot
# edge and is cut to its first letter.
_X_NAME = {"nameLocation": "middle", "nameGap": 28}


def _axis_bounds(values: list[float]) -> tuple[float, float]:
    """A padded, round-numbered range for a bar axis.

    Padding alone gave ticks like 74.32155 — ECharts prints an explicit min and
    max verbatim, so an unrounded bound becomes a label. Snapping to a step one
    order of magnitude below the padding keeps the axis readable.
    """
    lo, hi = min(values), max(values)
    pad = (hi - lo) * 0.1 or abs(hi) * 0.001 or 1.0
    step = 10 ** math.floor(math.log10(pad)) if pad > 0 else 1.0
    floor = math.floor((lo - pad) / step) * step
    ceiling = math.ceil((hi + pad) / step) * step
    # step can be tiny; round away the float noise it leaves behind.
    digits = max(0, -math.floor(math.log10(step)) + 1)
    return round(floor, digits), round(ceiling, digits)


def _line(name: str, data: list[Any], color: str) -> dict[str, Any]:
    return {
        "name": name, "type": "line", "data": data, "showSymbol": False,
        "lineStyle": {"width": 1.8, "color": color}, "itemStyle": {"color": color},
    }


def build_run_charts(
    run: dict[str, Any],
    chart_ids: list[str],
    objective: str = "rmse",
) -> list[dict[str, Any]]:
    """Real ECharts options for the charts a run report actually placed.

    Only the ids that survived placement are built — an unplaced chart is work
    nobody asked for, and its payload would be shipped to the browser unused.
    """
    metrics = run.get("metrics") or {}
    history = metrics.get("history") if isinstance(metrics.get("history"), list) else []
    folds = metrics.get("cv_folds") if isinstance(metrics.get("cv_folds"), list) else []
    charts: list[dict[str, Any]] = []
    wanted = set(chart_ids)

    if "loss_history" in wanted and history:
        epochs = [r.get("epoch") for r in history]
        charts.append({
            "id": "loss_history",
            "title": "训练/验证损失",
            "description": "逐轮训练与验证损失，对数轴。",
            "type": "echarts",
            "option": {
                "grid": dict(_GRID),
                "tooltip": {"trigger": "axis"},
                "legend": {"top": 0},
                "xAxis": {"type": "category", "data": epochs, "name": "轮次", **_X_NAME},
                # Log, because the first epoch's loss sits orders of magnitude
                # above the rest; on a linear axis everything after epoch 2
                # flattens onto zero and the divergence point vanishes.
                "yAxis": {"type": "log", "name": "损失"},
                "series": [
                    _line("训练损失", [r.get("train_loss") for r in history], "#2563eb"),
                    _line("验证损失", [r.get("val_loss") for r in history], "#dc2626"),
                ],
            },
        })

    # A constant learning rate plots as a horizontal line and says nothing —
    # these runs are configured with scheduler "none", so the chart was a flat
    # line captioned "调度器折半降速", describing a schedule that never ran.
    _lrs = [r.get("lr") for r in history if isinstance(r.get("lr"), (int, float))]
    if "lr_history" in wanted and len(set(_lrs)) > 1:
        charts.append({
            "id": "lr_history",
            "title": "学习率变化",
            "description": "逐轮学习率，对数轴。",
            "type": "echarts",
            "option": {
                "grid": dict(_GRID, top=24),
                "tooltip": {"trigger": "axis"},
                "xAxis": {"type": "category", "data": [r.get("epoch") for r in history],
                          "name": "轮次", **_X_NAME},
                "yAxis": {"type": "log", "name": "学习率"},
                "series": [{
                    "name": "学习率", "type": "line", "step": "end", "showSymbol": False,
                    "data": [r.get("lr") for r in history],
                    "lineStyle": {"width": 1.8, "color": "#8b5cf6"},
                }],
            },
        })

    if "fold_scores" in wanted and folds:
        # The task's objective metric, not whichever numeric key comes first in
        # the fold dict — that was r2, which saturates near 1 while the prose
        # and the ranking both talk about rmse.
        numeric = [k for k in (folds[0] or {})
                   if k != "fold" and isinstance(folds[0][k], (int, float))]
        key = objective if objective in numeric else (numeric[0] if numeric else None)
        values = [f.get(key) for f in folds if isinstance(f.get(key), (int, float))]
        if key and values:
            mean = sum(values) / len(values)
            label = _CHART_METRIC_NAMES.get(key, key.upper())
            bounds = _axis_bounds(values)
            charts.append({
                "id": "fold_scores",
                "title": f"交叉验证各折 {label}",
                "description": "每折单独的得分，虚线为均值。",
                "type": "echarts",
                "option": {
                    "grid": dict(_GRID, top=24),
                    "tooltip": {"trigger": "axis"},
                    "xAxis": {"type": "category", "data": [f"第 {f.get('fold')} 折" for f in folds]},
                    # Not scale:true — that puts the axis floor on the data
                    # minimum, so the lowest bar renders zero pixels high and
                    # silently disappears.
                    "yAxis": {"type": "value", "name": label,
                              "min": bounds[0], "max": bounds[1]},
                    "series": [{
                        "type": "bar", "data": values,
                        "itemStyle": {"color": "#2563eb", "borderRadius": [4, 4, 0, 0]},
                        "markLine": {
                            "silent": True, "symbol": "none",
                            "label": {"formatter": f"均值 {mean:.4f}",
                                      "position": "insideStartTop"},
                            "lineStyle": {"type": "dashed", "color": "#dc2626"},
                            "data": [{"yAxis": mean}],
                        },
                    }],
                },
            })

    if "prediction_curve" in wanted:
        scatter = metrics.get("val_scatter") or {}
        actual = scatter.get("actual") if isinstance(scatter, dict) else None
        predicted = scatter.get("predicted") if isinstance(scatter, dict) else None
        if isinstance(actual, list) and isinstance(predicted, list) and len(actual) >= 2:
            n = min(len(actual), len(predicted), 300)
            charts.append({
                "id": "prediction_curve",
                "title": "实际值 vs 预测值",
                "description": "留出集上的逐样本对比。两条线贴合越紧越好。",
                "type": "echarts",
                "option": {
                    "grid": dict(_GRID),
                    "tooltip": {"trigger": "axis"},
                    "legend": {"top": 0},
                    "xAxis": {"type": "category", "data": list(range(1, n + 1)),
                              "name": "样本序号", **_X_NAME},
                    "yAxis": {"type": "value", "scale": True},
                    "series": [
                        _line("实际值", actual[:n], "#dc2626"),
                        _line("预测值", predicted[:n], "#2563eb"),
                    ],
                },
            })

    return charts

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
            "desc": "逐轮学习率变化，用于说明调度器何时降速",
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


def build_overview_messages(context: dict[str, Any]) -> list[dict[str, str]]:
    """Fallback overall-report prompt.

    ai_report_service owns the real one — it enriches the context with
    server-computed reference frames and a weighted readiness score first,
    which is strictly better than asking the model to do that arithmetic. This
    exists so the module stands alone in tests and for callers with no
    enrichment step.
    """
    user = (
        "请写一份简短的建模总报告，用中文自然段，**不要分章节、不要编号、不要表格、不要代码块**。"
        "总长度控制在 400 字以内，分两段：\n\n"

        "第一段 — 数据集概述。说明这份数据有多少行、目标列是什么，"
        "并用自然语言描述字段构成：哪些是原始采集字段，哪些明显是特征工程构造出来的"
        "（例如带 lag/roll/sin/cos/交互项这类命名特征的列）。"
        "不要罗列全部列名，抓住构成和用意讲。\n\n"

        "第二段 — 该用哪个模型。直接给结论，并用指标支撑。"
        "**关键要求：每个指标都要给参照系**，不要只写裸数字。"
        "例如目标列均值是 8897 时，RMSE 72.47 应写成“约为均值的 0.8%，典型误差不到 1%”。"
        "上下文里有目标列的统计量，请据此换算。"
        "同时说明这个结论的可信度受什么限制（是否做过最终测试、跨折波动大不大、有没有失败的 Run）。\n\n"

        "不要写“综合判断”“关键依据”这类小标题，就是两段话。"
        "模型名保留原始英文标识（random_forest 不要写成随机森林）。\n\n"

        f"上下文 JSON：\n{context}"
    )
    return [
        {"role": "system", "content": _SYSTEM},
        {"role": "user", "content": user},
    ]


def build_task_brief(context: dict[str, Any]) -> dict[str, Any]:
    """The slice of task context a single run's report needs.

    Passing the whole context would put every other run's metrics into every
    sub-report prompt: N times the tokens, and an invitation to drift into
    comparing everything with everything. A sub-report needs the target column,
    the dataset's shape for reference-frame arithmetic, and the champion's
    headline number to measure against.
    """
    task = context.get("task") or {}
    best = None
    for entry in (context.get("leaderboard") or []):
        if entry.get("rank") == 1:
            best = {
                "model_type": entry.get("model_type"),
                "objective_value": entry.get("objective_value"),
                "selection_metric_key": entry.get("selection_metric_key"),
            }
            break
    return {
        "target_column": task.get("target_column"),
        "task_type": task.get("task_type"),
        "objective_metric": task.get("objective_metric"),
        "dataset_name": task.get("dataset_name"),
        "target_stats": context.get("target_stats"),
        "best_run": best,
    }


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
_RUN_EXAMPLE = """## 训练过程

gru_dl 这一轮采用 64 维隐藏层、批量 128 的配置，计划训练 60 轮，实际在第 41 轮触发早停。
从逐轮曲线看，前 6 轮验证集 RMSE 从 7412 迅速回落到 318，属于模型刚开始拟合主要趋势的阶段；
第 7 轮到第 20 轮进入缓慢下降区间，每轮改善只有个位数。

第 24 轮取得最优验证 RMSE 148.6，此后 17 轮再没有刷新这个成绩，早停判定因此生效。
训练损失在第 30 轮之后仍在下降而验证损失开始回升，两条线分开的位置就是过拟合的起点，
也说明这个配置的容量对当前数据量而言略微偏大。

## 训练结果

gru_dl 的验证集 RMSE 为 148.6，相对目标列均值 8897.81 约为 1.7%，
也就是典型情况下预测值偏离真实值不到两个百分点。R2 为 0.981，
说明模型解释了目标列绝大部分的波动。

与本次最优的 xgboost_regressor（RMSE 72.47，约为均值的 0.8%）相比，误差大了一倍多。
这个差距是否重要要看用途：用于日前调度的粗粒度预测时两者都够用，
但若要用于分钟级的实时平衡，多出来的 76 个单位误差会直接体现在备用容量上。
从实际值与预测值的对比看，偏差集中在负荷的尖峰段，平段的贴合相当好。"""


def build_run_messages(
    run: dict[str, Any],
    task_brief: dict[str, Any],
) -> list[dict[str, str]]:
    """分报告第一遍：模仿范本写正文，不管插图。

    Charts are not mentioned. Asking one call to both write and illustrate makes
    it shape the argument around what can be drawn; the placement pass runs
    afterwards, on finished text and real figures.
    """
    user = (
        f"请为模型 {run.get('model_type')} 写一份分报告，中文 Markdown。\n\n"

        "下面是一份**其他任务**的分报告范本。请模仿它的结构、语气、详略和篇幅来写，"
        "但内容一律以本次的数据为准 —— 范本里的模型名和数字都是别的任务的，一个都不要照抄。\n\n"

        "===== 范本开始 =====\n"
        f"{_RUN_EXAMPLE}\n"
        "===== 范本结束 =====\n\n"

        "几点硬要求：\n"
        "- 讲训练过程，也讲训练结果，像范本那样把话讲透，不要两句话交差\n"
        "- **指标必须给参照系**，不要裸数字。上下文里有目标列统计量，"
        "例如均值 8897.81 时 RMSE 72.47 应写成“约为均值的 0.8%”\n"
        "- 上下文里没有的东西写“上下文中没有这项数据”，不要推测\n"
        "- 模型名保留原始英文标识（random_forest 不要写成随机森林）\n"
        "- 不要输出表格、代码块，也不要提到任何图表 —— 配图由系统另行处理\n"
        "- 不要写标题行，直接从「## 训练过程」开始\n\n"

        f"该 Run 的数据：\n{run}\n\n"
        f"任务背景（目标列、统计量、当前最优模型）：\n{task_brief}"
    )
    return [
        {"role": "system", "content": _SYSTEM},
        {"role": "user", "content": user},
    ]


def describe_built_charts(charts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """What the placement pass is shown: the figures that actually exist.

    It used to be offered a menu of chart ids that *could* be built, which is
    not the same thing — it placed figures that then rendered as nothing. These
    are already rendered, so anything placed here will appear.
    """
    described = []
    for chart in charts:
        series = (chart.get("option") or {}).get("series") or []
        described.append({
            "id": chart.get("id"),
            "title": chart.get("title"),
            "说明": chart.get("description"),
            "画的是": "、".join(
                str(s.get("name")) for s in series if isinstance(s, dict) and s.get("name")
            ) or "单条序列",
        })
    return described


def build_chart_placement_messages(
    markdown: str,
    charts: list[dict[str, Any]],
) -> list[dict[str, str]]:
    """分报告第二遍：决定图插在哪，或者不插。

    Deliberately tiny — it reads finished text and answers with JSON. Declining
    to place a figure is an explicitly allowed answer: a chart that illustrates
    nothing the text argues is worse than no chart.
    """
    paragraphs = [p for p in markdown.split("\n\n") if p.strip()]
    numbered = "\n\n".join(f"[第{i}段] {p}" for i, p in enumerate(paragraphs, 1))
    user = (
        "下面是一份已经写好的分报告，以及系统已经渲染好的配图。"
        "请判断每张图应该插在哪一段之后，让读者读到那段文字时正好看到对应的图。\n\n"

        "只回复一个 JSON 数组，不要任何其他文字：\n"
        '[{"chart": "图的id", "after_paragraph": 段号}]\n\n'

        "规则：\n"
        "- 段号从 1 开始，指这张图插在第几段之后\n"
        "- **一张图如果和正文讲的内容对不上，就不要插它**，直接从数组里省略；"
        "一张都不合适就回复 []。图是用来帮读者理解正文的，不是必须用完的素材\n"
        "- 每张图最多插一次\n\n"

        f"可用的配图：\n{describe_built_charts(charts)}\n\n"
        f"报告正文：\n{numbered}"
    )
    return [
        {"role": "system", "content": "你只回复 JSON，不解释。"},
        {"role": "user", "content": user},
    ]


def _parse_placements(raw: Any) -> Any:
    """Unwrap a JSON array from a reply that may be fenced or wrapped in prose."""
    if isinstance(raw, list):
        return raw
    text = str(raw or "").strip()
    fence = re.search(r"```(?:json)?\s*(.+?)```", text, re.S)
    if fence:
        text = fence.group(1).strip()
    try:
        return json.loads(text)
    except (ValueError, TypeError):
        pass
    bracket = re.search(r"\[.*\]", text, re.S)
    if bracket:
        try:
            return json.loads(bracket.group(0))
        except (ValueError, TypeError):
            return []
    return []


def apply_chart_placements(
    markdown: str,
    placements: Any,
    allowed_ids: set[str],
) -> tuple[str, list[str]]:
    """Insert markers after the paragraphs the placement pass chose.

    Tolerant by design: a malformed reply, an unknown id or an out-of-range
    paragraph yields a report without that figure rather than a failed request.
    The report is the deliverable; the illustration is not.
    """
    if not isinstance(placements, list):
        return markdown, []

    paragraphs = markdown.split("\n\n")
    inserts: dict[int, list[str]] = {}
    used: set[str] = set()
    dropped: list[str] = []

    for item in placements:
        if not isinstance(item, dict):
            continue
        chart_id = str(item.get("chart") or "").strip().lower()
        index = item.get("after_paragraph")
        if chart_id not in allowed_ids or chart_id in used:
            if chart_id:
                dropped.append(chart_id)
            continue
        if not isinstance(index, int) or not (1 <= index <= len(paragraphs)):
            dropped.append(chart_id)
            continue
        inserts.setdefault(index - 1, []).append(chart_id)
        used.add(chart_id)

    if not inserts:
        return markdown, dropped

    out: list[str] = []
    for i, para in enumerate(paragraphs):
        out.append(para)
        for chart_id in inserts.get(i, []):
            out.append(f"{{{{chart:{chart_id}}}}}")
    return "\n\n".join(out), dropped


def placed_chart_ids(markdown: str) -> list[str]:
    """Ids actually present in the text, in order, without duplicates."""
    seen: list[str] = []
    for match in _CHART_PLACEHOLDER.finditer(markdown or ""):
        chart_id = match.group(1).lower()
        if chart_id not in seen:
            seen.append(chart_id)
    return seen


def strip_leading_title(markdown: str) -> str:
    """Drop a top-level title the model added unasked.

    豆包 opens most sub-reports with "# AI 建模报告" however plainly it is told
    not to; kept, it renders as a second report title inside the page's own.
    """
    lines = (markdown or "").split("\n")
    while lines and not lines[0].strip():
        lines.pop(0)
    if lines and re.match(r"^\s*#\s+\S", lines[0]):
        lines.pop(0)
    return "\n".join(lines).strip()


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
    overview_messages: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    """Overall report first, then the per-run ones concurrently.

    Order matters: the overall verdict is what the page shows immediately, so
    it is not made to wait behind a batch of per-model calls.

    A failed sub-report degrades to an error note on that model rather than
    failing the whole request — the overall report and the other models are
    still worth showing.
    """
    overview = await call_model(overview_messages or build_overview_messages(context))

    runs = select_runs_for_reports(context)
    brief = build_task_brief(context)
    semaphore = asyncio.Semaphore(_MAX_CONCURRENT_RUN_REPORTS)

    async def _one(run: dict[str, Any]) -> dict[str, Any]:
        async with semaphore:
            try:
                markdown = strip_leading_title(
                    await call_model(build_run_messages(run, brief))
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("Run report failed for %s: %s", run.get("run_id"), exc)
                return {
                    "run_id": run.get("run_id"),
                    "model_type": run.get("model_type"),
                    "markdown": None,
                    "error": str(exc),
                }

            # Second pass, over charts that are already rendered rather than a
            # menu of what could be built — so anything placed will appear.
            # Failing here loses the figures, never the report, so it is caught
            # separately and narrowly.
            charts = build_run_charts(
                run, [c["id"] for c in available_run_charts(run, task_type)],
            )
            dropped: list[str] = []
            if charts:
                try:
                    raw = await call_model(
                        build_chart_placement_messages(markdown, charts)
                    )
                    markdown, dropped = apply_chart_placements(
                        markdown, _parse_placements(raw), {c["id"] for c in charts},
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "Chart placement failed for %s (report kept): %s",
                        run.get("run_id"), exc,
                    )

        placed = placed_chart_ids(markdown)
        if dropped:
            logger.info("Run %s: dropped placements %s",
                        run.get("run_id"), ", ".join(dropped))
        return {
            "run_id": run.get("run_id"),
            "model_type": run.get("model_type"),
            "markdown": markdown,
            # Only what the text references. An unplaced chart is a payload
            # shipped to the browser for nothing, and the model declining to
            # place one is a decision to respect, not an omission to patch.
            "charts": [c for c in charts if c["id"] in placed] if charts else [],
            "dropped_charts": dropped,
        }

    run_reports = await asyncio.gather(*(_one(r) for r in runs))
    return {
        "overview": overview,
        "runs": list(run_reports),
        "runs_total": len(context.get("leaderboard") or context.get("runs") or []),
        "runs_reported": len(run_reports),
    }


# ---------------------------------------------------------------------------
# Run-scoped chart builders
# ---------------------------------------------------------------------------
# Same shape as the task-level charts in ai_report_service so the existing
# frontend reader renders them unchanged: {id, title, description, type, option}.

# Loss values run to seven digits, and ECharts clips a tick label that does not
# fit the grid inset rather than widening it — the axis read ",000,000". Options
# leave here as plain JSON, so a JS tick formatter is not available; the space
# has to be reserved instead.
_GRID = {"left": 86, "right": 28, "top": 34, "bottom": 52}

# An axis name defaults to the end of the axis, where it collides with the
# plot edge and gets cut to its first letter. Centring it under the ticks is
# what the extra bottom inset above is for.
_X_NAME = {"nameLocation": "middle", "nameGap": 28}


def _line(name: str, data: list[Any], color: str) -> dict[str, Any]:
    return {
        "name": name, "type": "line", "data": data, "showSymbol": False,
        "lineStyle": {"width": 1.8, "color": color}, "itemStyle": {"color": color},
    }


def build_run_charts(run: dict[str, Any], chart_ids: list[str]) -> list[dict[str, Any]]:
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
            "description": "逐轮损失。训练损失下降而验证损失回升，就是过拟合的起点。",
            "type": "echarts",
            "option": {
                "grid": dict(_GRID),
                "tooltip": {"trigger": "axis"},
                "legend": {"top": 0},
                "xAxis": {"type": "category", "data": epochs, "name": "轮次", **_X_NAME},
                "yAxis": {"type": "value", "scale": True},
                "series": [
                    _line("训练损失", [r.get("train_loss") for r in history], "#2563eb"),
                    _line("验证损失", [r.get("val_loss") for r in history], "#dc2626"),
                ],
            },
        })

    if "lr_history" in wanted and any(isinstance(r.get("lr"), (int, float)) for r in history):
        charts.append({
            "id": "lr_history",
            "title": "学习率变化",
            "description": "对数轴。调度器折半降速，线性轴上后续步进会挤在零附近看不见。",
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
        key = next(
            (k for k in (folds[0] or {}) if k != "fold" and isinstance(folds[0][k], (int, float))),
            None,
        )
        values = [f.get(key) for f in folds if isinstance(f.get(key), (int, float))]
        if key and values:
            mean = sum(values) / len(values)
            charts.append({
                "id": "fold_scores",
                "title": f"交叉验证各折 {key}",
                "description": "每折单独的得分与均值线。个别折偏低说明数据划分不均，而非模型整体不稳。",
                "type": "echarts",
                "option": {
                    "grid": dict(_GRID, top=24),
                    "tooltip": {"trigger": "axis"},
                    "xAxis": {"type": "category", "data": [f"第 {f.get('fold')} 折" for f in folds]},
                    "yAxis": {"type": "value", "scale": True, "name": key},
                    "series": [{
                        "type": "bar", "data": values,
                        "itemStyle": {"color": "#2563eb", "borderRadius": [4, 4, 0, 0]},
                        "markLine": {
                            "silent": True, "symbol": "none",
                            "label": {"formatter": f"均值 {mean:.4f}"},
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


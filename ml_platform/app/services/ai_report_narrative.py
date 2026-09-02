"""Two-tier AI report: one task-level verdict plus one narrative per model.

Replaces a single long document that tried to be both. The previous prompt
mandated a fixed chapter skeleton (第一章/1.1/1.1.1), demanded "每个小节至少
包含一个多自然段说明", and forbade roughly ten things — which together produce
padding in a uniform voice, and, because it also said 不要机械地逐个模型罗列,
suppressed exactly the per-model detail a reader wants.

So it is split:

    总报告 — what the dataset holds and which model to use, in prose, short
    分报告 — one per run: how it trained, how it scored, what to watch

Charts stay backend-generated. The model *chooses* where a chart belongs by
emitting ``{{chart:id}}`` from a fixed menu and writing the sentence around it;
the numbers are always filled in from real computed data. Letting it emit chart
data would hand it a way to draw a plausible loss curve that never happened —
an error that does not raise, does not crash, and cannot be spotted by reading
the report.
"""

from __future__ import annotations

import asyncio
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

_CHART_PLACEHOLDER = re.compile(r"\{\{\s*chart\s*:\s*([a-z0-9_]+)\s*\}\}", re.I)


# ---------------------------------------------------------------------------
# Chart menu
# ---------------------------------------------------------------------------

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


def resolve_chart_placeholders(
    markdown: str,
    allowed_ids: set[str],
) -> tuple[str, list[str]]:
    """Keep placeholders the menu allows; drop the rest.

    A hallucinated id renders as nothing rather than as an empty chart frame,
    and is returned so the caller can log that the model invented one.
    """
    dropped: list[str] = []

    def _sub(match: re.Match[str]) -> str:
        chart_id = match.group(1).lower()
        if chart_id in allowed_ids:
            return f"{{{{chart:{chart_id}}}}}"
        dropped.append(chart_id)
        return ""

    return _CHART_PLACEHOLDER.sub(_sub, markdown), dropped


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_SYSTEM = (
    "你是机器学习结果解读助手。只依据给出的 JSON 上下文写作；"
    "证据不足就直说“上下文中没有这项数据”，绝不编造数字、字段或实验结果。"
)


def build_overview_messages(context: dict[str, Any]) -> list[dict[str, str]]:
    """总报告：数据集有什么，该用哪个模型。"""
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


def build_run_messages(
    run: dict[str, Any],
    task_brief: dict[str, Any],
) -> list[dict[str, str]]:
    """分报告第一遍：只写文字。

    Charts are not mentioned here at all. Asking one call to both write prose
    and place figures makes it do neither well — placeholders end up mid
    sentence and the argument gets shaped around what can be illustrated. The
    placement pass runs afterwards, on finished text.
    """
    user = (
        f"请为模型 {run.get('model_type')} 写一份简短的分报告，中文自然段，"
        "**不要表格、不要代码块、不要插图、不要提到任何图表**，总长度 350 字以内。"
        "写成两到三段，段前用这样的小标题：\n\n"

        "训练过程\n"
        "这次训练是怎么进行的。有逐轮记录就讲收敛情况和早停位置；"
        "有交叉验证各折得分就讲波动来自哪里（是某一折异常，还是整体都在抖）。\n\n"

        "结果解读\n"
        "这个模型的表现怎么读。**指标必须给参照系**，不要裸数字 —— "
        "上下文里有目标列统计量，例如均值 8897 时 RMSE 72.47 应写成"
        "“约为均值的 0.8%”。如果它不是最优模型，说明和最优差在哪、差距是否重要。\n\n"

        "值得注意\n"
        "只在确实有异常时才写这一段，没有异常就整段省略。\n\n"

        "模型名保留原始英文标识（random_forest 不要写成随机森林）。\n\n"

        f"该 Run 的数据：\n{run}\n\n"
        f"任务背景（目标列、统计量、当前最优模型）：\n{task_brief}"
    )
    return [
        {"role": "system", "content": _SYSTEM},
        {"role": "user", "content": user},
    ]


def build_chart_placement_messages(
    markdown: str,
    charts: list[dict[str, str]],
) -> list[dict[str, str]]:
    """分报告第二遍：只决定图放在哪。

    Deliberately tiny: the model reads finished text and answers with a JSON
    array, so this pass costs a fraction of the first one despite being a
    second round trip.
    """
    menu = "\n".join(f"  {c['id']} — {c['desc']}" for c in charts)
    user = (
        "下面是一份已经写好的模型分报告，以及本次可用的图表清单。"
        "请判断每张图应该插在第几个自然段之后，只输出 JSON 数组，不要任何其它文字：\n\n"

        '[{"chart": "图表id", "after_paragraph": 段落序号}]\n\n'

        "规则：\n"
        "- 段落序号从 1 开始，只数正文自然段，不数小标题行\n"
        "- 只能使用清单里的 id；不合适就不要放，宁缺毋滥\n"
        "- 一张图最多出现一次\n"
        "- 图应该紧跟在解释它的那句话所在的段落之后\n"
        "- 没有任何图适合就输出 []\n\n"

        f"可用图表：\n{menu}\n\n"
        f"报告正文：\n{markdown}"
    )
    return [
        {"role": "system", "content": "你只输出 JSON，不输出解释。"},
        {"role": "user", "content": user},
    ]


def apply_chart_placements(
    markdown: str,
    placements: Any,
    allowed_ids: set[str],
) -> tuple[str, list[str]]:
    """Insert chart markers after the paragraphs the placement pass chose.

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


def _parse_placements(raw: Any) -> Any:
    """Pull the JSON array out of a placement reply.

    Models wrap JSON in prose or fences however often they are told not to, and
    a wrapped-but-correct answer should not cost the report its figures.
    """
    import json

    if isinstance(raw, list):
        return raw
    text = str(raw or "").strip()
    if not text:
        return []
    fenced = re.search(r"```(?:json)?\s*(.+?)```", text, re.S)
    if fenced:
        text = fenced.group(1).strip()
    start, end = text.find("["), text.rfind("]")
    if start == -1 or end <= start:
        return []
    try:
        return json.loads(text[start:end + 1])
    except ValueError:
        return []


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def select_runs_for_reports(context: dict[str, Any]) -> list[dict[str, Any]]:
    """The runs worth narrating: successful ones, best first, capped."""
    runs = [
        r for r in (context.get("runs") or [])
        if str(r.get("status", "")).upper() == "SUCCESS"
    ]
    # The leaderboard is already ordered; fall back to rank when it is absent.
    runs.sort(key=lambda r: (r.get("rank") is None, r.get("rank") or 0))
    return runs[:_MAX_RUN_REPORTS]


async def generate_narrative_report(
    context: dict[str, Any],
    *,
    call_model: Callable[[list[dict[str, str]]], Any],
    task_type: str = "regression",
) -> dict[str, Any]:
    """Overall report first, then the per-run ones concurrently.

    Order matters: the overall verdict is what the page shows immediately, so
    it is not made to wait behind a batch of per-model calls.

    A failed sub-report degrades to an error note on that model rather than
    failing the whole request — the overall report and the other models are
    still worth showing.
    """
    overview = await call_model(build_overview_messages(context))

    runs = select_runs_for_reports(context)
    brief = build_task_brief(context)
    semaphore = asyncio.Semaphore(_MAX_CONCURRENT_RUN_REPORTS)

    async def _one(run: dict[str, Any]) -> dict[str, Any]:
        charts = available_run_charts(run, task_type)
        allowed = {c["id"] for c in charts}
        async with semaphore:
            try:
                markdown = await call_model(build_run_messages(run, brief))
            except Exception as exc:  # noqa: BLE001
                logger.warning("Run report failed for %s: %s", run.get("run_id"), exc)
                return {
                    "run_id": run.get("run_id"),
                    "model_type": run.get("model_type"),
                    "markdown": None,
                    "error": str(exc),
                }

            # Second pass, on finished text. Failing here loses the figures,
            # never the report — so it is caught separately and narrowly.
            dropped: list[str] = []
            if charts:
                try:
                    raw = await call_model(build_chart_placement_messages(markdown, charts))
                    markdown, dropped = apply_chart_placements(
                        markdown, _parse_placements(raw), allowed,
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "Chart placement failed for %s (report kept): %s",
                        run.get("run_id"), exc,
                    )

        if dropped:
            logger.info(
                "Run %s: dropped chart placements %s",
                run.get("run_id"), ", ".join(dropped),
            )
        placed = placed_chart_ids(markdown)
        return {
            "run_id": run.get("run_id"),
            "model_type": run.get("model_type"),
            "markdown": markdown,
            # Only what the text actually references: an unplaced chart is a
            # payload shipped to the browser for nothing.
            "charts": build_run_charts(run, placed),
            "dropped_charts": dropped,
        }

    run_reports = await asyncio.gather(*(_one(r) for r in runs))
    return {
        "overview": overview,
        "runs": list(run_reports),
        "runs_total": len([r for r in (context.get("runs") or [])
                           if str(r.get("status", "")).upper() == "SUCCESS"]),
        "runs_reported": len(run_reports),
    }


# ---------------------------------------------------------------------------
# Run-scoped chart builders
# ---------------------------------------------------------------------------
# Same shape as the task-level charts in ai_report_service so the existing
# frontend reader renders them unchanged: {id, title, description, type, option}.

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
                "grid": {"left": 56, "right": 20, "top": 34, "bottom": 40},
                "tooltip": {"trigger": "axis"},
                "legend": {"top": 0},
                "xAxis": {"type": "category", "data": epochs, "name": "Epoch"},
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
                "grid": {"left": 68, "right": 20, "top": 24, "bottom": 40},
                "tooltip": {"trigger": "axis"},
                "xAxis": {"type": "category", "data": [r.get("epoch") for r in history], "name": "Epoch"},
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
                    "grid": {"left": 56, "right": 20, "top": 24, "bottom": 40},
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
                    "grid": {"left": 60, "right": 20, "top": 34, "bottom": 40},
                    "tooltip": {"trigger": "axis"},
                    "legend": {"top": 0},
                    "xAxis": {"type": "category", "data": list(range(1, n + 1)), "name": "样本"},
                    "yAxis": {"type": "value", "scale": True},
                    "series": [
                        _line("实际值", actual[:n], "#dc2626"),
                        _line("预测值", predicted[:n], "#2563eb"),
                    ],
                },
            })

    return charts


def placed_chart_ids(markdown: str) -> list[str]:
    """Chart ids that survived into the finished text, in order of appearance."""
    seen: list[str] = []
    for match in _CHART_PLACEHOLDER.finditer(markdown or ""):
        chart_id = match.group(1).lower()
        if chart_id not in seen:
            seen.append(chart_id)
    return seen

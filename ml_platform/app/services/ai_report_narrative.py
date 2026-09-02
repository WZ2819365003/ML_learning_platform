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
    charts: list[dict[str, str]],
) -> list[dict[str, str]]:
    """分报告：这个模型训练得怎么样、结果怎么读。"""
    chart_menu = (
        "\n".join(f"  {{{{chart:{c['id']}}}}} — {c['desc']}" for c in charts)
        if charts else "  （本次没有可用图表，请勿插入任何 {{chart:...}}）"
    )
    user = (
        f"请为模型 {run.get('model_type')} 写一份简短的分报告，中文自然段，"
        "**不要表格、不要代码块**，总长度 350 字以内。写成两到三段：\n\n"

        "训练过程 — 这次训练是怎么进行的。有逐轮记录就讲收敛情况和早停位置；"
        "有交叉验证各折得分就讲波动来自哪里（是某一折异常，还是整体都在抖）。\n\n"

        "结果解读 — 这个模型的表现怎么读，指标要给参照系而不是裸数字。"
        "如果它不是最优模型，说明和最优模型差在哪、差距是否重要。\n\n"

        "值得注意 — 只在确实有异常时才写这一段，没有就不写。\n\n"

        "可用图表（**只能用下面列出的 id**，把占位符单独放一行，"
        "紧接在解释它的那句话之后；不需要图就不要放）：\n"
        f"{chart_menu}\n\n"

        "严禁自己生成图表数据或描述不存在的曲线。模型名保留原始英文标识。\n\n"

        f"该 Run 的数据：\n{run}\n\n"
        f"任务背景（目标列、统计量、当前最优模型）：\n{task_brief}"
    )
    return [
        {"role": "system", "content": _SYSTEM},
        {"role": "user", "content": user},
    ]


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
                markdown = await call_model(build_run_messages(run, brief, charts))
            except Exception as exc:  # noqa: BLE001
                logger.warning("Run report failed for %s: %s", run.get("run_id"), exc)
                return {
                    "run_id": run.get("run_id"),
                    "model_type": run.get("model_type"),
                    "markdown": None,
                    "error": str(exc),
                }
        markdown, dropped = resolve_chart_placeholders(markdown, allowed)
        if dropped:
            logger.info(
                "Run %s report referenced unavailable charts: %s",
                run.get("run_id"), ", ".join(dropped),
            )
        return {
            "run_id": run.get("run_id"),
            "model_type": run.get("model_type"),
            "markdown": markdown,
            "charts": sorted(allowed),
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

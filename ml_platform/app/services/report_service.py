"""Modeling-task report — renders a finalized task as Markdown (M3-1).

Why Markdown and not HTML: the report has to be diffable, greppable and
pasteable into whatever document the reader already uses. Embedded base64
charts would destroy all three, and rendering charts server-side would mean
standing up a second plotting stack for pictures the frontend already draws
better. The frontend renders this text and adds its existing ECharts figures
at the section anchors.

The one rule that shapes everything below: **selection metrics and final-test
metrics never share a table.** Leaderboard numbers are ``selection_cv_mean_*``
/ ``selection_val_*`` — cross-validated scores used to *choose* a model. The
final number is ``final_test_*``, measured once on the sealed hold-out. Put
them side by side and a reader will compare them and conclude "the model got
worse on test", which is exactly the misreading the B0/B1 evaluation-integrity
work exists to prevent. They are different measurements, not two runs of one.

See docs/superpowers/specs/2026-07-22-modeling-task-report-design.md.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from fastapi import HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.database import Dataset, ExperimentRun, ModelingTask
from app.services.modeling_task_service import (
    task_final_evaluation_state,
    task_leaderboard,
)

_MISSING = "—"
_TOP_K = 10


def _fmt(value: Any, decimals: int = 4) -> str:
    """Render a metric. Missing shows as an em dash, never blank or 'None'."""
    if value is None:
        return _MISSING
    if isinstance(value, bool):
        return "是" if value else "否"
    if isinstance(value, (int, float)):
        if isinstance(value, int) or float(value).is_integer():
            return str(int(value))
        return f"{value:.{decimals}f}"
    text = str(value).strip()
    return text or _MISSING


def _metric_label(key: str) -> str:
    """Strip the phase prefix for display; the phase is stated by the section."""
    for prefix in ("final_test_", "selection_cv_mean_", "selection_val_"):
        if key.startswith(prefix):
            return key[len(prefix):]
    return key


def _platform_version() -> str:
    try:
        from app.main import app

        return getattr(app, "version", None) or _MISSING
    except Exception:  # noqa: BLE001 — a report must not fail over its footer
        return _MISSING


async def build_task_report(db: AsyncSession, task_id: str) -> str:
    """Render a finalized modeling task as Markdown.

    Raises 409 when the task has not been finalized: before that the sealed
    hold-out is unopened, so the only numbers available are selection metrics —
    precisely the ones that must not be presented as "model performance".
    """
    task = (
        await db.execute(select(ModelingTask).where(ModelingTask.id == task_id))
    ).scalar_one_or_none()
    if task is None:
        raise HTTPException(status_code=404, detail=f"建模任务 {task_id} 不存在")

    state = task_final_evaluation_state(task)
    if state.get("state") != "FINALIZED":
        raise HTTPException(
            status_code=409,
            detail=(
                "该任务尚未执行最终评估，无法生成报告。"
                "请先在任务详情页执行「最终评估」，确认冠军模型在封存测试集上的表现。"
            ),
        )

    # The winner is whoever was frozen at finalize time — NOT today's leaderboard
    # leader. The leaderboard re-sorts as new experiments land, so reading rank 1
    # would let an already-published report silently change its subject.
    winner_run_id = state.get("winner_run_id")
    winner = (
        await db.execute(select(ExperimentRun).where(ExperimentRun.id == winner_run_id))
    ).scalar_one_or_none()
    if winner is None:
        raise HTTPException(
            status_code=409,
            detail=(
                f"报告数据不完整：最终评估记录的冠军 run（{winner_run_id}）已不存在，"
                "无法生成报告。"
            ),
        )

    dataset = None
    if task.dataset_id:
        dataset = (
            await db.execute(select(Dataset).where(Dataset.id == task.dataset_id))
        ).scalar_one_or_none()

    leaderboard = await task_leaderboard(db, task_id, top_k=_TOP_K)

    sections = [
        _section_headline(task, state, winner),
        _section_overview(task, dataset),
        _section_final_evaluation(task, state),
        _section_candidates(task, leaderboard, winner_run_id),
        "## 技术附录\n\n> 以下内容面向复现与追溯，外部读者可跳过。",
        _section_method(winner),
        _section_hyperparams(winner),
        _section_feature_importance(winner),
        _section_reproducibility(task, state),
    ]
    return "\n\n".join(s for s in sections if s).rstrip() + "\n"


# ---------------------------------------------------------------------------
# 上半部 · 结论
# ---------------------------------------------------------------------------

def _winner_model_name(winner: ExperimentRun) -> str:
    return str((winner.params or {}).get("model_type") or _MISSING)


def _section_headline(task: ModelingTask, state: dict, winner: ExperimentRun) -> str:
    objective = task.objective_metric or "accuracy"
    final_metrics = state.get("final_metrics") or {}
    value = final_metrics.get(f"final_test_{objective}")
    if value is None and final_metrics:
        # Objective not present under its own name; fall back to the sole metric.
        value = next(iter(final_metrics.values()))

    return (
        f"# 建模任务报告 · {task.name or task_id_short(task.id)}\n\n"
        f"**结论：选定模型为 `{_winner_model_name(winner)}`，"
        f"在封存测试集上的 {objective} 为 {_fmt(value)}。**"
    )


def task_id_short(task_id: str | None) -> str:
    return (task_id or "")[:8] or _MISSING


def _section_overview(task: ModelingTask, dataset: Dataset | None) -> str:
    rows = [
        ("数据集", task.dataset_name or (dataset.name if dataset else None)),
        ("样本量", dataset.row_count if dataset else None),
        ("任务类型", task.task_type),
        ("目标列", task.target_column),
        ("目标指标", task.objective_metric),
        ("优化方向", "越大越好" if (task.objective_direction or "max") == "max" else "越小越好"),
    ]
    body = "\n".join(f"| {label} | {_fmt(value)} |" for label, value in rows)
    return "## 1. 任务概览\n\n| 项 | 值 |\n|---|---|\n" + body


def _section_final_evaluation(task: ModelingTask, state: dict) -> str:
    final_metrics = state.get("final_metrics") or {}
    if not final_metrics:
        body = "最终评估未产生指标。"
    else:
        rows = "\n".join(
            f"| {_metric_label(k)} | {_fmt(v)} |" for k, v in sorted(final_metrics.items())
        )
        body = "| 指标 | 值 |\n|---|---|\n" + rows
    return (
        "## 2. 最终评估\n\n"
        "> **封存测试集结果，全程仅开启一次。** 这些数据在模型选择阶段完全未被使用，"
        "因此是对模型泛化能力的无偏估计。\n\n" + body
    )


def _section_candidates(
    task: ModelingTask, leaderboard: list[dict[str, Any]], winner_run_id: str | None
) -> str:
    header = (
        "## 3. 候选模型对比\n\n"
        "> ⚠️ **以下为模型选择阶段指标（交叉验证均值），不可与第 2 节的最终评估结果"
        "直接比较。** 两者是不同的测量：本节用于在候选之间排序，第 2 节用于报告"
        "最终性能。数值不同属于正常现象，不代表模型「在测试集上变差了」。\n"
    )
    if not leaderboard:
        return header + "\n无其他候选模型。"

    objective = task.objective_metric or "accuracy"
    rows = []
    for entry in leaderboard:
        metrics = entry.get("metrics") or {}
        score = (
            metrics.get(f"selection_cv_mean_{objective}")
            or metrics.get(f"selection_val_{objective}")
            or metrics.get(objective)
        )
        mark = " ⭐" if entry.get("run_id") == winner_run_id else ""
        model = str((entry.get("params") or {}).get("model_type") or _MISSING)
        rows.append(
            f"| {entry.get('rank', _MISSING)} | `{model}`{mark} | "
            f"{_fmt(score)} | {entry.get('strategy_type') or _MISSING} |"
        )

    note = (
        f"\n\n⭐ 标记为最终选定模型。仅列前 {_TOP_K} 名，"
        "完整榜单见平台内任务详情页。"
    )
    return (
        header
        + f"\n| 排名 | 模型 | 选择阶段 {objective} | 策略 |\n|---|---|---|---|\n"
        + "\n".join(rows)
        + note
    )


# ---------------------------------------------------------------------------
# 下半部 · 技术附录
# ---------------------------------------------------------------------------

def _section_method(winner: ExperimentRun) -> str:
    params = winner.params or {}
    meta = winner.search_meta or {}
    rows = [
        ("交叉验证折数", params.get("cv_folds")),
        ("测试集比例", params.get("test_size")),
        ("随机种子", params.get("random_state")),
        ("评估模式", meta.get("evaluation_mode")),
    ]
    body = "\n".join(f"| {label} | {_fmt(value)} |" for label, value in rows)
    return (
        "### 4. 评估方法\n\n| 项 | 值 |\n|---|---|\n" + body + "\n\n"
        "交叉验证的每一折都**只在训练折内**拟合预处理与模型，折外数据不参与任何拟合，"
        "以避免信息泄漏。封存测试集在模型选择全程不可见，仅在最终评估时开启一次。"
    )


def _section_hyperparams(winner: ExperimentRun) -> str:
    params = dict(winner.params or {})
    hyper = params.get("hyperparameters")
    shown = hyper if isinstance(hyper, dict) and hyper else params
    if not shown:
        return "### 5. 冠军模型超参数\n\n未记录超参数。"
    rows = "\n".join(f"| `{k}` | {_fmt(v)} |" for k, v in sorted(shown.items()))
    return "### 5. 冠军模型超参数\n\n| 参数 | 值 |\n|---|---|\n" + rows


def _section_feature_importance(winner: ExperimentRun) -> str:
    importances = (winner.metrics or {}).get("shap_importances")
    if not isinstance(importances, dict) or not importances:
        return (
            "### 6. 特征重要性\n\n"
            "该模型未生成特征重要性（SHAP 未计算或计算失败）。"
        )
    # shap_service builds this from ``mean_abs_shap``, so values are already
    # non-negative and pre-sorted. Re-sorting defensively costs nothing and
    # keeps the report correct if that upstream contract ever changes; abs()
    # guards the same way without altering today's output.
    top = sorted(importances.items(), key=lambda kv: abs(kv[1] or 0), reverse=True)[:_TOP_K]
    rows = "\n".join(f"| {i} | `{name}` | {_fmt(value)} |" for i, (name, value) in enumerate(top, 1))
    return (
        f"### 6. 特征重要性（SHAP 前 {len(top)} 名）\n\n"
        "| # | 特征 | 平均绝对 SHAP 值 |\n|---|---|---|\n" + rows
    )


def _section_reproducibility(task: ModelingTask, state: dict) -> str:
    rows = [
        ("建模任务 ID", task.id),
        ("数据集版本 ID", task.dataset_version_id),
        ("冠军 run ID", state.get("winner_run_id")),
        ("最终评估 ID", state.get("evaluation_id")),
        ("最终评估时间", state.get("finalized_at")),
        ("平台版本", _platform_version()),
        ("报告生成时间 (UTC)", datetime.now(timezone.utc).isoformat(timespec="seconds")),
    ]
    body = "\n".join(f"| {label} | {_fmt(value)} |" for label, value in rows)
    return "### 7. 复现信息\n\n| 项 | 值 |\n|---|---|\n" + body

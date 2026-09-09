"""Doubao/Ark-backed AI report generation for modeling tasks."""
from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from typing import Any

import httpx
from fastapi import HTTPException
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.models.database import (
    AIReportArchive,
    Dataset,
    ExperimentRun,
    ModelingTask,
    PlatformExperiment,
)
from app.services.modeling_task_service import (
    task_final_evaluation_state,
    task_leaderboard,
)
from app.services.report_facts import validation_scheme

logger = logging.getLogger(__name__)

_TOP_RUNS = 8
_MAX_CONTEXT_CHARS = 12000
_SCATTER_POINTS = 500
# v2: charts are semantic specs, the prose carries {{chart:id}} markers, and
# report_blocks / tables / headline_metrics are gone (see build_rich_report_payload).
_REPORT_SCHEMA_VERSION = "ai_report.rich.v2"


def _utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _require_api_key(settings: Any) -> str:
    key = (getattr(settings, "doubao_api_key", "") or "").strip()
    if not key:
        raise HTTPException(
            status_code=503,
            detail="未配置 ARK_API_KEY/DOUBAO_API_KEY，无法生成豆包 AI 报告。",
        )
    return key


def _compact_value(value: Any, *, depth: int = 0) -> Any:
    if depth >= 3:
        return str(value)[:160]
    if isinstance(value, dict):
        return {
            str(k)[:80]: _compact_value(v, depth=depth + 1)
            for k, v in list(value.items())[:16]
        }
    if isinstance(value, (list, tuple)):
        return [_compact_value(v, depth=depth + 1) for v in list(value)[:12]]
    if isinstance(value, (str, int, float, bool)) or value is None:
        if isinstance(value, str):
            return value[:500]
        return value
    return str(value)[:160]


_CURVE_METRIC_KEYS = {
    "history",
    "training_history",
    "loss_history",
    "evals_result",
    "val_roc_fpr",
    "val_roc_tpr",
    "roc_fpr",
    "roc_tpr",
    "fpr",
    "tpr",
    "y_true",
    "y_pred",
    "actual",
    "predicted",
    "actuals",
    "predictions",
    "prediction_curve",
    # A dict of parallel actual/predicted lists. Without this it fell through to
    # the general branch of _compact_metrics, where it lost the race for the ten
    # non-scalar slots on alphabetical order and was dropped outright — so the
    # 实际值 vs 预测值 chart could never be built from a leaderboard entry.
    "val_scatter",
}


def _compact_curve_value(value: Any, *, limit: int = 120) -> Any:
    if isinstance(value, dict):
        # Parallel series (val_scatter's actual/predicted). Truncating each to
        # the same limit keeps them aligned; _compact_value would cut them to
        # twelve points, which plots as a curve that is not one.
        return {
            str(key)[:80]: (
                _compact_curve_value(val, limit=limit)
                if isinstance(val, (list, tuple, dict))
                else _compact_value(val, depth=2)
            )
            for key, val in list(value.items())[:12]
        }
    if isinstance(value, (list, tuple)):
        compact_items: list[Any] = []
        for item in list(value)[:limit]:
            if isinstance(item, dict):
                compact_items.append({
                    str(key)[:80]: _compact_value(val, depth=2)
                    for key, val in list(item.items())[:12]
                })
            else:
                compact_items.append(_compact_value(item, depth=2))
        return compact_items
    return _compact_value(value)


def _curve_prompt_summary(value: Any) -> Any:
    if isinstance(value, (list, tuple)):
        items = list(value)
        if not items:
            return {"points": 0}
        return {
            "points": len(items),
            "first": _compact_value(items[0], depth=2),
            "last": _compact_value(items[-1], depth=2),
        }
    if isinstance(value, dict):
        summary: dict[str, Any] = {}
        for key, item in list(value.items())[:8]:
            if isinstance(item, (list, tuple)):
                summary[str(key)] = {"points": len(item)}
            elif isinstance(item, dict):
                summary[str(key)] = _curve_prompt_summary(item)
            else:
                summary[str(key)] = _compact_value(item, depth=2)
        return summary
    return _compact_value(value, depth=2)


def _context_for_llm(value: Any, *, depth: int = 0) -> Any:
    if depth >= 5:
        return _compact_value(value, depth=depth)
    if isinstance(value, dict):
        result: dict[str, Any] = {}
        for key, item in list(value.items())[:40]:
            key_text = str(key)
            if key_text in {"shap_importances", "top_shap_importances"}:
                continue
            if key_text in _CURVE_METRIC_KEYS:
                result[key_text] = _curve_prompt_summary(item)
            else:
                result[key_text] = _context_for_llm(item, depth=depth + 1)
        return result
    if isinstance(value, (list, tuple)):
        return [_context_for_llm(item, depth=depth + 1) for item in list(value)[:16]]
    return _compact_value(value, depth=depth)


def _compact_metrics(metrics: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(metrics, dict):
        return {}
    compact: dict[str, Any] = {}
    shap = metrics.get("shap_importances")
    for key, value in sorted(metrics.items()):
        if key == "shap_importances" or key in _CURVE_METRIC_KEYS:
            continue
        if isinstance(value, (int, float, str, bool)) or value is None:
            compact[key] = value
        elif len(compact) < 10:
            compact[key] = _compact_value(value)
        if len(compact) >= 16:
            break
    for key in _CURVE_METRIC_KEYS:
        if key in metrics:
            # The trainers keep a 500-point validation tail; the predicted-vs-
            # actual chart wants all of it, and the prompt only ever sees a
            # point count, so nothing is saved by cutting it to 120 here.
            limit = _SCATTER_POINTS if key == "val_scatter" else 120
            compact[key] = _compact_curve_value(metrics[key], limit=limit)
    # Travels with val_scatter. Alphabetical order puts it past the sixteen-key
    # cap above, and without it a cross-validation fold is captioned as a
    # hold-out — the chart claims an evaluation that never happened.
    if "val_scatter_source" in metrics:
        compact["val_scatter_source"] = metrics["val_scatter_source"]
    if isinstance(shap, dict) and shap:
        top = sorted(
            shap.items(),
            key=lambda kv: abs(kv[1] or 0) if isinstance(kv[1], (int, float)) else 0,
            reverse=True,
        )[:8]
        compact["top_shap_importances"] = [
            {"feature": str(name), "mean_abs_shap": value}
            for name, value in top
        ]
    return compact


def _compact_params(params: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(params, dict):
        return {}
    keys = ["model_type", "hyperparameters", "cv_folds", "test_size", "random_state"]
    selected = {key: params.get(key) for key in keys if key in params}
    if not selected:
        selected = dict(list(params.items())[:10])
    return _compact_value(selected)


def _model_name_from_params(params: dict[str, Any] | None) -> str | None:
    if not isinstance(params, dict):
        return None
    value = params.get("model_type") or params.get("model")
    return str(value) if value else None


def _round_number(value: Any, decimals: int = 4) -> float | None:
    if not isinstance(value, (int, float)):
        return None
    return round(float(value), decimals)


def _fmt_value(value: Any, decimals: int = 4) -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "是" if value else "否"
    if isinstance(value, (int, float)):
        if float(value).is_integer():
            return str(int(value))
        return f"{float(value):.{decimals}f}"
    text = str(value).strip()
    return text or "—"


# ---------------------------------------------------------------------------
# Reference frames
# ---------------------------------------------------------------------------
# A metric on its own says nothing to a reader: "RMSE 72.47" is only meaningful
# once you know the target averages 8897. These translations are computed here
# rather than asked of the model, because arithmetic is exactly what an LLM is
# least reliable at, and a wrong ratio in a report is worse than no ratio.

_ERROR_METRIC_KEYS = ("rmse", "mae", "mse", "final_test_rmse", "final_test_mae")


def _target_column_stats(context: dict[str, Any]) -> dict[str, Any] | None:
    """Numeric summary of the target column, when the dataset profile has one."""
    task = context.get("task") or {}
    dataset = context.get("dataset") or {}
    target = task.get("target_column")
    if not target:
        return None
    for name, info in _iter_column_info(dataset.get("columns_info")):
        if name != target or not isinstance(info, dict):
            continue
        mean = info.get("mean")
        if not isinstance(mean, (int, float)):
            return None
        return {
            "mean": float(mean),
            "min": info.get("min"),
            "max": info.get("max"),
            "std": info.get("std"),
        }
    return None


def _majority_class_share(context: dict[str, Any]) -> float | None:
    """Share of the largest class — the accuracy of always guessing it.

    A classifier at 0.92 reads very differently depending on whether this is
    0.50 or 0.90, and the reader cannot tell without being told.
    """
    task = context.get("task") or {}
    dataset = context.get("dataset") or {}
    target = task.get("target_column")
    if not target:
        return None
    for name, info in _iter_column_info(dataset.get("columns_info")):
        if name != target or not isinstance(info, dict):
            continue
        counts = info.get("value_counts") or info.get("top_values")
        if isinstance(counts, dict) and counts:
            values = [v for v in counts.values() if isinstance(v, (int, float))]
            if values and sum(values) > 0:
                return max(values) / sum(values)
        return None
    return None


def build_reference_frames(context: dict[str, Any]) -> dict[str, Any]:
    """Plain-language translations of this task's headline numbers.

    Handed to the model as established facts it may quote, so the report can say
    "off by about 0.8% of a typical value" instead of restating the raw metric.
    """
    task = context.get("task") or {}
    task_type = str(task.get("task_type") or "").lower()
    frames: dict[str, Any] = {}

    best = (context.get("leaderboard") or [{}])[0] if context.get("leaderboard") else {}
    metrics = {**(context.get("metrics") or {}), **(best.get("metrics") or {})}

    if task_type == "regression":
        stats = _target_column_stats(context)
        if stats and abs(stats["mean"]) > 1e-9:
            frames["target"] = {
                "column": task.get("target_column"),
                "mean": round(stats["mean"], 4),
                "range": [stats.get("min"), stats.get("max")],
            }
            for key in _ERROR_METRIC_KEYS:
                value = metrics.get(key)
                if isinstance(value, (int, float)):
                    pct = abs(value) / abs(stats["mean"]) * 100
                    frames[key] = {
                        "value": round(float(value), 4),
                        "pct_of_mean": round(pct, 2),
                        "plain": (
                            f"{key.upper()} {value:.4g}，约为目标列均值 "
                            f"{stats['mean']:.4g} 的 {pct:.1f}%"
                        ),
                    }
    else:
        share = _majority_class_share(context)
        if share is not None:
            frames["majority_baseline"] = {
                "share": round(share, 4),
                "plain": (
                    f"多数类占比 {share * 100:.1f}%，即“永远猜多数类”就能拿到 "
                    f"{share * 100:.1f}% 的准确率；模型的准确率必须显著高于这个数才有意义"
                ),
            }
    return frames


# Weights for the readiness score. They sum to 100 and are stated in the report
# so the number can be argued with — the point is an auditable summary, not a
# precise measurement.
_SCORE_RUBRIC = (
    ("final_evaluation", 40, "已在封存测试集上完成最终评估"),
    ("cross_fold_stability", 30, "跨折波动可接受（最大变异系数 ≤ 15%）"),
    ("run_success", 30, "本任务的 Run 全部训练成功"),
)


_SUCCESS_STATUSES = frozenset({"SUCCESS", "COMPLETED"})


def _cv_variation_percentages(metric_maps: Any) -> list[float]:
    """Coefficient of variation, as a percentage, for each cv_std_/cv_avg_ pair."""
    out: list[float] = []
    for metrics in metric_maps:
        if not isinstance(metrics, dict):
            continue
        for key, std in metrics.items():
            if not key.startswith("cv_std_") or not isinstance(std, (int, float)):
                continue
            avg = metrics.get(f"cv_avg_{key[len('cv_std_'):]}")
            if isinstance(avg, (int, float)) and abs(avg) > 1e-9:
                out.append(abs(std) / abs(avg) * 100)
    return out


def compute_readiness_score(context: dict[str, Any]) -> dict[str, Any]:
    """Score this task against a fixed, published rubric.

    Replaces a number the model used to invent: the prompt asked for
    "总分：xx/100" and the backend scraped it back out with a regex, so the
    same task could score differently on a rerun and nothing anywhere defined
    what the number meant. Deriving it from facts already in the context makes
    it reproducible and lets the report show its working.
    """
    task = context.get("task") or {}
    # `leaderboard`, `successful_run_examples` and `run_status_counts` are what
    # build_task_report_context actually emits. A top-level `runs` or `metrics`
    # never existed, so the two checks below read nothing and scored every task
    # 0/100 in silence — no error, just a number that was always wrong.
    runs = context.get("runs") or context.get("leaderboard") or []
    run_examples = context.get("successful_run_examples") or []

    checks: list[dict[str, Any]] = []

    finalized = bool(
        task.get("final_evaluation_state") == "FINALIZED"
        or task.get("final_test_value") is not None
        or any(r.get("final_test_value") is not None for r in runs if isinstance(r, dict))
    )
    checks.append({"key": "final_evaluation", "passed": finalized})

    # The cv_std_/cv_avg_ pairs live in each run's own metrics map.
    cvs = _cv_variation_percentages(
        [context.get("metrics")]
        + [r.get("metrics") for r in runs if isinstance(r, dict)]
        + [r.get("metrics") for r in run_examples if isinstance(r, dict)]
    )
    # No CV data is not a failure — it is an unknown, and scoring an unknown as
    # a pass would overstate readiness.
    checks.append({
        "key": "cross_fold_stability",
        "passed": bool(cvs) and max(cvs) <= 15,
        "detail": f"最大变异系数 {max(cvs):.1f}%" if cvs else "无交叉验证数据",
    })

    # run_status_counts covers every Run including failed ones, which is
    # exactly the population this check is about. Leaderboard entries carry no
    # status at all, so counting them read "" and failed 7/7-successful tasks.
    counts = context.get("run_status_counts") or {}
    if counts:
        total = sum(v for v in counts.values() if isinstance(v, int))
        succeeded = sum(
            v for k, v in counts.items()
            if isinstance(v, int) and str(k).upper() in _SUCCESS_STATUSES
        )
    else:
        statuses = [
            str(r.get("status") or "").upper()
            for r in (run_examples or runs) if isinstance(r, dict)
        ]
        total = len(statuses)
        succeeded = sum(1 for st in statuses if st in _SUCCESS_STATUSES)
    checks.append({
        "key": "run_success",
        "passed": total > 0 and succeeded == total,
        "detail": f"{succeeded}/{total} 成功" if total else "无 Run 记录",
    })

    weights = {key: weight for key, weight, _ in _SCORE_RUBRIC}
    labels = {key: label for key, _, label in _SCORE_RUBRIC}
    score = sum(weights[c["key"]] for c in checks if c["passed"])
    for check in checks:
        check["weight"] = weights[check["key"]]
        check["label"] = labels[check["key"]]

    return {"score": score, "checks": checks}


def _extract_ai_score(markdown: str) -> int | None:
    """Legacy scrape of a model-written 总分.

    Kept only for archived reports generated before the score became computed;
    new reports carry it in the payload instead.
    """
    match = re.search(r"总分\s*[:：]\s*(\d{1,3})(?:\s*/\s*100)?", markdown or "")
    if not match:
        return None
    return max(0, min(100, int(match.group(1))))


def _metric_value_from_entry(entry: dict[str, Any]) -> float | None:
    for key in ("objective_value", "selection_value", "final_test_value"):
        value = _round_number(entry.get(key))
        if value is not None:
            return value
    return None


def _final_metric_value(task: dict[str, Any]) -> tuple[str | None, Any]:
    objective = task.get("objective_metric") or ""
    final_eval = task.get("final_evaluation") or {}
    metrics = final_eval.get("final_metrics") or {}
    if not isinstance(metrics, dict) or not metrics:
        return None, None
    preferred = f"final_test_{objective}"
    if preferred in metrics:
        return preferred, metrics[preferred]
    key = next(iter(metrics.keys()))
    return key, metrics[key]


def _best_run_level_final(context: dict[str, Any]) -> dict[str, Any] | None:
    task = context.get("task") or {}
    direction = task.get("objective_direction") or "max"
    candidates: list[tuple[float, dict[str, Any]]] = []
    for item in context.get("leaderboard") or []:
        value = _round_number(item.get("final_test_value"))
        if value is not None:
            candidates.append((value, item))
    if not candidates:
        return None
    return sorted(candidates, key=lambda pair: pair[0], reverse=direction == "max")[0][1]


def _available_final_metric(context: dict[str, Any]) -> tuple[str | None, Any, str | None]:
    task = context.get("task") or {}
    key, value = _final_metric_value(task)
    if value is not None:
        return key, value, "task_final"

    objective = task.get("objective_metric") or ""
    item = _best_run_level_final(context)
    if item is not None:
        key = item.get("final_test_metric_key") or (f"final_test_{objective}" if objective else "final_test")
        return key, item.get("final_test_value"), "run_level"
    return None, None, None


def _percentage_text(value: Any) -> str:
    if not isinstance(value, (int, float)):
        return "—"
    ratio = float(value)
    if 0 <= ratio <= 1:
        ratio *= 100
    return f"{ratio:.2f}%"


def _short_json(value: Any, *, max_len: int = 96) -> str:
    if value is None:
        return "—"
    if isinstance(value, str):
        text = value
    else:
        text = json.dumps(value, ensure_ascii=False, default=str)
    text = text.strip()
    if not text:
        return "—"
    return text if len(text) <= max_len else text[: max_len - 1] + "…"


def _iter_column_info(columns_info: Any) -> list[tuple[str, dict[str, Any]]]:
    if isinstance(columns_info, dict):
        rows = []
        for name, info in columns_info.items():
            if isinstance(info, dict):
                rows.append((str(name), info))
            else:
                rows.append((str(name), {"dtype": info}))
        return rows
    if isinstance(columns_info, list):
        rows = []
        for index, item in enumerate(columns_info):
            if not isinstance(item, dict):
                rows.append((f"column_{index + 1}", {"dtype": item}))
                continue
            name = item.get("name") or item.get("column") or item.get("field") or f"column_{index + 1}"
            rows.append((str(name), item))
        return rows
    return []


def _column_role(column: str, info: dict[str, Any], target_column: str | None) -> str:
    if target_column and column == target_column:
        return "目标列"
    lowered = str(column or "").lower()
    if lowered.endswith(("_sin", "_cos")):
        return "周期编码特征"
    if re.search(r"_lag_\d+$", lowered):
        return "滞后特征"
    if "_roll_" in lowered:
        return "滚动统计特征"
    dtype = str(info.get("dtype") or info.get("type") or "").lower()
    unique_count = info.get("unique_count")
    if "bool" in dtype:
        return "布尔特征"
    if any(token in dtype for token in ("int", "float", "double", "decimal", "number")):
        if isinstance(unique_count, (int, float)) and unique_count <= 20:
            return "离散数值特征"
        return "数值特征"
    if any(token in dtype for token in ("category", "object", "string", "str")):
        if isinstance(unique_count, (int, float)) and unique_count > 50:
            return "高基数字段"
        return "类别/文本特征"
    return "输入字段"


def _column_note(role: str, info: dict[str, Any]) -> str:
    missing_count = info.get("missing_count")
    missing_rate = info.get("missing_rate")
    notes: list[str] = []
    if role == "目标列":
        notes.append("模型学习的标签")
    elif role == "数值特征":
        notes.append("可直接参与建模，必要时做缩放或分箱")
    elif role == "离散数值特征":
        notes.append("需确认是编码类别还是连续数值")
    elif role == "周期编码特征":
        notes.append("以正弦/余弦保留周期边界的连续关系")
    elif role == "滞后特征":
        notes.append("引用历史时刻，验证时必须保持时间顺序")
    elif role == "滚动统计特征":
        notes.append("概括近期窗口，验证时必须避免未来信息进入窗口")
    elif role == "高基数字段":
        notes.append("进入模型前通常需要编码或降维")
    elif role == "类别/文本特征":
        notes.append("进入模型前通常需要编码")
    if isinstance(missing_count, (int, float)) and missing_count > 0:
        notes.append(f"存在 {_fmt_value(missing_count)} 个缺失")
    elif isinstance(missing_rate, (int, float)) and missing_rate > 0:
        notes.append(f"缺失率 {_percentage_text(missing_rate)}")
    return "；".join(notes) or "字段质量未见明显异常"


_FIELD_LABELS = {
    "age": "年龄",
    "tenure": "使用时长",
    "monthly_charges": "月费用",
    "churn": "是否流失",
    "target": "预测目标",
    "type": "设备类型",
    "air_temperature": "空气温度",
    "process_temperature": "过程温度",
    "rotational_speed": "转速",
    "torque": "扭矩",
    "tool_wear": "刀具磨损",
    "machine_failure": "设备故障",
    "product_id": "产品编号",
}

_FIELD_TOKEN_LABELS = {
    "age": "年龄",
    "air": "空气",
    "amount": "金额",
    "charge": "费用",
    "charges": "费用",
    "churn": "流失",
    "count": "数量",
    "customer": "客户",
    "failure": "故障",
    "id": "ID",
    "label": "标签",
    "machine": "设备",
    "monthly": "月",
    "process": "过程",
    "product": "产品",
    "rotational": "旋转",
    "speed": "速度",
    "target": "目标",
    "temperature": "温度",
    "tenure": "使用时长",
    "tool": "刀具",
    "torque": "扭矩",
    "type": "类型",
    "wear": "磨损",
}


def _readable_field_name(column: str, *, role: str | None = None) -> str:
    raw = str(column or "").strip()
    if not raw:
        return "—"
    key = re.sub(r"\[[^\]]+\]", "", raw.lower()).strip()
    key = re.sub(r"[\s\-]+", "_", key)
    label = _FIELD_LABELS.get(key)
    if label is None:
        tokens = [token for token in re.split(r"[_\-\s]+", key) if token]
        translated = [_FIELD_TOKEN_LABELS.get(token) for token in tokens]
        if tokens and all(translated):
            label = "".join(str(item) for item in translated)
    if label is None and role == "目标列":
        label = "预测目标"
    return f"{label}（{raw}）" if label and label != raw else raw


def _build_data_profile_table(context: dict[str, Any]) -> dict[str, Any] | None:
    task = context.get("task") or {}
    dataset = context.get("dataset") or {}
    columns_info = _iter_column_info(dataset.get("columns_info"))
    rows = []
    for column, info in columns_info[:40]:
        role = _column_role(column, info, task.get("target_column"))
        rows.append({
            "column": _readable_field_name(column, role=role),
            "role": role,
            "dtype": info.get("dtype") or info.get("type") or "—",
            "missing_count": _fmt_value(info.get("missing_count")),
            "missing_rate": _percentage_text(info.get("missing_rate")),
            "unique_count": _fmt_value(info.get("unique_count")),
            "note": _column_note(role, info),
        })
    if not rows:
        return None
    return {
        "id": "data_profile",
        "title": "数据字段概况",
        "columns": [
            {"key": "column", "title": "字段"},
            {"key": "role", "title": "角色"},
            {"key": "dtype", "title": "类型"},
            {"key": "missing_count", "title": "缺失数"},
            {"key": "missing_rate", "title": "缺失率"},
            {"key": "unique_count", "title": "唯一值"},
            {"key": "note", "title": "解读"},
        ],
        "rows": rows,
    }


def _merge_run_context(context: dict[str, Any]) -> list[dict[str, Any]]:
    successful_by_id = {
        run.get("run_id"): run
        for run in context.get("successful_run_examples") or []
        if run.get("run_id")
    }
    merged: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in context.get("leaderboard") or []:
        run_id = item.get("run_id")
        row = dict(successful_by_id.get(run_id) or {})
        row.update(item)
        if run_id:
            seen.add(run_id)
        merged.append(row)
    for run in context.get("successful_run_examples") or []:
        run_id = run.get("run_id")
        if run_id and run_id in seen:
            continue
        merged.append(run)
    return merged[:_TOP_RUNS]


def _key_params_text(row: dict[str, Any]) -> str:
    params = row.get("params") or {}
    if isinstance(params, dict):
        hyperparameters = params.get("hyperparameters")
        if hyperparameters:
            return _short_json(hyperparameters, max_len=120)
        remaining = {
            key: value
            for key, value in params.items()
            if key not in {"model_type", "model", "cv_folds", "test_size", "random_state"}
        }
        if remaining:
            return _short_json(remaining, max_len=120)
    return "未记录关键超参数"


def _validation_setting_text(row: dict[str, Any]) -> str:
    params = row.get("params") or {}
    parts: list[str] = []
    if isinstance(params, dict):
        if params.get("cv_folds") is not None:
            parts.append(f"cv_folds={params['cv_folds']}")
        if params.get("test_size") is not None:
            parts.append(f"test_size={params['test_size']}")
        if params.get("random_state") is not None:
            parts.append(f"random_state={params['random_state']}")
    search_meta = row.get("search_meta") or {}
    if isinstance(search_meta, dict):
        mode = search_meta.get("evaluation_mode") or search_meta.get("search_mode")
        if mode:
            parts.append(f"evaluation_mode={mode}")
    return "；".join(str(part) for part in parts) or "未记录验证设置"


def _build_parameter_settings_table(context: dict[str, Any]) -> dict[str, Any] | None:
    rows = []
    for row in _merge_run_context(context):
        rows.append({
            "model_type": row.get("model_type") or "—",
            "strategy_type": row.get("strategy_type") or "—",
            "trial_no": _fmt_value(row.get("trial_no")),
            "validation_setting": _validation_setting_text(row),
            "key_params": _key_params_text(row),
            "run_id": row.get("run_id"),
        })
    if not rows:
        return None
    return {
        "id": "parameter_settings",
        "title": "参数设置",
        "columns": [
            {"key": "model_type", "title": "模型"},
            {"key": "strategy_type", "title": "策略"},
            {"key": "trial_no", "title": "Trial"},
            {"key": "validation_setting", "title": "验证设置"},
            {"key": "key_params", "title": "关键参数"},
            {"key": "run_id", "title": "Run ID"},
        ],
        "rows": rows,
    }


def _build_evidence(context: dict[str, Any]) -> list[str]:
    task = context.get("task") or {}
    dataset = context.get("dataset") or {}
    counts = context.get("run_status_counts") or {}
    leaderboard = context.get("leaderboard") or []
    final_key, final_value, final_source = _available_final_metric(context)
    evidence = [
        f"任务：{task.get('name') or '—'}，类型：{task.get('task_type') or '—'}，目标指标：{task.get('objective_metric') or '—'}。",
        f"数据集：{dataset.get('name') or '—'}，样本量：{_fmt_value(dataset.get('row_count'))}，列数：{_fmt_value(dataset.get('column_count'))}。",
        f"Run 状态：成功 {counts.get('SUCCESS', 0)}，失败 {counts.get('FAILED', 0)}，运行中 {counts.get('RUNNING', 0)}。",
    ]
    final_best = _best_run_level_final(context)
    if final_best is not None:
        evidence.append(
            f"当前最终测试最佳：{final_best.get('model_type') or '—'}，最终测试分：{_fmt_value(final_best.get('final_test_value'))}。"
        )
    elif leaderboard:
        best = leaderboard[0]
        evidence.append(
            f"当前选择阶段榜首：{best.get('model_type') or '—'}，选择分：{_fmt_value(_metric_value_from_entry(best))}。"
        )
    if final_value is not None:
        prefix = "Run 级最终测试指标" if final_source == "run_level" else "最终测试指标"
        evidence.append(f"{prefix}：{final_key} = {_fmt_value(final_value)}。")
    else:
        evidence.append("最终测试指标：尚未执行最终评估。")
    return evidence


def _enrich_context(context: dict[str, Any]) -> dict[str, Any]:
    """The context plus the target statistics the facts and charts read.

    Computed once here rather than inside the overall report's prompt, which is
    where it used to live — every sub-report then saw a null target mean and
    reached for the number in the prompt's own example.
    """
    enriched = dict(context)
    if "_target_stats" not in enriched:
        enriched["_target_stats"] = _target_column_stats(context)
    return enriched


def _build_meta(context: dict[str, Any]) -> dict[str, Any]:
    task = context.get("task") or {}
    dataset = context.get("dataset") or {}
    counts = context.get("run_status_counts") or {}
    board = context.get("leaderboard") or []
    return {
        "task_name": task.get("name"),
        "dataset_name": dataset.get("name") or task.get("dataset_name"),
        "target_column": task.get("target_column"),
        "task_type": task.get("task_type"),
        "objective_metric": task.get("objective_metric"),
        "run_count": sum(int(v) for v in counts.values()) if counts else len(board),
        "model_count": (len(context.get("model_types") or [])
                        or len({e.get("model_type") for e in board if e.get("model_type")})),
    }


def _build_appendix_tables(context: dict[str, Any]) -> list[dict[str, Any]]:
    """The two wide tables that stay out of the prose: column profile, parameters."""
    return [
        table
        for table in (_build_data_profile_table(context), _build_parameter_settings_table(context))
        if table is not None
    ]


def build_rich_report_payload(
    context: dict[str, Any] | str,
    markdown: str,
    charts: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Everything the page needs beyond the markdown, in the payload contract.

    Charts are semantic specs (see report_charts), never renderer options; when
    the caller has not already built and placed them, they are built here and
    filtered to the {{chart:id}} markers the document actually carries.
    """
    from app.services import ai_report_narrative, report_charts, report_facts

    structured = _enrich_context(context if isinstance(context, dict) else {})
    facts = report_facts.build_overview_facts(structured)
    if charts is None:
        charts = ai_report_narrative.keep_placed(
            report_charts.build_overview_charts(structured), markdown,
        )
    return {
        "report_schema_version": _REPORT_SCHEMA_VERSION,
        "headline": (facts.get("headline") or {}).get("sentence") or "",
        "meta": _build_meta(structured),
        "charts": charts,
        "appendix_tables": _build_appendix_tables(structured),
        "evidence": _build_evidence(structured),
    }


def _archive_title(markdown: str | None) -> str:
    match = re.search(r"^#\s+(.+)$", markdown or "", flags=re.MULTILINE)
    return (match.group(1).strip() if match else "AI 建模报告")[:255]


def _archive_ai_score(payload: dict[str, Any] | None) -> str | None:
    if not isinstance(payload, dict):
        return None
    for item in payload.get("headline_metrics") or []:
        if isinstance(item, dict) and item.get("key") == "ai_score":
            value = item.get("value")
            return str(value) if value is not None else None
    score = _extract_ai_score(str(payload.get("markdown") or ""))
    return f"{score}/100" if score is not None else None


def _archive_created_at(archive: AIReportArchive) -> str | None:
    created = archive.created_at
    return created.isoformat(timespec="seconds") if created else None


async def archive_ai_report(
    db: AsyncSession,
    report: dict[str, Any],
    owner_username: str | None = None,
) -> dict[str, Any]:
    task_id = report.get("task_id")
    if not task_id:
        raise HTTPException(status_code=400, detail="AI 报告缺少 task_id，无法归档。")
    task_stmt = select(ModelingTask).where(ModelingTask.id == str(task_id))
    if owner_username:
        task_stmt = task_stmt.where(ModelingTask.owner_username == owner_username)
    task = (await db.execute(task_stmt)).scalar_one_or_none()
    if task is None:
        raise HTTPException(status_code=404, detail=f"建模任务 {task_id} 不存在")

    archive = AIReportArchive(
        task_id=str(task_id),
        title=_archive_title(report.get("markdown")),
        model=report.get("model"),
        source=report.get("source") or "doubao",
        markdown=report.get("markdown") or "",
        payload={},
    )
    db.add(archive)
    await db.flush()

    payload = dict(report)
    payload["archive_id"] = archive.id
    payload["archived_at"] = _archive_created_at(archive)
    archive.payload = payload
    await db.flush()
    return payload


async def list_ai_report_archives(
    db: AsyncSession,
    task_id: str,
    owner_username: str | None = None,
) -> list[dict[str, Any]]:
    task_stmt = select(ModelingTask).where(ModelingTask.id == task_id)
    if owner_username:
        task_stmt = task_stmt.where(ModelingTask.owner_username == owner_username)
    task = (await db.execute(task_stmt)).scalar_one_or_none()
    if task is None:
        raise HTTPException(status_code=404, detail=f"建模任务 {task_id} 不存在")
    rows = await db.execute(
        select(AIReportArchive)
        .where(AIReportArchive.task_id == task_id)
        .order_by(AIReportArchive.created_at.desc())
    )
    archives = rows.scalars().all()
    return [
        {
            "id": archive.id,
            "task_id": archive.task_id,
            "title": archive.title,
            "model": archive.model,
            "source": archive.source,
            "generated_at": (archive.payload or {}).get("generated_at") if archive.payload else None,
            "archived_at": _archive_created_at(archive),
            "ai_score": _archive_ai_score(archive.payload),
        }
        for archive in archives
    ]


async def get_ai_report_archive(
    db: AsyncSession,
    task_id: str,
    report_id: str,
    owner_username: str | None = None,
) -> dict[str, Any]:
    task_stmt = select(ModelingTask).where(ModelingTask.id == task_id)
    if owner_username:
        task_stmt = task_stmt.where(ModelingTask.owner_username == owner_username)
    task = (await db.execute(task_stmt)).scalar_one_or_none()
    if task is None:
        raise HTTPException(status_code=404, detail=f"建模任务 {task_id} 不存在")
    row = await db.execute(
        select(AIReportArchive).where(
            AIReportArchive.id == report_id,
            AIReportArchive.task_id == task_id,
        )
    )
    archive = row.scalar_one_or_none()
    if archive is None:
        raise HTTPException(status_code=404, detail=f"AI 报告归档 {report_id} 不存在")

    payload = dict(archive.payload or {})
    payload.setdefault("task_id", archive.task_id)
    payload.setdefault("model", archive.model)
    payload.setdefault("source", archive.source)
    payload.setdefault("markdown", archive.markdown)
    payload.setdefault("report_schema_version", _REPORT_SCHEMA_VERSION)
    payload["archive_id"] = archive.id
    payload["archived_at"] = _archive_created_at(archive)
    payload["title"] = archive.title
    return payload


def _serialize_run(run: ExperimentRun, experiment_index: dict[str, dict[str, Any]]) -> dict[str, Any]:
    exp = experiment_index.get(run.experiment_id, {})
    return {
        "run_id": run.id,
        "experiment_name": exp.get("name"),
        "strategy_type": exp.get("strategy_type") or run.source_experiment_type,
        "trial_no": run.trial_no,
        "status": run.status,
        "rank": run.rank,
        "model_type": _model_name_from_params(run.params),
        "params": _compact_params(run.params),
        "metrics": _compact_metrics(run.metrics),
        "search_meta": _compact_value(run.search_meta or {}),
        "error_message": (run.error_message or "")[:500] or None,
    }


def _serialize_leaderboard_entry(entry: dict[str, Any]) -> dict[str, Any]:
    return {
        "rank": entry.get("rank"),
        "run_id": entry.get("run_id"),
        "experiment_name": entry.get("experiment_name"),
        "strategy_type": entry.get("strategy_type"),
        "trial_no": entry.get("trial_no"),
        "objective_value": entry.get("objective_value"),
        "selection_metric_key": entry.get("selection_metric_key"),
        "selection_value": entry.get("selection_value"),
        "final_test_metric_key": entry.get("final_test_metric_key"),
        "final_test_value": entry.get("final_test_value"),
        "model_type": _model_name_from_params(entry.get("params")),
        "metrics": _compact_metrics(entry.get("metrics")),
        "params": _compact_params(entry.get("params")),
    }


async def build_task_report_context(
    db: AsyncSession,
    task_id: str,
    owner_username: str | None = None,
) -> dict[str, Any]:
    task_stmt = select(ModelingTask).where(ModelingTask.id == task_id)
    if owner_username:
        task_stmt = task_stmt.where(ModelingTask.owner_username == owner_username)
    task = (await db.execute(task_stmt)).scalar_one_or_none()
    if task is None:
        raise HTTPException(status_code=404, detail=f"建模任务 {task_id} 不存在")

    dataset = None
    if task.dataset_id:
        dataset = (
            await db.execute(select(Dataset).where(Dataset.id == task.dataset_id))
        ).scalar_one_or_none()

    exp_rows = await db.execute(
        select(PlatformExperiment)
        .where(PlatformExperiment.modeling_task_id == task_id)
        .order_by(PlatformExperiment.created_at.asc())
    )
    experiments = exp_rows.scalars().all()
    experiment_index = {
        exp.id: {"name": exp.name, "strategy_type": exp.strategy_type}
        for exp in experiments
    }
    exp_ids = [exp.id for exp in experiments]

    status_counts: dict[str, int] = {}
    runs: list[ExperimentRun] = []
    if exp_ids:
        count_rows = await db.execute(
            select(ExperimentRun.status, func.count(ExperimentRun.id))
            .where(ExperimentRun.experiment_id.in_(exp_ids))
            .group_by(ExperimentRun.status)
        )
        status_counts = {str(status): int(count) for status, count in count_rows.all()}

        run_rows = await db.execute(
            select(ExperimentRun)
            .where(ExperimentRun.experiment_id.in_(exp_ids))
            .order_by(ExperimentRun.created_at.desc())
        )
        runs = run_rows.scalars().all()

    leaderboard = await task_leaderboard(
        db,
        task_id,
        top_k=_TOP_RUNS,
        owner_username=owner_username,
    )
    failed_runs = [run for run in runs if run.status == "FAILED"][:3]
    successful_runs = [run for run in runs if run.status == "SUCCESS"][:_TOP_RUNS]

    return {
        "task": {
            "id": task.id,
            "name": task.name,
            "description": task.description,
            "dataset_name": task.dataset_name or (dataset.name if dataset else None),
            "dataset_version_id": task.dataset_version_id,
            "target_column": task.target_column,
            "task_type": task.task_type,
            "objective_metric": task.objective_metric,
            "objective_direction": task.objective_direction,
            "status": task.status,
            "best_run_id": task.best_run_id,
            "summary_snapshot": _compact_value(task.summary_snapshot or {}),
            "final_evaluation": task_final_evaluation_state(task),
        },
        "dataset": {
            "id": dataset.id if dataset else task.dataset_id,
            "name": dataset.name if dataset else task.dataset_name,
            "row_count": dataset.row_count if dataset else None,
            "column_count": dataset.column_count if dataset else None,
            # Structured tables must remain complete. The prompt serializer can
            # compact its own copy, but the report payload must not silently
            # turn a 35-column dataset into a 16-row profile.
            "columns_info": dataset.columns_info or {} if dataset else {},
            "column_names": list((dataset.columns_info or {}).keys()) if dataset else [],
        },
        "experiments": [
            {
                "id": exp.id,
                "name": exp.name,
                "strategy_type": exp.strategy_type,
                "selected_models": _compact_value(exp.selected_models or []),
                "search_space": _compact_value(exp.search_space or {}),
                "budget_config": _compact_value(exp.budget_config or {}),
                "status": exp.status,
                "best_run_id": exp.best_run_id,
            }
            for exp in experiments
        ],
        # Over every run, not the top-k leaderboard: with eighteen runs the
        # deep-learning models rank past the cut, and the cover said
        # "18 个 Run / 3 种模型" for a task that trained five.
        "model_types": sorted({
            str(_model_name_from_params(run.params) or "")
            for run in runs
        } - {""}),
        "run_status_counts": status_counts,
        "leaderboard": [_serialize_leaderboard_entry(entry) for entry in leaderboard],
        "successful_run_examples": [
            _serialize_run(run, experiment_index) for run in successful_runs
        ],
        "failed_run_examples": [
            _serialize_run(run, experiment_index) for run in failed_runs
        ],
        "report_requirements": {
            "language": "简体中文",
            "style": "结论先行，章节按内容出现，指标必须带参照系",
            "reader_goal": "让读者了解任务要做什么、做了什么、训练过程怎么样、效果怎么样",
            "scope": "单 task 研究报告，不是单模型说明书",
            "presentation": "信息密度大的内容用 ECharts 图；信息密度低的内容用表格。只展示数据集概况、参数设置、训练过程数据、模型评价",
            "model_name_rule": "模型名称必须保留原始英文/代码标识，不要翻译或音译",
            "structured_tables": ["数据集概况", "参数设置", "模型评价"],
            "structured_charts": ["训练过程/调参过程指标曲线", "ROC 曲线", "预测结果曲线"],
            "sections": "由内容决定；结论小节必需，其余小节有数据才写",
            "suggestion_scope": "建议只能围绕数据集概况、参数设置、训练过程数据、模型评价展开；不要提出 SHAP、特征重要性、部署监控或其他未展示数据类别",
        },
    }


def build_ai_report_messages(context: dict[str, Any] | str) -> list[dict[str, str]]:
    # Counted here rather than left to the model. Asked how many models were
    # trained, it answered "3" for a task with 7 — the number of experiment
    # batches, which is also in the context and is not the same thing.
    leaderboard = context.get("leaderboard") or [] if isinstance(context, dict) else []
    model_count = len(leaderboard)
    best_model = (leaderboard[0] or {}).get("model_type") if leaderboard else None
    if isinstance(context, str):
        context_text = context[:_MAX_CONTEXT_CHARS]
    else:
        # Reference frames and the readiness score are computed here and handed
        # over as facts. Asking the model to divide one metric by another is
        # asking it to do the thing it is worst at, in a document people will
        # quote.
        enriched = dict(context)
        try:
            enriched["reference_frames"] = build_reference_frames(context)
            enriched["readiness"] = compute_readiness_score(context)
        except Exception as exc:  # noqa: BLE001 — a report without them beats no report
            logger.warning("Reference frame computation failed: %s", exc)
        context_text = json.dumps(
            _context_for_llm(enriched),
            ensure_ascii=False,
            indent=2,
            default=str,
        )[:_MAX_CONTEXT_CHARS]

    system = (
        "你是一位帮工程团队读懂建模结果的分析师。你只依据给出的 JSON 上下文写作；"
        "上下文没有的事实一律写“暂无数据”，绝不推测业务背景、数据来源或未记录的实验。"
        "上下文里的 reference_frames 与 readiness 已由后端算好，可以直接引用，不要自己重算。"
    )

    # A worked exemplar plus a handful of correctness guards, rather than a
    # structure. Two revisions of prescribed shape — first a 三章 skeleton, then
    # a numbered requirement list — both produced padding: sections written to
    # be filled rather than because there was something to say. What the reader
    # wants from the overall report is two things, a verdict and an account of
    # what the data holds, so that is what the example shows.
    user = (
        "请根据下面的建模任务上下文，写一份中文 Markdown 总报告。\n\n"

        "下面是一份**其他任务**的总报告范本。请模仿它的结构、语气、详略和篇幅，"
        "但内容一律以本次上下文为准 —— 范本里的模型名和数字都是别的任务的，一个都不要照抄。\n\n"

        "===== 范本开始 =====\n"
        "## 结论\n\n"
        "这个模型可以进入试用，但还不能直接上生产。本次共训练 6 个模型，"
        "其中 gbdt_regressor 表现最好，在选择集上的 MAE 为 3.42，"
        "约为目标列均值 214.7 的 1.6%，也就是典型情况下预测误差不到两个百分点，"
        "对这个量级的需求预测是够用的。排在第二的 lightgbm_regressor 是 3.91，"
        "两者差距不到 0.5，如果更看重推理速度，选后者也说得过去。\n\n"
        "主要风险是这个成绩尚未在封存测试集上验证过。3.42 来自选择阶段，"
        "而选择阶段的数据参与过模型挑选，天然偏乐观。建议先完成最终评估再决定是否上线。\n\n"
        "## 数据集概况\n\n"
        "本次使用的数据集共 52560 行、28 列，目标列是 demand。"
        "其中十来列是原始采集字段，包括时间戳、气温、湿度和几个站点侧的直接观测量，"
        "它们构成了这份数据的事实基础。\n\n"
        "余下的列是训练流程构造出来的特征，大致分三类："
        "demand_lag_1、demand_lag_24 这样的滞后项，把前一时刻和前一天同一时刻的值带进来；"
        "demand_roll_mean_168 这样的滚动统计量，描述近一周的整体水平；"
        "以及 hour_sin、hour_cos 这类三角变换，把一天之内的周期性显式地交给模型。"
        "这批特征是树模型能取得上述精度的主要原因 —— 原始字段本身并不包含时间上的邻近信息。\n"
        "===== 范本结束 =====\n\n"

        "几点硬要求：\n"
        "- **指标必须带参照系**，不要裸数字。上下文的 reference_frames 里已经算好了换算结果，"
        "直接引用。分类任务要对照多数类基线：准确率 92% 在多数类占 90% 时几乎没有价值\n"
        "- **区分选择分和最终测试分**。选择分不能当作泛化能力的证据；"
        "如果还没做最终评估，必须说明当前结论的可信度受限\n"
        "- 讲数据集时抓构成和用意，说明哪些是原始采集字段、哪些是构造出来的特征，"
        "不要罗列全部列名\n"
        "- 上下文没有的事实一律写“暂无数据”，不要推测业务背景或数据来源\n"
        f"- **本次共训练了 {model_count} 个模型，其中表现最好的是 {best_model or '暂无'}**。"
        "这两项直接用，不要自己数、也不要自己挑；"
        "experiments 里的数量是实验批次数，不是模型数\n"
        "- 模型名一律照抄上下文里的英文标识，不要翻译成中文，也不要写上下文里没有出现过的模型名\n"
        "- 只写这两节，不要写建议、后续工作、给决策人员之类的段落 —— 每个模型的细节"
        "由分报告承担，这里不要逐个展开\n"
        "- 不要输出表格、代码块或图表；不要写一级标题\n\n"

        "## 建模任务上下文 JSON\n\n"
        f"{context_text}"
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def _normalise_markdown(markdown: str) -> str:
    text = (markdown or "").strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    if not text:
        raise HTTPException(status_code=502, detail="豆包返回了空报告。")
    if not text.startswith("#"):
        text = "# AI 建模报告\n\n" + text
    return text.rstrip() + "\n"


def _highlight_report_lead_sentences(markdown: str) -> str:
    """Bold the first sentence of prose paragraphs so the UI can color them."""
    blocks = re.split(r"(\n\s*\n)", markdown or "")
    rendered: list[str] = []
    # An ASCII "." only ends a sentence when it is not a decimal point and
    # something follows it. Treating every "." as a terminator bolded up to the
    # first one, so "RMSE 为 72.47" rendered as "RMSE 为 72.**47" — the bold
    # closing inside the number. The full-width 。！？ are unambiguous.
    sentence_pattern = re.compile(
        r"^(.+?(?:[。！？]|(?<!\d)[.!?](?=\s|$)))(\s*)(.*)$", flags=re.DOTALL
    )
    for block in blocks:
        stripped = block.strip()
        if not stripped:
            rendered.append(block)
            continue
        if stripped.startswith(("#", "|", ">", "```", "- ", "* ", "1. ", "**")):
            rendered.append(block)
            continue
        match = sentence_pattern.match(stripped)
        if not match:
            rendered.append(block)
            continue
        first_sentence = match.group(1).strip()
        rest = match.group(3).strip()
        highlighted = f"**{first_sentence}**" + (f" {rest}" if rest else "")
        prefix = block[: len(block) - len(block.lstrip())]
        suffix = block[len(block.rstrip()):]
        rendered.append(prefix + highlighted + suffix)
    return "".join(rendered).rstrip() + "\n"


def _separate_repeated_bold_leads(markdown: str) -> str:
    """Split adjacent bold conclusion sentences into separate paragraphs."""
    blocks = re.split(r"(\n\s*\n)", markdown or "")
    rendered: list[str] = []
    split_pattern = re.compile(
        r"(?<=[。！？])\s*(?=(\*\*[^*\n]{4,120}?[。！？.!?]\*\*))"
    )
    for block in blocks:
        stripped = block.strip()
        if not stripped or stripped.startswith(("#", "|", ">", "```", "- ", "* ", "1. ")):
            rendered.append(block)
            continue
        rendered.append(split_pattern.sub("\n\n", block))
    return "".join(rendered).rstrip() + "\n"


def _collect_model_identifiers(context: dict[str, Any] | str) -> set[str]:
    if not isinstance(context, dict):
        return set()
    names: set[str] = set()

    def add(value: Any) -> None:
        if isinstance(value, str) and value.strip():
            names.add(value.strip())

    for exp in context.get("experiments") or []:
        for model in exp.get("selected_models") or []:
            add(model)
    for row in (context.get("leaderboard") or []) + (context.get("successful_run_examples") or []):
        add(row.get("model_type"))
        params = row.get("params") or {}
        if isinstance(params, dict):
            add(params.get("model_type"))
            add(params.get("model"))
    return names


def _preserve_model_identifiers(markdown: str, context: dict[str, Any] | str) -> str:
    identifiers = _collect_model_identifiers(context)
    if not identifiers:
        return markdown

    canonical_by_lower = {name.lower(): name for name in identifiers}
    replacements = {
        "arima": ["阿里玛"],
        "random_forest": ["随机森林"],
        "logistic_regression": ["逻辑回归"],
        "svm": ["支持向量机"],
        "xgboost": ["极端梯度提升", "XGBoost 模型"],
        "lightgbm": ["LightGBM 模型"],
        "mlp_dl": ["多层感知机"],
    }
    text = markdown
    for canonical_lower, aliases in replacements.items():
        canonical = canonical_by_lower.get(canonical_lower)
        if not canonical and canonical_lower == "arima":
            canonical = next(
                (name for name in identifiers if name.upper() == "ARIMA"),
                None,
            )
        if not canonical:
            continue
        for alias in aliases:
            text = text.replace(alias, canonical)
    return text


async def _request_chat_completion(settings: Any, messages: list[dict[str, str]]) -> str:
    api_key = _require_api_key(settings)
    base_url = (getattr(settings, "doubao_base_url", "") or "").rstrip("/")
    model = getattr(settings, "doubao_model", "") or "doubao-seed-1-8-251228"
    timeout = float(getattr(settings, "doubao_timeout_s", 30) or 30)
    max_tokens = min(int(getattr(settings, "doubao_max_tokens", 1800) or 1800), 1800)
    url = f"{base_url}/chat/completions"
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.2,
        "max_tokens": max_tokens,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(url, headers=headers, json=payload)
        response.raise_for_status()
        body = response.json()
    except httpx.HTTPStatusError as exc:
        status_code = exc.response.status_code
        logger.warning("Doubao chat completion failed with HTTP %s", status_code)
        hint = "请检查 ARK_API_KEY/DOUBAO_API_KEY、模型名和账号权限。"
        raise HTTPException(
            status_code=502,
            detail=f"豆包 API 请求失败（HTTP {status_code}）。{hint}",
        ) from exc
    except (httpx.RequestError, ValueError) as exc:
        logger.warning("Doubao chat completion request failed: %s", type(exc).__name__)
        raise HTTPException(
            status_code=502,
            detail=f"无法连接或解析豆包 API 响应：{type(exc).__name__}",
        ) from exc

    try:
        content = body["choices"][0]["message"].get("content") or ""
    except (KeyError, IndexError, TypeError) as exc:
        raise HTTPException(status_code=502, detail="豆包 API 响应格式不符合预期。") from exc
    return str(content)


async def generate_ai_report_from_context(
    context: dict[str, Any] | str,
    *,
    task_id: str | None = None,
    settings: Any | None = None,
) -> dict[str, Any]:
    settings = settings or get_settings()
    _require_api_key(settings)
    messages = build_ai_report_messages(context)
    markdown = _normalise_markdown(await _request_chat_completion(settings, messages))
    markdown = _preserve_model_identifiers(markdown, context)
    markdown = _highlight_report_lead_sentences(markdown)
    markdown = _separate_repeated_bold_leads(markdown)
    return {
        "task_id": task_id,
        "model": getattr(settings, "doubao_model", None),
        "source": "doubao",
        "generated_at": _utc_iso(),
        "markdown": markdown,
        **build_rich_report_payload(context, markdown),
    }


async def generate_ai_task_report(
    db: AsyncSession,
    task_id: str,
    *,
    settings: Any | None = None,
    owner_username: str | None = None,
) -> dict[str, Any]:
    settings = settings or get_settings()
    _require_api_key(settings)
    context = await build_task_report_context(
        db,
        task_id,
        owner_username=owner_username,
    )
    narrative = await _generate_narrative(context, task_id, settings)
    report = {
        "task_id": task_id,
        "model": getattr(settings, "doubao_model", None),
        "source": "doubao",
        "generated_at": _utc_iso(),
        # `markdown` stays the overall report so every existing consumer —
        # the archive title, the download button, the legacy reader — keeps
        # working against the same field.
        "markdown": narrative["overview"],
        "run_reports": narrative["runs"],
        "runs_total": narrative["runs_total"],
        "runs_reported": narrative["runs_reported"],
        # Lifted out of the context so the sub-report list can mark the winner
        # without the frontend digging through the report's own input.
        "best_run_id": (context.get("task") or {}).get("best_run_id"),
        **build_rich_report_payload(
            context, narrative["overview"], charts=narrative["overview_charts"],
        ),
    }
    return await archive_ai_report(db, report, owner_username=owner_username)


async def _generate_narrative(
    context: dict[str, Any],
    task_id: str,
    settings: Any,
) -> dict[str, Any]:
    """Overall report then per-run reports, through the narrative module."""
    from app.services import ai_report_narrative

    async def _call(messages: list[dict[str, str]]) -> str:
        return _normalise_markdown(await _request_chat_completion(settings, messages))

    task_type = ((context.get("task") or {}).get("task_type") or "regression")
    result = await ai_report_narrative.generate_narrative_report(
        _enrich_context(context),
        call_model=_call,
        task_type=task_type,
    )
    # Only the overall report is post-processed. Sub-reports are assembled from
    # computed facts and read as written; the lead-sentence bolding welded a
    # section heading to the sentence under it.
    def _post(markdown: str) -> str:
        # No lead-sentence bolding: every paragraph now opens with a computed
        # sentence, so bolding the first one of each bolds most of the report.
        # The headings and tables carry the hierarchy instead.
        return _preserve_model_identifiers(markdown, context)

    result["overview"] = _post(result["overview"])
    for run_report in result["runs"]:  # noqa: B007 — identifiers only
        # Sub-reports get identifier protection only. The lead-sentence
        # bolding is built for the overall report's long unbroken prose; on a
        # sub-report it welds the section heading to the sentence beneath it
        # into one bold run, which is what made these look mangled.
        if run_report.get("markdown"):
            run_report["markdown"] = _preserve_model_identifiers(
                run_report["markdown"], context,
            )
    return result


async def check_doubao_reachability(settings: Any | None = None) -> dict[str, Any]:
    settings = settings or get_settings()
    messages = [
        {"role": "system", "content": "你是连通性测试助手。"},
        {"role": "user", "content": "只回复两个字：可达"},
    ]
    content = await _request_chat_completion(settings, messages)
    return {
        "ok": True,
        "model": getattr(settings, "doubao_model", None),
        "reply": content.strip(),
        "checked_at": _utc_iso(),
    }

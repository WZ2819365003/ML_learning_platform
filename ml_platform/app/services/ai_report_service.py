"""Doubao/Ark-backed AI report generation for modeling tasks."""
from __future__ import annotations

import json
import logging
import math
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

logger = logging.getLogger(__name__)

_TOP_RUNS = 8
_MAX_CONTEXT_CHARS = 12000
_REPORT_SCHEMA_VERSION = "ai_report.rich.v1"


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
}


def _compact_curve_value(value: Any, *, limit: int = 120) -> Any:
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
            compact[key] = _compact_curve_value(metrics[key])
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


def _extract_ai_score(markdown: str) -> int | None:
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


def _top_shap_items(context: dict[str, Any]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for entry in context.get("leaderboard") or []:
        metrics = entry.get("metrics") or {}
        candidates.extend(metrics.get("top_shap_importances") or [])
    for entry in context.get("successful_run_examples") or []:
        metrics = entry.get("metrics") or {}
        candidates.extend(metrics.get("top_shap_importances") or [])
    seen: set[str] = set()
    unique: list[dict[str, Any]] = []
    for item in candidates:
        feature = str(item.get("feature") or "").strip()
        value = _round_number(item.get("mean_abs_shap"))
        if not feature or value is None or feature in seen:
            continue
        seen.add(feature)
        unique.append({"feature": feature, "mean_abs_shap": value})
    return sorted(unique, key=lambda item: abs(item["mean_abs_shap"]), reverse=True)[:8]


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


def _build_headline_metrics(context: dict[str, Any], markdown: str) -> list[dict[str, Any]]:
    task = context.get("task") or {}
    dataset = context.get("dataset") or {}
    counts = context.get("run_status_counts") or {}
    leaderboard = context.get("leaderboard") or []
    final_best = _best_run_level_final(context)
    best = final_best or (leaderboard[0] if leaderboard else {})
    score = _extract_ai_score(markdown)
    final_key, final_value, final_source = _available_final_metric(context)
    best_score = _metric_value_from_entry(best)

    metrics = [
        {
            "key": "ai_score",
            "label": "AI 总分",
            "value": f"{score}/100" if score is not None else "—",
            "detail": "由豆包根据当前任务事实生成",
            "tone": "success" if score is not None and score >= 80 else "warning",
        },
        {
            "key": "best_model",
            "label": "当前最优模型",
            "value": best.get("model_type") or "—",
            "detail": "按最终测试指标判断" if final_best else (best.get("strategy_type") or "按选择阶段榜单判断"),
            "tone": "default",
        },
        {
            "key": "selection_score",
            "label": f"选择阶段 {task.get('objective_metric') or 'score'}",
            "value": _fmt_value(best_score),
            "detail": best.get("selection_metric_key") or "候选模型排序指标",
            "tone": "processing",
        },
        {
            "key": "final_test",
            "label": "最终测试",
            "value": _fmt_value(final_value),
            "detail": (
                f"Run 级 {final_key}"
                if final_source == "run_level" and final_key
                else final_key or "尚未执行最终评估"
            ),
            "tone": "success" if final_value is not None else "warning",
        },
        {
            "key": "run_count",
            "label": "Run 概况",
            "value": str(sum(int(v) for v in counts.values())) if counts else "0",
            "detail": f"成功 {counts.get('SUCCESS', 0)} / 失败 {counts.get('FAILED', 0)}",
            "tone": "default",
        },
        {
            "key": "dataset_size",
            "label": "数据规模",
            "value": _fmt_value(dataset.get("row_count")),
            "detail": f"{_fmt_value(dataset.get('column_count'))} 列 · {dataset.get('name') or '未绑定数据集'}",
            "tone": "default",
        },
    ]
    return metrics


def _leaderboard_chart(context: dict[str, Any]) -> dict[str, Any] | None:
    task = context.get("task") or {}
    entries = []
    for item in context.get("leaderboard") or []:
        value = _metric_value_from_entry(item)
        if value is None:
            continue
        label = item.get("model_type") or item.get("run_id") or "run"
        if item.get("trial_no"):
            label = f"{label}#{item['trial_no']}"
        entries.append({"label": str(label), "value": value})
    if not entries:
        return None
    return {
        "id": "leaderboard_top_runs",
        "title": "候选模型排行榜",
        "description": "按任务目标指标展示 Top Run，指标来自模型选择阶段。",
        "type": "echarts",
        "height": 300,
        "option": {
            "grid": {"left": 56, "right": 18, "top": 28, "bottom": 72},
            "tooltip": {"trigger": "axis"},
            "xAxis": {
                "type": "category",
                "data": [item["label"] for item in entries],
                "axisLabel": {"rotate": 24, "fontSize": 11},
            },
            "yAxis": {"type": "value", "name": task.get("objective_metric") or "score"},
            "series": [
                {
                    "type": "bar",
                    "data": [item["value"] for item in entries],
                    "itemStyle": {"color": "#2563eb", "borderRadius": [4, 4, 0, 0]},
                    "label": {"show": True, "position": "top"},
                }
            ],
        },
    }


def _feature_importance_chart(context: dict[str, Any]) -> dict[str, Any] | None:
    items = _top_shap_items(context)
    if not items:
        return None
    ordered = list(reversed(items))
    return {
        "id": "feature_importance",
        "title": "关键特征重要性",
        "description": "来自 SHAP 或服务端回退解释的平均绝对重要性。",
        "type": "echarts",
        "height": 320,
        "option": {
            "grid": {"left": 132, "right": 22, "top": 24, "bottom": 32},
            "tooltip": {"trigger": "axis"},
            "xAxis": {"type": "value", "name": "mean |SHAP|"},
            "yAxis": {
                "type": "category",
                "data": [item["feature"] for item in ordered],
                "axisLabel": {"fontSize": 11},
            },
            "series": [
                {
                    "type": "bar",
                    "data": [item["mean_abs_shap"] for item in ordered],
                    "itemStyle": {"color": "#10b981", "borderRadius": [0, 4, 4, 0]},
                    "label": {"show": True, "position": "right"},
                }
            ],
        },
    }


def _numeric(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        number = float(value)
    elif isinstance(value, str):
        try:
            number = float(value.strip())
        except ValueError:
            return None
    else:
        return None
    if not math.isfinite(number):
        return None
    return round(number, 6)


def _numeric_sequence(value: Any, *, limit: int = 160) -> list[float]:
    if not isinstance(value, (list, tuple)):
        return []
    numbers: list[float] = []
    for item in list(value)[:limit]:
        number = _numeric(item)
        if number is None:
            return []
        numbers.append(number)
    return numbers


def _history_from_evals_result(evals_result: dict[str, Any]) -> list[dict[str, Any]]:
    series_by_key: dict[str, list[float]] = {}
    for dataset_name, metrics in list(evals_result.items())[:4]:
        if not isinstance(metrics, dict):
            continue
        for metric_name, values in list(metrics.items())[:6]:
            sequence = _numeric_sequence(values, limit=120)
            if sequence:
                series_by_key[f"{dataset_name}_{metric_name}"] = sequence
    if not series_by_key:
        return []
    length = max(len(values) for values in series_by_key.values())
    rows: list[dict[str, Any]] = []
    for index in range(length):
        row: dict[str, Any] = {"epoch": index + 1}
        for key, values in series_by_key.items():
            if index < len(values):
                row[key] = values[index]
        rows.append(row)
    return rows


def _history_points(metrics: dict[str, Any]) -> list[dict[str, Any]]:
    history: Any = None
    for key in ("history", "training_history", "loss_history"):
        if metrics.get(key):
            history = metrics[key]
            break
    if history is None and isinstance(metrics.get("evals_result"), dict):
        history = _history_from_evals_result(metrics["evals_result"])

    if isinstance(history, dict):
        history = _history_from_evals_result(history)

    points: list[dict[str, Any]] = []
    if isinstance(history, (list, tuple)):
        for index, item in enumerate(list(history)[:120], start=1):
            if isinstance(item, dict):
                x_value = (
                    item.get("epoch")
                    or item.get("step")
                    or item.get("iteration")
                    or item.get("round")
                    or index
                )
                point_metrics = {}
                for key, value in item.items():
                    if key in {"epoch", "step", "iteration", "round"}:
                        continue
                    number = _numeric(value)
                    if number is not None:
                        point_metrics[str(key)] = number
                if point_metrics:
                    points.append({"x": _numeric(x_value) or index, "metrics": point_metrics})
            else:
                number = _numeric(item)
                if number is not None:
                    points.append({"x": index, "metrics": {"loss": number}})
    return points


def _history_metric_keys(points: list[dict[str, Any]]) -> list[str]:
    seen: set[str] = set()
    for point in points:
        metrics = point.get("metrics") or {}
        for key in metrics:
            seen.add(str(key))
    preferred = [
        "train_loss",
        "training_loss",
        "val_loss",
        "validation_loss",
        "loss",
        "train_accuracy",
        "validation_accuracy",
        "val_acc",
        "accuracy",
        "val_f1_macro",
        "f1",
        "roc_auc",
        "auc",
    ]
    ordered = [key for key in preferred if key in seen]
    ordered.extend(sorted(seen - set(ordered)))
    return ordered[:3]


def _run_label(row: dict[str, Any]) -> str:
    model = row.get("model_type") or row.get("run_id") or "run"
    trial = row.get("trial_no")
    return f"{model}#{trial}" if trial is not None else str(model)


def _metric_axis_index(metric_key: str) -> int:
    key = metric_key.lower()
    if any(token in key for token in ("loss", "rmse", "mae", "mse", "error")):
        return 0
    return 1


def _strategy_sort_order(strategy_type: Any) -> int:
    order = {"baseline": 0, "grid_search": 1, "bayesian_search": 2}
    return order.get(str(strategy_type or ""), 9)


def _trial_point_label(row: dict[str, Any], index: int) -> str:
    strategy = row.get("strategy_type") or "run"
    trial = row.get("trial_no")
    model = row.get("model_type") or "model"
    if trial is not None:
        return f"{strategy}#{trial}"
    return f"{strategy}-{model}-{index + 1}"


def _run_metric_value(row: dict[str, Any], metric_key: str) -> float | None:
    metrics = row.get("metrics") or {}
    value = None
    if metric_key == row.get("selection_metric_key"):
        value = row.get("selection_value")
        if value is None:
            value = row.get("objective_value")
    elif metric_key == row.get("final_test_metric_key"):
        value = row.get("final_test_value")
    if value is None and isinstance(metrics, dict):
        value = metrics.get(metric_key)
    return _numeric(value)


def _trial_metric_candidates(context: dict[str, Any]) -> list[str]:
    task = context.get("task") or {}
    objective = str(task.get("objective_metric") or "accuracy")
    candidates = [
        f"selection_cv_mean_{objective}",
        f"final_test_{objective}",
        "selection_cv_mean_f1",
        "final_test_f1",
        "selection_cv_mean_roc_auc",
        "final_test_roc_auc",
    ]
    seen: set[str] = set()
    ordered: list[str] = []
    for key in candidates:
        if key not in seen:
            ordered.append(key)
            seen.add(key)
    return ordered


def _build_trial_metric_curve_chart(context: dict[str, Any]) -> dict[str, Any] | None:
    rows = [
        row
        for row in _merge_run_context(context)
        if (row.get("status") in (None, "SUCCESS")) and row.get("run_id")
    ]
    if len(rows) < 2:
        return None
    rows = sorted(
        rows,
        key=lambda row: (
            _strategy_sort_order(row.get("strategy_type")),
            str(row.get("strategy_type") or ""),
            _numeric(row.get("trial_no")) if _numeric(row.get("trial_no")) is not None else 9999,
            str(row.get("model_type") or ""),
            str(row.get("run_id") or ""),
        ),
    )[:_TOP_RUNS]
    labels = [_trial_point_label(row, index) for index, row in enumerate(rows)]

    series: list[dict[str, Any]] = []
    for metric_key in _trial_metric_candidates(context):
        data = []
        numeric_points = 0
        for label, row in zip(labels, rows):
            value = _run_metric_value(row, metric_key)
            if value is not None:
                numeric_points += 1
            data.append([label, value])
        if numeric_points < 2:
            continue
        series.append({
            "name": _metric_label(metric_key),
            "type": "line",
            "smooth": True,
            "connectNulls": True,
            "symbolSize": 6,
            "data": data,
        })
        if len(series) >= 4:
            break
    if not series:
        return None
    return {
        "id": "training_curves",
        "title": "调参过程指标曲线",
        "description": (
            "当前 Run 未记录逐 epoch 的 loss/history，因此图中展示真实 Trial 级选择指标与最终测试指标，"
            "用于观察不同策略和参数组合的过程表现。"
        ),
        "type": "echarts",
        "height": 340,
        "option": {
            "grid": {"left": 52, "right": 24, "top": 38, "bottom": 92},
            "tooltip": {"trigger": "axis"},
            "legend": {"type": "scroll", "bottom": 0},
            "xAxis": {
                "type": "category",
                "name": "Strategy / Trial",
                "data": labels,
                "axisLabel": {"rotate": 25},
            },
            "yAxis": {"type": "value", "name": "metric value"},
            "series": series,
        },
    }


def _build_training_curves_chart(context: dict[str, Any]) -> dict[str, Any] | None:
    series: list[dict[str, Any]] = []
    for row in _merge_run_context(context):
        metrics = row.get("metrics") or {}
        if not isinstance(metrics, dict):
            continue
        points = _history_points(metrics)
        if not points:
            continue
        label = _run_label(row)
        for metric_key in _history_metric_keys(points):
            data = [
                [point["x"], point["metrics"][metric_key]]
                for point in points
                if metric_key in point.get("metrics", {})
            ]
            if len(data) < 2:
                continue
            series.append({
                "name": f"{label} {_metric_label(metric_key)}",
                "type": "line",
                "smooth": True,
                "showSymbol": False,
                "yAxisIndex": _metric_axis_index(metric_key),
                "data": data,
            })
            if len(series) >= 8:
                break
        if len(series) >= 8:
            break
    if not series:
        return _build_trial_metric_curve_chart(context)
    return {
        "id": "training_curves",
        "title": "训练过程曲线",
        "description": "来自 Run 记录的 history/training_history/loss_history/evals_result；没有逐轮记录的模型不会显示在图中。",
        "type": "echarts",
        "height": 340,
        "option": {
            "grid": {"left": 52, "right": 56, "top": 38, "bottom": 78},
            "tooltip": {"trigger": "axis"},
            "legend": {"type": "scroll", "bottom": 0},
            "xAxis": {"type": "value", "name": "epoch/step"},
            "yAxis": [
                {"type": "value", "name": "loss"},
                {"type": "value", "name": "score", "min": 0, "max": 1},
            ],
            "series": series,
        },
    }


def _roc_series_from_metrics(metrics: dict[str, Any]) -> list[list[float]]:
    pairs = [
        ("val_roc_fpr", "val_roc_tpr"),
        ("roc_fpr", "roc_tpr"),
        ("fpr", "tpr"),
    ]
    for fpr_key, tpr_key in pairs:
        fpr = _numeric_sequence(metrics.get(fpr_key), limit=160)
        tpr = _numeric_sequence(metrics.get(tpr_key), limit=160)
        if fpr and tpr and len(fpr) == len(tpr):
            return [[x, y] for x, y in zip(fpr, tpr)]
    return []


def _build_roc_curve_chart(context: dict[str, Any]) -> dict[str, Any] | None:
    series: list[dict[str, Any]] = []
    for row in _merge_run_context(context):
        metrics = row.get("metrics") or {}
        if not isinstance(metrics, dict):
            continue
        data = _roc_series_from_metrics(metrics)
        if len(data) < 2:
            continue
        series.append({
            "name": _run_label(row),
            "type": "line",
            "smooth": True,
            "showSymbol": False,
            "data": data,
        })
        if len(series) >= 4:
            break
    if not series:
        return None
    series.append({
        "name": "random_baseline",
        "type": "line",
        "symbol": "none",
        "lineStyle": {"type": "dashed", "color": "#94a3b8"},
        "data": [[0, 0], [1, 1]],
    })
    return {
        "id": "roc_curve",
        "title": "ROC 曲线",
        "description": "分类任务中用于观察模型区分正负样本的能力；曲线越靠近左上角越好。",
        "type": "echarts",
        "height": 320,
        "option": {
            "grid": {"left": 48, "right": 24, "top": 36, "bottom": 72},
            "tooltip": {"trigger": "axis"},
            "legend": {"type": "scroll", "bottom": 0},
            "xAxis": {"type": "value", "name": "FPR", "min": 0, "max": 1},
            "yAxis": {"type": "value", "name": "TPR", "min": 0, "max": 1},
            "series": series,
        },
    }


def _prediction_pairs_from_metrics(metrics: dict[str, Any]) -> tuple[list[float], list[float]]:
    pairs = [
        ("y_true", "y_pred"),
        ("actual", "predicted"),
        ("actuals", "predictions"),
    ]
    for actual_key, predicted_key in pairs:
        actual = _numeric_sequence(metrics.get(actual_key), limit=200)
        predicted = _numeric_sequence(metrics.get(predicted_key), limit=200)
        if actual and predicted and len(actual) == len(predicted):
            return actual, predicted
    curve = metrics.get("prediction_curve")
    if isinstance(curve, (list, tuple)):
        actual: list[float] = []
        predicted: list[float] = []
        for item in list(curve)[:200]:
            if not isinstance(item, dict):
                continue
            actual_value = _numeric(item.get("actual") or item.get("y_true"))
            predicted_value = _numeric(item.get("predicted") or item.get("y_pred"))
            if actual_value is None or predicted_value is None:
                continue
            actual.append(actual_value)
            predicted.append(predicted_value)
        if actual and len(actual) == len(predicted):
            return actual, predicted
    return [], []


def _build_prediction_curve_chart(context: dict[str, Any]) -> dict[str, Any] | None:
    for row in _merge_run_context(context):
        metrics = row.get("metrics") or {}
        if not isinstance(metrics, dict):
            continue
        actual, predicted = _prediction_pairs_from_metrics(metrics)
        if len(actual) < 2:
            continue
        x_values = list(range(1, len(actual) + 1))
        label = _run_label(row)
        return {
            "id": "prediction_curve",
            "title": "预测结果曲线",
            "description": f"展示 {label} 的真实值与预测值随样本序号的变化，用于观察误差是否集中在局部样本。",
            "type": "echarts",
            "height": 320,
            "option": {
                "grid": {"left": 52, "right": 24, "top": 36, "bottom": 62},
                "tooltip": {"trigger": "axis"},
                "legend": {"bottom": 0},
                "xAxis": {"type": "category", "name": "样本序号", "data": x_values},
                "yAxis": {"type": "value", "name": "value"},
                "series": [
                    {
                        "name": "actual",
                        "type": "line",
                        "showSymbol": False,
                        "data": actual,
                    },
                    {
                        "name": "predicted",
                        "type": "line",
                        "showSymbol": False,
                        "data": predicted,
                    },
                ],
            },
        }
    return None


def _run_status_chart(context: dict[str, Any]) -> dict[str, Any] | None:
    counts = {
        key: int(value)
        for key, value in (context.get("run_status_counts") or {}).items()
        if int(value or 0) > 0
    }
    if not counts:
        return None
    color_by_status = {
        "SUCCESS": "#10b981",
        "FAILED": "#ef4444",
        "RUNNING": "#2563eb",
        "PENDING": "#f59e0b",
        "QUEUED": "#8b5cf6",
    }
    return {
        "id": "run_status_distribution",
        "title": "Run 状态分布",
        "description": "用于快速判断实验是否完成、失败是否集中。",
        "type": "echarts",
        "height": 280,
        "option": {
            "tooltip": {"trigger": "item"},
            "legend": {"bottom": 0},
            "series": [
                {
                    "type": "pie",
                    "radius": ["42%", "68%"],
                    "center": ["50%", "44%"],
                    "data": [
                        {
                            "name": key,
                            "value": value,
                            "itemStyle": {"color": color_by_status.get(key, "#64748b")},
                        }
                        for key, value in counts.items()
                    ],
                    "label": {"formatter": "{b}: {c}"},
                }
            ],
        },
    }


def _build_charts(context: dict[str, Any]) -> list[dict[str, Any]]:
    charts = [
        _build_training_curves_chart(context),
        _build_roc_curve_chart(context),
        _build_prediction_curve_chart(context),
    ]
    return [chart for chart in charts if chart is not None]


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


def _first_metric(metrics: dict[str, Any], candidates: list[str]) -> tuple[str | None, Any]:
    if not isinstance(metrics, dict):
        return None, None
    for key in candidates:
        if key in metrics:
            return key, metrics[key]
    return None, None


def _metric_text(key: str | None, value: Any) -> str:
    if key is None and value is None:
        return "—"
    if key is None:
        return _fmt_value(value)
    label = _metric_label(key)
    if label and label != key:
        return f"{label}（{key}）={_fmt_value(value)}"
    return f"{key}={_fmt_value(value)}"


def _metric_lookup(metrics: dict[str, Any], *keys: str) -> Any:
    if not isinstance(metrics, dict):
        return None
    for key in keys:
        if key in metrics:
            return metrics[key]
    return None


def _metric_label(metric: str | None) -> str:
    key = (metric or "").lower()
    labels = {
        "accuracy": "准确率",
        "final_test_accuracy": "最终测试准确率",
        "selection_cv_mean_accuracy": "交叉验证平均准确率",
        "cv_avg_accuracy": "交叉验证平均准确率",
        "validation_accuracy": "验证准确率",
        "train_accuracy": "训练准确率",
        "test_accuracy": "测试准确率",
        "val_acc": "验证准确率",
        "f1": "F1 分数",
        "final_test_f1": "最终测试 F1",
        "selection_cv_mean_f1": "交叉验证平均 F1",
        "roc_auc": "ROC-AUC",
        "final_test_roc_auc": "最终测试 ROC-AUC",
        "selection_cv_mean_roc_auc": "交叉验证平均 ROC-AUC",
        "auc": "AUC",
        "rmse": "RMSE",
        "mae": "MAE",
        "mse": "MSE",
        "r2": "R2",
        "val_loss": "验证损失",
        "loss": "损失",
    }
    return labels.get(key, metric or "指标")


def _metric_direction_text(metric: str | None, objective_direction: str | None = None) -> str:
    key = (metric or "").lower()
    if objective_direction and key in ("objective", "score"):
        return "越大越好" if objective_direction == "max" else "越小越好"
    if any(token in key for token in ("loss", "rmse", "mae", "mse", "error")):
        return "越小越好"
    return "越大越好"


def _metric_usage_text(metric: str | None) -> str:
    key = (metric or "").lower()
    if "cv" in key or "selection" in key or "val" in key:
        return "选择/验证阶段"
    if "final" in key or key in {"accuracy", "f1", "roc_auc", "auc", "rmse", "mae", "mse", "r2"}:
        return "最终效果"
    return "辅助判断"


def _strip_markdown_tables(markdown: str | None) -> str:
    """Remove AI-generated Markdown tables; structured tables render separately."""
    if not markdown:
        return ""
    cleaned: list[str] = []
    previous_blank = False
    for raw_line in markdown.splitlines():
        line = raw_line.strip()
        if line.startswith("|") and line.endswith("|") and "|" in line[1:-1]:
            if not previous_blank:
                cleaned.append("")
                previous_blank = True
            continue
        cleaned.append(raw_line)
        previous_blank = line == ""
    text = "\n".join(cleaned)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text + ("\n" if text else "")


def _effect_note(selection_value: Any, final_value: Any, direction: str | None) -> str:
    selection = _round_number(selection_value)
    final = _round_number(final_value)
    if selection is None and final is None:
        return "暂无可对比的效果指标"
    if final is None:
        return "尚未记录最终测试，先按选择阶段指标理解相对表现"
    if selection is None:
        return "已有最终测试，可作为当前泛化效果依据"
    gap = final - selection
    if (direction or "max") == "min":
        gap = selection - final
    if abs(gap) <= 0.03:
        return "选择阶段与最终测试接近，稳定性初步可接受"
    if gap < 0:
        return "最终测试弱于选择阶段，需要关注过拟合或数据切分差异"
    return "最终测试优于选择阶段，建议复核数据切分并继续做稳定性验证"


def _build_metric_comparison_table(context: dict[str, Any]) -> dict[str, Any] | None:
    task = context.get("task") or {}
    objective = task.get("objective_metric") or "score"
    direction = task.get("objective_direction") or "max"
    rows = []
    for item in context.get("leaderboard") or []:
        metrics = item.get("metrics") or {}
        selection_value = item.get("selection_value") or item.get("objective_value")
        final_value = item.get("final_test_value")
        test_accuracy = final_value if (item.get("final_test_metric_key") or "").endswith("accuracy") else None
        if test_accuracy is None:
            test_accuracy = _metric_lookup(metrics, "final_test_accuracy", "accuracy", "test_accuracy", "val_acc")
        rows.append({
            "rank": item.get("rank"),
            "model_type": item.get("model_type") or "—",
            "strategy_type": item.get("strategy_type") or "—",
            "trial_no": _fmt_value(item.get("trial_no")),
            "selection_metric": _metric_text(item.get("selection_metric_key"), selection_value),
            "test_accuracy": _fmt_value(test_accuracy),
            "test_f1": _fmt_value(_metric_lookup(metrics, "final_test_f1", "f1", "test_f1", "val_f1_macro")),
            "test_roc_auc": _fmt_value(_metric_lookup(metrics, "final_test_roc_auc", "roc_auc", "auc", "val_auc_roc")),
            "test_rmse": _fmt_value(_metric_lookup(metrics, "final_test_rmse", "rmse", "test_rmse")),
            "test_mae": _fmt_value(_metric_lookup(metrics, "final_test_mae", "mae", "test_mae")),
            "effect_note": _effect_note(selection_value, final_value, direction),
            "run_id": item.get("run_id"),
        })
    if not rows:
        return None
    columns = [
        {"key": "rank", "title": "排名"},
        {"key": "model_type", "title": "模型"},
        {"key": "strategy_type", "title": "策略"},
        {"key": "trial_no", "title": "Trial"},
        {"key": "selection_metric", "title": "选择/验证指标"},
        {"key": "test_accuracy", "title": "测试准确率"},
        {"key": "test_f1", "title": "测试 F1"},
        {"key": "test_roc_auc", "title": "测试 ROC-AUC"},
        {"key": "test_rmse", "title": "测试 RMSE"},
        {"key": "test_mae", "title": "测试 MAE"},
        {"key": "effect_note", "title": "评价结论"},
        {"key": "run_id", "title": "Run ID"},
    ]
    optional_metric_keys = {"test_accuracy", "test_f1", "test_roc_auc", "test_rmse", "test_mae"}
    columns = [
        column
        for column in columns
        if column["key"] not in optional_metric_keys
        or any(row.get(column["key"]) not in (None, "", "—") for row in rows)
    ]
    return {
        "id": "metric_comparison",
        "title": "模型评价（准确率相关）",
        "columns": columns,
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


def _training_setup_text(row: dict[str, Any]) -> str:
    params = row.get("params") or {}
    search_meta = row.get("search_meta") or {}
    parts = []
    if row.get("strategy_type"):
        parts.append(f"策略={row['strategy_type']}")
    if row.get("trial_no") is not None:
        parts.append(f"Trial={row['trial_no']}")
    if isinstance(params, dict):
        if params.get("cv_folds") is not None:
            parts.append(f"交叉验证={params['cv_folds']} 折")
        if params.get("test_size") is not None:
            parts.append(f"测试集比例={params['test_size']}")
        if params.get("random_state") is not None:
            parts.append(f"随机种子={params['random_state']}")
    if isinstance(search_meta, dict):
        mode = search_meta.get("evaluation_mode") or search_meta.get("search_mode")
        if mode:
            parts.append(f"评估方式={mode}")
    return "；".join(str(part) for part in parts) or "训练设置未完整记录"


def _process_summary_text(row: dict[str, Any]) -> str:
    metrics = row.get("metrics") or {}
    history = None
    if isinstance(metrics, dict):
        for key in ("training_history", "history", "loss_history", "evals_result"):
            value = metrics.get(key)
            if value:
                history = value
                break
    if isinstance(history, list):
        return f"记录了 {len(history)} 个训练过程点，可继续查看收敛或波动情况"
    if isinstance(history, dict):
        return f"记录了 {len(history)} 组训练过程指标，可用于复核收敛轨迹"
    return "未记录逐轮曲线，当前依据训练/验证/测试汇总指标判断过程质量"


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


def _run_effect_text(row: dict[str, Any], objective: str, direction: str) -> str:
    metrics = row.get("metrics") or {}
    selection_value = row.get("selection_value") or row.get("objective_value")
    final_value = row.get("final_test_value")
    if selection_value is None and isinstance(metrics, dict):
        _, selection_value = _first_metric(
            metrics,
            [f"selection_cv_mean_{objective}", f"validation_{objective}", f"val_{objective}"],
        )
    if final_value is None and isinstance(metrics, dict):
        _, final_value = _first_metric(metrics, [f"final_test_{objective}", f"test_{objective}"])
    return _effect_note(selection_value, final_value, direction)


def _build_model_training_process_table(context: dict[str, Any]) -> dict[str, Any] | None:
    task = context.get("task") or {}
    objective = task.get("objective_metric") or "score"
    direction = task.get("objective_direction") or "max"
    rows = []
    for row in _merge_run_context(context):
        rows.append({
            "model_type": row.get("model_type") or "—",
            "strategy_type": row.get("strategy_type") or "—",
            "trial_no": _fmt_value(row.get("trial_no")),
            "status": row.get("status") or "SUCCESS",
            "training_setup": _training_setup_text(row),
            "key_params": _key_params_text(row),
            "process_summary": _process_summary_text(row),
            "effect_summary": _run_effect_text(row, objective, direction),
            "run_id": row.get("run_id"),
        })
    if not rows:
        return None
    return {
        "id": "model_training_process",
        "title": "模型训练过程",
        "columns": [
            {"key": "model_type", "title": "模型"},
            {"key": "strategy_type", "title": "策略"},
            {"key": "trial_no", "title": "Trial"},
            {"key": "status", "title": "状态"},
            {"key": "training_setup", "title": "训练设置"},
            {"key": "key_params", "title": "关键参数"},
            {"key": "process_summary", "title": "过程记录"},
            {"key": "effect_summary", "title": "过程解读"},
            {"key": "run_id", "title": "Run ID"},
        ],
        "rows": rows,
    }


def _best_run(rows: list[dict[str, Any]], direction: str) -> dict[str, Any]:
    def score(row: dict[str, Any]) -> float:
        value = _round_number(row.get("final_test_value"))
        if value is None:
            value = _round_number(row.get("selection_value") or row.get("objective_value"))
        if value is None:
            return float("-inf") if direction == "max" else float("inf")
        return value

    return sorted(rows, key=score, reverse=direction == "max")[0]


def _group_risk_text(rows: list[dict[str, Any]], objective: str, direction: str) -> str:
    gaps: list[float] = []
    finals: list[float] = []
    for row in rows:
        selection = _round_number(row.get("selection_value") or row.get("objective_value"))
        final = _round_number(row.get("final_test_value"))
        if final is not None:
            finals.append(final)
        if selection is not None and final is not None:
            gap = final - selection if direction == "max" else selection - final
            gaps.append(gap)
    if not finals:
        return "缺少最终测试指标，当前只能看选择阶段表现"
    if gaps and min(gaps) < -0.05:
        return f"存在 {objective} 测试表现明显低于选择阶段的 Run，需关注过拟合或切分差异"
    if len(finals) >= 2 and max(finals) - min(finals) > 0.05:
        return "同组 Run 最终效果波动较大，调参结论需要更多试验支撑"
    return "选择阶段与最终测试整体接近，稳定性初步可接受"


def _search_profile_text(rows: list[dict[str, Any]]) -> str:
    strategies = {str(row.get("strategy_type") or "—") for row in rows}
    trials = [row.get("trial_no") for row in rows if row.get("trial_no") is not None]
    strategy_text = "、".join(sorted(strategies))
    if trials:
        return f"{strategy_text}；覆盖 Trial {min(trials)}-{max(trials)}"
    return strategy_text


def _build_model_training_summary_table(context: dict[str, Any]) -> dict[str, Any] | None:
    task = context.get("task") or {}
    objective = task.get("objective_metric") or "score"
    direction = task.get("objective_direction") or "max"
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in _merge_run_context(context):
        key = (str(row.get("model_type") or "—"), str(row.get("strategy_type") or "—"))
        grouped.setdefault(key, []).append(row)

    rows = []
    for (model_type, strategy_type), group_rows in grouped.items():
        best = _best_run(group_rows, direction)
        rows.append({
            "model_type": model_type,
            "strategy_type": strategy_type,
            "trial_count": str(len(group_rows)),
            "search_profile": _search_profile_text(group_rows),
            "best_selection_metric": _metric_text(best.get("selection_metric_key"), best.get("selection_value") or best.get("objective_value")),
            "best_final_metric": _metric_text(best.get("final_test_metric_key"), best.get("final_test_value")),
            "key_params": _key_params_text(best),
            "stability_risk": _group_risk_text(group_rows, objective, direction),
        })
    if not rows:
        return None
    return {
        "id": "model_training_summary",
        "title": "模型训练摘要",
        "columns": [
            {"key": "model_type", "title": "模型"},
            {"key": "strategy_type", "title": "策略"},
            {"key": "trial_count", "title": "Trial 数"},
            {"key": "search_profile", "title": "训练/搜索方式"},
            {"key": "best_selection_metric", "title": "最佳选择指标"},
            {"key": "best_final_metric", "title": "最佳测试指标"},
            {"key": "key_params", "title": "代表参数"},
            {"key": "stability_risk", "title": "稳定性与风险"},
        ],
        "rows": rows,
    }


def _metric_glossary_candidates(context: dict[str, Any]) -> list[str]:
    task = context.get("task") or {}
    seen: list[str] = []

    def add(metric: str | None) -> None:
        if metric and metric not in seen:
            seen.append(metric)

    add(task.get("objective_metric"))
    for row in context.get("leaderboard") or []:
        add(row.get("selection_metric_key"))
        add(row.get("final_test_metric_key"))
        metrics = row.get("metrics") or {}
        if isinstance(metrics, dict):
            for key in ("accuracy", "f1", "roc_auc", "auc", "rmse", "mae", "mse", "r2", "val_loss"):
                if key in metrics:
                    add(key)
    return seen[:10]


def _metric_meaning(metric: str) -> str:
    key = metric.lower()
    if "accuracy" in key:
        return "预测正确的样本占比，适合类别较均衡的分类任务。"
    if key in {"f1", "final_test_f1"} or key.endswith("_f1"):
        return "精确率和召回率的调和平均，适合同时关注误报和漏报。"
    if "roc_auc" in key or key == "auc":
        return "模型区分正负样本的能力，越接近 1 区分能力越强。"
    if "rmse" in key:
        return "预测误差平方平均后开根号，对大误差更敏感。"
    if "mae" in key:
        return "预测值与真实值的平均绝对误差，更直观反映平均偏差。"
    if "r2" in key:
        return "模型解释目标波动的比例，通常越接近 1 越好。"
    if "loss" in key:
        return "训练或验证损失，用于观察收敛和过拟合。"
    return "平台记录的模型评估指标，需要结合任务类型和优化方向解读。"


def _build_metric_glossary_table(context: dict[str, Any]) -> dict[str, Any] | None:
    task = context.get("task") or {}
    rows = []
    for metric in _metric_glossary_candidates(context):
        rows.append({
            "metric": metric,
            "label": _metric_label(metric),
            "stage": _metric_usage_text(metric),
            "direction": _metric_direction_text(metric, task.get("objective_direction")),
            "meaning": _metric_meaning(metric),
        })
    if not rows:
        return None
    return {
        "id": "metric_glossary",
        "title": "指标读法说明",
        "columns": [
            {"key": "metric", "title": "指标"},
            {"key": "label", "title": "中文名"},
            {"key": "stage", "title": "使用阶段"},
            {"key": "direction", "title": "方向"},
            {"key": "meaning", "title": "怎么读"},
        ],
        "rows": rows,
    }


def _build_recommendation_plan_table(context: dict[str, Any]) -> dict[str, Any] | None:
    task = context.get("task") or {}
    dataset = context.get("dataset") or {}
    counts = context.get("run_status_counts") or {}
    columns = _iter_column_info(dataset.get("columns_info"))
    missing_columns = [
        name
        for name, info in columns
        if (info.get("missing_count") or 0) or (info.get("missing_rate") or 0)
    ]
    final_key, final_value, final_source = _available_final_metric(context)
    rows: list[dict[str, Any]] = []

    if missing_columns:
        rows.append({
            "priority": "P0",
            "area": "数据质量",
            "evidence": f"{len(missing_columns)} 个字段存在缺失：{'、'.join(missing_columns[:5])}",
            "action": "补齐缺失处理策略，区分数值填充、类别填充和是否删除异常样本。",
            "expected_benefit": "减少模型把缺失模式误当作有效信号的风险。",
        })
    else:
        rows.append({
            "priority": "P1",
            "area": "数据理解",
            "evidence": f"字段画像显示 {len(columns)} 个字段，缺失风险不突出。",
            "action": "补充字段业务含义、取值范围和异常值统计，完善数据概况解释。",
            "expected_benefit": "让报告从“列名解释”升级为可审计的数据质量说明。",
        })

    if final_source == "run_level":
        rows.append({
            "priority": "P0",
            "area": "最终评估",
            "evidence": f"已有 Run 级 `{final_key}`={_fmt_value(final_value)}，但任务级 final_evaluation 尚未固化。",
            "action": "执行或确认最终评估，把胜出 Run、测试集指标和评估版本写入任务级状态。",
            "expected_benefit": "避免报告结论依赖临时榜单，提高上线判断可信度。",
        })
    elif final_value is None:
        rows.append({
            "priority": "P0",
            "area": "最终评估",
            "evidence": "当前没有可用最终测试指标。",
            "action": "先完成独立测试集评估，再生成面向决策的报告。",
            "expected_benefit": "避免只凭选择阶段分数判断泛化效果。",
        })

    rows.append({
        "priority": "P1",
        "area": "训练验证",
        "evidence": f"当前成功 Run {counts.get('SUCCESS', 0)} 个，目标指标 `{task.get('objective_metric') or '—'}`。",
        "action": "比较各模型选择指标与测试指标差距；差距大的模型优先增加交叉验证折数或重复切分验证。",
        "expected_benefit": "识别过拟合和数据切分偶然性，提升模型选择稳定性。",
    })

    if _top_shap_items(context):
        top_features = "、".join(item["feature"] for item in _top_shap_items(context)[:3])
        rows.append({
            "priority": "P2",
            "area": "特征解释",
            "evidence": f"当前最重要特征集中在 {top_features}。",
            "action": "围绕高影响特征做异常值检查、业务合理性复核和必要的衍生特征。",
            "expected_benefit": "提升可解释性，并降低模型依赖错误字段的风险。",
        })

    return {
        "id": "recommendation_plan",
        "title": "建议行动计划",
        "columns": [
            {"key": "priority", "title": "优先级"},
            {"key": "area", "title": "方向"},
            {"key": "evidence", "title": "依据"},
            {"key": "action", "title": "动作"},
            {"key": "expected_benefit", "title": "预期收益"},
        ],
        "rows": rows,
    }


def _build_tables(context: dict[str, Any]) -> list[dict[str, Any]]:
    tables: list[dict[str, Any]] = []
    for table in (
        _build_data_profile_table(context),
        _build_parameter_settings_table(context),
        _build_metric_comparison_table(context),
    ):
        if table is not None:
            tables.append(table)
    return tables


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


def _section_from_markdown(markdown: str, heading: str) -> str | None:
    pattern = re.compile(
        rf"(^##\s+{re.escape(heading)}[\s\S]*?)(?=^##\s+|\Z)",
        flags=re.MULTILINE,
    )
    match = pattern.search(markdown or "")
    if not match:
        return None
    return match.group(1).strip() + "\n"


def _section_from_markdown_any(markdown: str, headings: list[str]) -> str | None:
    for heading in headings:
        section = _section_from_markdown(markdown, heading)
        if section:
            return section
    return None


def _intro_title(markdown: str) -> str:
    first_heading = re.search(r"^#\s+(.+)$", markdown or "", flags=re.MULTILINE)
    return f"# {first_heading.group(1).strip()}\n" if first_heading else "# AI 建模报告\n"


def _task_scope_markdown(context: dict[str, Any]) -> str:
    task = context.get("task") or {}
    dataset = context.get("dataset") or {}
    experiments = context.get("experiments") or []
    counts = context.get("run_status_counts") or {}
    total_runs = sum(int(value or 0) for value in counts.values())
    success_runs = int(counts.get("SUCCESS", 0) or 0)
    model_names: list[str] = []
    strategy_names: list[str] = []
    for exp in experiments:
        strategy = str(exp.get("strategy_type") or "—")
        if strategy not in strategy_names:
            strategy_names.append(strategy)
        for model in exp.get("selected_models") or []:
            model_text = str(model)
            if model_text not in model_names:
                model_names.append(model_text)
    output_text = "分类标签与类别概率" if task.get("task_type") == "classification" else "连续数值预测"

    return (
        "### 1.2 任务范围\n\n"
        "#### 1.2.1 入参与出参\n\n"
        f"**本任务的研究边界是清晰的单 task 建模评估。** 本次建模使用数据集 "
        f"`{dataset.get('name') or '—'}`，目标列是 `{task.get('target_column') or '—'}`，"
        f"任务类型为 `{task.get('task_type') or '—'}`，主要评价指标为 "
        f"`{task.get('objective_metric') or '—'}`。业务出参是{output_text}，平台出参是参数记录、"
        "训练过程数据、模型评价结果和本报告归档。\n\n"
        f"**本任务的完成情况应从训练批次、Run 结果和结构化证据共同判断。** 本次共执行 "
        f"{len(experiments)} 个训练配置批次、{total_runs} 个 Run，其中成功 {success_runs} 个。"
        f"参与模型包括 {'、'.join(model_names[:8]) or '暂无记录'}，训练策略包括 "
        f"{'、'.join(strategy_names) or '暂无记录'}；下文只展开四类核心证据：数据集概况、参数设置、"
        "训练过程数据和模型评价。"
    )


def _input_output_markdown(context: dict[str, Any]) -> str:
    task = context.get("task") or {}
    dataset = context.get("dataset") or {}
    experiments = context.get("experiments") or []
    selected_models: list[str] = []
    strategies: list[str] = []
    for exp in experiments:
        strategy = str(exp.get("strategy_type") or "—")
        if strategy not in strategies:
            strategies.append(strategy)
        for model in exp.get("selected_models") or []:
            model_text = str(model)
            if model_text not in selected_models:
                selected_models.append(model_text)
    model_text = "、".join(selected_models[:8]) or "暂无记录"
    strategy_text = "、".join(strategies) or "暂无记录"
    output_text = "分类标签与类别概率" if task.get("task_type") == "classification" else "连续数值预测"

    return (
        "## 输入与输出说明\n\n"
        f"本次建模的入参包括数据集 `{dataset.get('name') or '—'}`、目标列 "
        f"`{task.get('target_column') or '—'}`、任务类型 `{task.get('task_type') or '—'}`，"
        f"以及候选模型 `{model_text}` 和训练策略 `{strategy_text}`。这些输入决定了平台如何切分数据、"
        "训练候选模型、计算选择阶段指标，并最终把结果汇总给报告生成器。\n\n"
        f"本任务的业务出参是{output_text}；平台出参包括训练过程曲线、预测结果曲线、"
        "效果指标表、关键特征解释、实验状态和这份 AI 报告归档。阅读时要把“预测输出”和"
        "“评估输出”分开：前者服务于模型使用，后者服务于判断模型是否可靠。"
    )


def _data_profile_markdown(context: dict[str, Any]) -> str:
    task = context.get("task") or {}
    dataset = context.get("dataset") or {}
    columns_info = _iter_column_info(dataset.get("columns_info"))
    target_column = task.get("target_column") or "—"
    feature_count = max(len(columns_info) - (1 if target_column != "—" else 0), 0)
    if not columns_info:
        field_sentence = "当前数据集没有保存字段画像，因此只能读取样本量和列数，无法逐字段解释。"
    else:
        field_names = "、".join(
            _readable_field_name(
                column,
                role=_column_role(column, info, task.get("target_column")),
            )
            for column, info in columns_info[:8]
        )
        suffix = "等字段" if len(columns_info) > 8 else "这些字段"
        target_label = _readable_field_name(str(target_column), role="目标列")
        field_sentence = f"字段画像显示，本表覆盖 {field_names}{suffix}，其中 {target_label} 是目标列。"

    return (
        "### 2.1 数据集概况\n\n"
        "#### 2.1.1 数据输入结论\n\n"
        f"**本任务的数据输入已经具备基本可解释边界。** 数据集 `{dataset.get('name') or '—'}` "
        f"当前记录了 {_fmt_value(dataset.get('row_count'))} 行、{_fmt_value(dataset.get('column_count'))} 列。"
        f"{field_sentence}除目标列外，大约有 {feature_count} 个可用输入字段需要进入后续特征处理和模型训练。\n\n"
        "**字段概况表用于说明模型实际接收了哪些输入。** 下表把每个字段的角色、类型、缺失情况和唯一值数量列出；"
        "它的作用不是替代数据质量审计，而是帮助读者判断目标列是否明确、输入字段是否完整、哪些字段可能需要补缺失或编码。"
    )


def _training_process_markdown(context: dict[str, Any]) -> str:
    counts = context.get("run_status_counts") or {}
    successful = int(counts.get("SUCCESS", 0) or 0)
    failed = int(counts.get("FAILED", 0) or 0)
    return (
        "### 2.3 训练过程\n\n"
        "#### 2.3.1 过程数据结论\n\n"
        f"**训练过程的重点是指标轨迹，而不是运行成功率可视化。** 本任务当前成功 Run 为 {successful} 个，"
        f"失败 Run 为 {failed} 个；这个信息只用于判断实验是否完整，不作为主要图表内容。\n\n"
        "**训练曲线用于判断过程是否支撑最终结论。** 下方 ECharts 图优先展示 Run 记录中的 "
        "`history`、`training_history`、`loss_history` 或 `evals_result`；如果当前训练器没有保存逐 epoch 曲线，"
        "则展示真实 Trial 级选择指标与最终测试指标。读图时重点看三件事：过程指标是否随参数变化稳定、"
        "验证指标是否与测试指标脱节、不同策略是否持续带来效果提升。"
    )


def _effect_summary_markdown(context: dict[str, Any]) -> str:
    task = context.get("task") or {}
    final_key, final_value, final_source = _available_final_metric(context)
    best_final = _best_run_level_final(context)
    best_model = (best_final or {}).get("model_type")
    best_key = (best_final or {}).get("final_test_metric_key") or final_key
    best_value = (best_final or {}).get("final_test_value")
    if best_value is None:
        best_value = final_value
    best_sentence = ""
    if best_model and best_value is not None:
        best_sentence = (
            f"当前表现最好的 {best_model} {_metric_label(best_key)}为 {_fmt_value(best_value)}，"
            "这个数字比原始指标 key 更适合作为读者理解模型效果的主句。"
        )
    if final_value is None:
        final_sentence = "当前尚未记录最终测试指标，因此只能先把选择/验证阶段指标当作相对比较依据。"
    elif final_source == "run_level":
        final_sentence = (
            f"当前已有 Run 级最终测试指标：{_metric_text(final_key, final_value)}，"
            "可以用于比较模型泛化效果；但任务级 final_evaluation 尚未固化，正式结论仍需完成最终评估确认。"
        )
    else:
        final_sentence = (
            f"当前已有{_metric_text(final_key, final_value)}，可以作为模型泛化效果的主要证据。"
        )
    return (
        "### 2.4 模型评价\n\n"
        "#### 2.4.1 评价结论\n\n"
        f"**模型评价应优先用表格横向比较。** `{task.get('objective_metric') or '—'}`、F1、ROC-AUC "
        "这类低密度指标更适合放在同一张表中对照不同模型、策略和 Trial，而不是重复绘制成柱状图。"
        f"{best_sentence or final_sentence}{final_sentence if best_sentence else ''}\n\n"
        "**评价表的阅读顺序是先看最终测试，再看选择阶段与最终测试的差距。** 下表把选择/验证指标和最终测试指标放在同一行；"
        "如果差距过大，即使选择阶段排名靠前，也不能直接作为本 task 的推荐结论。"
    )


def _param_snippet(params: dict[str, Any] | None) -> str:
    compact = _compact_params(params)
    if not isinstance(compact, dict) or not compact:
        return "未记录关键超参数"
    parts = []
    for key, value in list(compact.items())[:6]:
        rendered = json.dumps(value, ensure_ascii=False) if isinstance(value, dict) else _fmt_value(value)
        parts.append(f"`{key}`={rendered}")
    return "，".join(parts)


def _parameter_markdown(context: dict[str, Any]) -> str:
    leaderboard = context.get("leaderboard") or []
    if not leaderboard:
        return (
            "### 2.2 参数设置\n\n"
            "#### 2.2.1 训练配置结论\n\n"
            "**当前参数证据不足以解释训练配置。** 当前没有成功 Run 的参数记录，无法解释本 task 的训练配置；"
            "建议先完成至少一个成功训练，再查看参数与指标之间的关系。"
        )
    best = _best_run_level_final(context) or leaderboard[0]
    model = best.get("model_type") or "未知模型"
    strategy = best.get("strategy_type") or "未知策略"
    params = _param_snippet(best.get("params"))
    return (
        "### 2.2 参数设置\n\n"
        "#### 2.2.1 训练配置结论\n\n"
        f"**参数设置表记录的是本 task 每个 Run 的训练配置证据。** 当前最终测试口径下最值得关注的是 "
        f"`{model}` / `{strategy}`，关键参数为 {params}。"
        "参数部分的读法不是看谁的参数最多，而是看同一任务下参数变化是否带来稳定的评价提升；"
        "如果参数变化没有换来最终测试提升，后续调参应收窄搜索范围。"
    )


def _process_chapter_markdown() -> str:
    return (
        "## 第二章 过程与评价\n\n"
        "**本章只围绕结构化证据解释训练过程和效果。** 报告正文将数据集概况、参数设置、训练过程数据、"
        "模型评价四类信息按阅读顺序展开；字段和准确率等低密度信息用表格呈现，训练曲线和预测曲线等高密度信息用 ECharts 呈现。"
    )


def _metric_visual_markdown(context: dict[str, Any]) -> str:
    task = context.get("task") or {}
    final_key, final_value, final_source = _available_final_metric(context)
    if final_value is None:
        final_sentence = "当前尚未记录最终测试指标，因此只能先用选择阶段指标判断候选模型的相对表现。"
    elif final_source == "run_level":
        final_sentence = f"当前已有 Run 级最终测试指标 `{final_key}`={_fmt_value(final_value)}，但任务级最终评估尚未固化。"
    else:
        final_sentence = f"当前已有最终测试指标 `{final_key}`={_fmt_value(final_value)}。"
    return (
        "## 指标与图表读法\n\n"
        f"任务目标指标是 `{task.get('objective_metric') or '—'}`，优化方向是 "
        f"`{task.get('objective_direction') or '—'}`。选择阶段指标用于挑选候选模型，"
        f"最终测试指标用于报告模型泛化表现，二者不能直接混在一张结论表里解读。{final_sentence}\n\n"
        "效果指标表回答“哪个模型更好、差距在哪里”；训练过程曲线回答“训练是否收敛、是否过拟合”；"
        "ROC 或预测结果曲线回答“模型输出是否稳定”。下面保留的特征重要性图回答"
        "“模型主要依赖哪些输入变量”，它更适合解释模型行为，而不是重复展示准确率。"
    )


def _block_chart(chart_id: str, caption: str) -> dict[str, Any]:
    return {"type": "chart", "id": f"{chart_id}_block", "chart_id": chart_id, "caption": caption}


def _block_table(table_id: str, caption: str) -> dict[str, Any]:
    return {"type": "table", "id": f"{table_id}_block", "table_id": table_id, "caption": caption}


def _build_report_blocks(
    context: dict[str, Any],
    markdown: str,
    charts: list[dict[str, Any]],
    tables: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    chart_ids = {chart["id"] for chart in charts}
    table_ids = {table["id"] for table in tables}
    conclusion = _strip_markdown_tables(
        _section_from_markdown_any(markdown, ["第一章 结论", "一、结论"]) or markdown
    )
    suggestions = _strip_markdown_tables(
        _section_from_markdown_any(markdown, ["第三章 建议", "三、建议"])
    )

    blocks: list[dict[str, Any]] = [
        {"type": "markdown", "id": "conclusion", "markdown": _intro_title(markdown) + "\n" + conclusion},
        {"type": "markdown", "id": "task_scope", "markdown": _task_scope_markdown(context)},
        {"type": "markdown", "id": "process_chapter", "markdown": _process_chapter_markdown()},
        {"type": "markdown", "id": "data_profile_explanation", "markdown": _data_profile_markdown(context)},
    ]
    if "data_profile" in table_ids:
        blocks.append(_block_table(
            "data_profile",
            "这张表说明数据集中有哪些字段、每个字段扮演什么角色，以及是否存在缺失或编码风险。",
        ))
    blocks.append({"type": "markdown", "id": "parameter_explanation", "markdown": _parameter_markdown(context)})
    if "parameter_settings" in table_ids:
        blocks.append(_block_table(
            "parameter_settings",
            "这张表只保留本 task 训练真正需要追溯的配置：模型、策略、Trial、验证设置和关键参数。",
        ))
    blocks.append({"type": "markdown", "id": "model_training_process_explanation", "markdown": _training_process_markdown(context)})
    if "training_curves" in chart_ids:
        blocks.append(_block_chart(
            "training_curves",
            "这张图优先展示逐 epoch 的 loss/score 曲线；没有逐轮记录时展示 Trial 级选择指标与最终测试指标，用来判断调参过程是否稳定。",
        ))
    blocks.append({"type": "markdown", "id": "effect_summary", "markdown": _effect_summary_markdown(context)})
    if "metric_comparison" in table_ids:
        blocks.append(_block_table(
            "metric_comparison",
            "这张表用来横向比较准确率等效果指标，优先看最终测试指标，其次看选择/验证阶段指标。",
        ))
    if "roc_curve" in chart_ids:
        blocks.append(_block_chart(
            "roc_curve",
            "这张图用于分类任务，观察模型区分正负样本的能力；曲线越靠近左上角，区分能力越强。",
        ))
    if "prediction_curve" in chart_ids:
        blocks.append(_block_chart(
            "prediction_curve",
            "这张图把真实值与预测值按样本序号放在一起，方便观察预测偏差是否集中在局部样本。",
        ))
    if suggestions:
        blocks.append({"type": "markdown", "id": "suggestions", "markdown": suggestions})
    return blocks


def build_rich_report_payload(context: dict[str, Any] | str, markdown: str) -> dict[str, Any]:
    structured = context if isinstance(context, dict) else {}
    headline_metrics = _build_headline_metrics(structured, markdown)
    charts = _build_charts(structured)
    tables = _build_tables(structured)
    return {
        "report_schema_version": _REPORT_SCHEMA_VERSION,
        "headline_metrics": headline_metrics,
        "charts": charts,
        "tables": tables,
        "evidence": _build_evidence(structured),
        "report_blocks": _build_report_blocks(structured, markdown, charts, tables),
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
            "columns_info": _compact_value(dataset.columns_info or {}) if dataset else {},
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
            "style": "报告格式，总分结构，自然段表达",
            "reader_goal": "让读者了解任务要做什么、做了什么、训练过程怎么样、效果怎么样",
            "scope": "单 task 研究报告，不是单模型说明书",
            "presentation": "信息密度大的内容用 ECharts 图；信息密度低的内容用表格。只展示数据集概况、参数设置、训练过程数据、模型评价",
            "model_name_rule": "模型名称必须保留原始英文/代码标识，不要翻译或音译",
            "structured_tables": ["数据集概况", "参数设置", "模型评价"],
            "structured_charts": ["训练过程/调参过程指标曲线", "ROC 曲线", "预测结果曲线"],
            "sections": ["第一章 结论", "第二章 过程与评价", "第三章 建议"],
            "suggestion_scope": "建议只能围绕数据集概况、参数设置、训练过程数据、模型评价展开；不要提出 SHAP、特征重要性、部署监控或其他未展示数据类别",
        },
    }


def build_ai_report_messages(context: dict[str, Any] | str) -> list[dict[str, str]]:
    if isinstance(context, str):
        context_text = context[:_MAX_CONTEXT_CHARS]
    else:
        context_text = json.dumps(
            _context_for_llm(context),
            ensure_ascii=False,
            indent=2,
            default=str,
        )[:_MAX_CONTEXT_CHARS]

    system = (
        "你是机器学习建模结果审计与解释助手。你必须只依据用户给出的 JSON 上下文写报告；"
        "如果证据不足，就明确写“暂无足够证据”，不要编造数据、业务背景或实验结果。"
    )
    user = (
        "请根据下面的建模任务上下文生成一份中文 Markdown 研究报告。严格使用这个结构，"
        "不要输出代码块，不要输出额外章节，不要输出 Markdown 表格：\n\n"
        "报告给人看的核心目标是让读者清楚了解：任务要做什么、实际做了什么、"
        "训练过程怎么样、效果怎么样。请用正式报告口吻写，但每个自然段都要有明确证据，"
        "不要写成很短的摘要，不要使用口语化表达。每个小节至少包含一个多自然段说明。\n\n"
        "这是一份单 task 研究报告，不是某一个模型的说明书。报告的主语应该是“本任务”，"
        "不要机械地逐个模型罗列；只有当模型差异直接影响本 task 结论时才展开比较。\n\n"
        "章节必须有层次感，严格使用“第一章、第二章、第三章”以及“1.1、1.1.1”这类编号。"
        "每个自然段的第一句应写成明确结论句，前端会将第一句加粗亮色显示；"
        "因此第一句必须能独立表达本段结论，后续句子再做解释和补充说明。"
        "如果同一小节有多个结论或多条建议，必须用空行拆成多个自然段，不能挤在一个段落里。\n\n"
        "信息呈现原则：信息密度大的内容用 ECharts 图，例如有逐轮记录时的 loss 曲线、ROC 曲线、预测曲线；"
        "如果没有逐 epoch 记录，后端会使用真实 Trial 级选择指标与最终测试指标生成调参过程曲线，不能编造 loss。"
        "信息密度低的内容用表格，例如数据集字段概况、参数设置、Accuracy/F1/ROC-AUC 等模型评价指标。"
        "只展示数据集概况、参数设置、训练过程数据、模型评价，其他内容不要扩展成新的图表或表格。\n\n"
        "模型名称必须保留原始英文/代码标识，不要翻译或音译；例如 ARIMA 不要写成阿里玛，"
        "random_forest 不要写成随机森林，logistic_regression 不要写成逻辑回归。\n\n"
        "# AI 建模报告\n\n"
        "## 第一章 结论\n\n"
        "### 1.1 综合判断\n\n"
        "#### 1.1.1 任务结论\n\n"
        "总分：xx/100。随后用 1-2 个自然段说明整体判断、是否建议进入下一步、主要风险。"
        "必须说明本任务的入参是什么（数据集、目标列、任务类型、候选模型/策略），"
        "出参是什么（预测输出、评估指标、报告输出），但不要把所有字段挤成一个超长句。\n\n"
        "#### 1.1.2 关键依据\n\n"
        "用 1-2 个自然段说明结论所依据的核心数据，包括最终测试指标、训练过程稳定性和主要限制。\n\n"
        "## 第二章 过程与评价\n\n"
        "### 2.1 数据集概况\n\n"
        "#### 2.1.1 数据输入结论\n\n"
        "用自然段解释数据规模、目标列和字段质量，不要写字段百科。\n\n"
        "### 2.2 参数设置\n\n"
        "#### 2.2.1 训练配置结论\n\n"
        "用自然段解释本 task 的参数设置和搜索策略，不要机械罗列所有参数。\n\n"
        "### 2.3 训练过程\n\n"
        "#### 2.3.1 过程数据结论\n\n"
        "用自然段解释训练过程图应该如何支持或限制结论；如果上下文没有逐轮 loss，只能解释 Trial 级指标变化，不能编造 loss。\n\n"
        "### 2.4 模型评价\n\n"
        "#### 2.4.1 评价结论\n\n"
        "用自然段解释最终评估/选择阶段指标含义。"
        "必须区分选择阶段指标和最终测试指标；没有最终测试时说明结论可信度受限。"
        "准确率、召回率、F1、AUC、RMSE 这类低密度评价指标会由后端模型评价表展示，"
        "你只需要用自然段总结读法和结论，不要自己写 Markdown 表格。"
        "训练过程中的 loss 曲线、Trial 指标曲线、ROC 曲线、预测曲线会由前端用 ECharts 根据结构化 payload 渲染；"
        "你要根据上下文实际存在的曲线类型解释训练是否稳定，不要把曲线数据改写成表格，也不要声称看到了不存在的 loss 曲线。"
        "运行成功率不要作为重点图表，只在影响结论置信度时用自然段说明。"
        "如果上下文包含多个 Run，请围绕本 task 的参数变化、过程曲线和最终评价差异概括，不要写成模型百科。\n\n"
        "## 第三章 建议\n\n"
        "### 3.1 后续工作\n\n"
        "#### 3.1.1 优化建议\n\n"
        "用自然段提出可执行建议。每条建议必须连接前文四类数据中的证据：数据集概况、参数设置、训练过程数据或模型评价，"
        "不要罗列空泛建议，也不要新增未展示的数据类别。不要提出 SHAP、特征重要性、部署监控、业务流程、"
        "额外解释图等未在本报告结构化内容中展示的数据类别。多条建议必须分别成为独立自然段，每段第一句为结论句。\n\n"
        "建模任务上下文 JSON：\n"
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
    sentence_pattern = re.compile(r"^(.+?[。！？.!?])(\s*)(.*)$", flags=re.DOTALL)
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
        r"(?<=[。！？.!?])\s*(?=(\*\*[^*\n]{4,120}?[。！？.!?]\*\*))"
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
    report = await generate_ai_report_from_context(
        context,
        task_id=task_id,
        settings=settings,
    )
    return await archive_ai_report(db, report, owner_username=owner_username)


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

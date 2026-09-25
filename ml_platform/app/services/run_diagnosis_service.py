"""
Run Diagnosis Service — turn a single ExperimentRun's metrics/params/logs into
human-readable insights for the RunInspector context tab.

The service is pure (no IO): callers hand it the run, its sibling runs and
(optional) latest log lines; the service returns a dict with four fields plus
a narrative string that the frontend renders as an <Alert type="info">.

Returned shape::

    {
      "overfit": {
        "verdict": "overfit" | "underfit" | "ok" | "unknown",
        "gap_pct": float | None,                # (train-val)/val × 100
        "train_metric": float | None,
        "val_metric": float | None,
        "basis": str,                            # which metric pair we used
      },
      "failure_reason": {
        "keyword": str | None,                   # ValueError / MemoryError / CUDA ...
        "excerpt": str | None,                   # log line
        "explanation": str | None,               # 中文解释
      } | None,
      "param_impact": [
        {"param": str, "direction": "推高"|"拉低", "correlation": float}
      ],
      "peer_comparison": {
        "rank": int | None,
        "total": int,
        "value": float | None,
        "peer_mean": float | None,
        "delta_pct": float | None,               # (value - peer_mean)/peer_mean × 100
        "metric": str,
        "direction": "max" | "min",
      },
      "narrative": str,
    }

Keeping the logic dependency-free (std-lib only) so unit tests don't need
numpy/pandas just to assert the narrative string.
"""
from __future__ import annotations

import math
import re
from typing import Any, Iterable

from app.core.evaluation_metrics import resolve_objective_metrics


# ---------------------------------------------------------------------------
# Log keyword → 中文 attribution table. Ordered: first hit wins.
# ---------------------------------------------------------------------------
_FAILURE_DICT: list[tuple[str, str]] = [
    ("CUDA out of memory",   "GPU 显存不足，建议减小 batch_size 或选更小的模型。"),
    ("OutOfMemoryError",     "内存不足，尝试减少特征数或减小训练数据量。"),
    ("MemoryError",          "内存耗尽，建议关闭其它进程或缩小数据规模。"),
    ("ConvergenceWarning",   "模型未收敛，尝试增大 max_iter 或调整学习率。"),
    ("NaN",                  "训练出现 NaN/Inf 值，检查特征标准化与学习率设置。"),
    ("ValueError",           "参数或数据格式非法，请核对超参数类型与数据列。"),
    ("KeyError",              "列名缺失或字段对不上，检查数据集的列名。"),
    ("FileNotFoundError",    "依赖的文件路径不存在，数据集/模型文件可能被清理。"),
    ("TimeoutError",         "任务超时，尝试增大 timeout 或减少 epochs/trials。"),
    ("ImportError",          "依赖包缺失，请检查 requirements.txt 安装是否完整。"),
    ("PermissionError",      "文件权限问题，请检查 storage 目录可写性。"),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _finite(x: Any) -> float | None:
    """Return x as a float, or None if not a finite number."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _pick_train_val_pair(metrics: dict[str, Any]) -> tuple[str, float | None, float | None]:
    """Best-effort extract a (train, val) metric pair.

    Priority:
      1. explicit train_<m> / val_<m> pair from DL history last-epoch snapshot
      2. cv_avg_<m> (already an out-of-sample estimate) vs raw <m>
      3. nothing usable → (basis="", None, None)
    """
    if not isinstance(metrics, dict):
        return "", None, None

    # 1. Last-epoch DL snapshot inside metrics.history[-1]
    history = metrics.get("history")
    if isinstance(history, list) and history:
        last = history[-1]
        if isinstance(last, dict):
            # prefer accuracy pair, fall back to loss pair
            for train_key, val_key, basis in (
                ("train_acc",  "val_acc",  "train_acc vs val_acc"),
                ("train_loss", "val_loss", "train_loss vs val_loss"),
                ("acc",        "val_acc",  "acc vs val_acc"),
                ("loss",       "val_loss", "loss vs val_loss"),
            ):
                t = _finite(last.get(train_key))
                v = _finite(last.get(val_key))
                if t is not None and v is not None:
                    return basis, t, v

    # 2. cv_avg_<metric>  vs raw <metric>
    for raw_key in ("accuracy", "f1", "roc_auc", "rmse", "mae", "r2"):
        cv_key = f"cv_avg_{raw_key}"
        if cv_key in metrics and raw_key in metrics:
            t = _finite(metrics[raw_key])      # held-out test (single split)
            v = _finite(metrics[cv_key])       # CV mean (more robust)
            if t is not None and v is not None:
                return f"{raw_key} vs cv_avg_{raw_key}", t, v

    return "", None, None


def _classify_overfit(
    train: float | None, val: float | None, metric_basis: str,
) -> tuple[str, float | None]:
    """Return (verdict, gap_pct). Positive gap_pct = train > val."""
    if train is None or val is None or val == 0:
        return "unknown", None
    gap_pct = (train - val) / abs(val) * 100.0

    # loss-style metrics invert the sign: train_loss > val_loss means underfit
    is_loss = "loss" in metric_basis or "rmse" in metric_basis or "mae" in metric_basis
    if is_loss:
        # For losses: train << val means overfit (trained loss much smaller)
        if gap_pct < -15:
            return "overfit", gap_pct
        if gap_pct > 15:
            return "underfit", gap_pct
        return "ok", gap_pct
    else:
        # Accuracy-style: train >> val means overfit
        if gap_pct > 15:
            return "overfit", gap_pct
        if gap_pct < -15:
            return "underfit", gap_pct
        return "ok", gap_pct


# ---------------------------------------------------------------------------
# Failure attribution
# ---------------------------------------------------------------------------

def _scan_failure_logs(logs: Iterable[dict[str, Any]]) -> dict[str, str] | None:
    """Scan latest ERROR-level log lines for a known failure keyword."""
    for entry in reversed(list(logs)):  # most recent first
        level = str(entry.get("level") or "").upper()
        if level not in {"ERROR", "CRITICAL", "WARNING"}:
            continue
        msg = str(entry.get("message") or "")
        if not msg:
            continue
        for keyword, explanation in _FAILURE_DICT:
            if keyword.lower() in msg.lower():
                return {
                    "keyword": keyword,
                    "excerpt": msg[:300],
                    "explanation": explanation,
                }
    return None


# ---------------------------------------------------------------------------
# Param impact via Pearson correlation (no numpy dep).
# ---------------------------------------------------------------------------

def _pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 3:
        return None
    n = len(xs)
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    denom_x = math.sqrt(sum((x - mean_x) ** 2 for x in xs))
    denom_y = math.sqrt(sum((y - mean_y) ** 2 for y in ys))
    if denom_x == 0 or denom_y == 0:
        return None
    r = num / (denom_x * denom_y)
    return r if math.isfinite(r) else None


def _extract_numeric_hparams(run_params: dict[str, Any]) -> dict[str, float]:
    """Flatten run.params.hyperparameters keeping only numeric values."""
    hp = run_params.get("hyperparameters") if isinstance(run_params, dict) else None
    if not isinstance(hp, dict):
        return {}
    flat: dict[str, float] = {}
    for k, v in hp.items():
        if isinstance(v, bool):
            continue  # skip booleans — correlation is meaningless
        val = _finite(v)
        if val is not None:
            flat[str(k)] = val
    return flat


def _param_impact(
    run_metric: float | None,
    siblings: list[dict[str, Any]],
    metric_name: str,
    direction: str,
    this_run_id: str | None,
) -> list[dict[str, Any]]:
    """Top-2 hyperparameters whose values most correlate with the objective,
    across SUCCESS siblings of the same experiment.
    """
    samples: list[tuple[dict[str, float], float]] = []
    for sib in siblings or []:
        if str(sib.get("status", "")).upper() != "SUCCESS":
            continue
        metrics = sib.get("metrics") or {}
        val = resolve_objective_metrics(metrics, metric_name).selection_value
        if val is None:
            continue
        hp = _extract_numeric_hparams(sib.get("params") or {})
        if hp:
            samples.append((hp, val))

    if len(samples) < 3:
        return []

    # Collect the set of params that appear in >= 60% of samples
    param_counts: dict[str, int] = {}
    for hp, _v in samples:
        for k in hp:
            param_counts[k] = param_counts.get(k, 0) + 1
    threshold = max(3, int(0.6 * len(samples)))
    usable = [k for k, c in param_counts.items() if c >= threshold]

    impacts: list[dict[str, Any]] = []
    for param in usable:
        xs, ys = [], []
        for hp, v in samples:
            if param in hp:
                xs.append(hp[param])
                ys.append(v)
        # skip if everyone has the same value
        if len(set(xs)) < 2:
            continue
        r = _pearson(xs, ys)
        if r is None:
            continue
        # Interpret direction:
        # - "max" + positive r  → 该参数推高指标
        # - "max" + negative r  → 该参数拉低指标
        # - "min"               → 反之
        if direction == "max":
            narrative_dir = "推高" if r > 0 else "拉低"
        else:
            narrative_dir = "推高" if r < 0 else "拉低"  # for min, negative r helps
        impacts.append(
            {"param": param, "direction": narrative_dir, "correlation": round(r, 3)}
        )

    # Sort by |r| desc, take top 2
    impacts.sort(key=lambda d: abs(d["correlation"]), reverse=True)
    return impacts[:2]


# ---------------------------------------------------------------------------
# Peer comparison — where this run sits in the sibling pack
# ---------------------------------------------------------------------------

def _peer_comparison(
    run_metric: float | None,
    siblings: list[dict[str, Any]],
    metric_name: str,
    direction: str,
    this_run_id: str | None,
) -> dict[str, Any]:
    values: list[tuple[str, float]] = []
    for sib in siblings or []:
        if str(sib.get("status", "")).upper() != "SUCCESS":
            continue
        val = resolve_objective_metrics(
            sib.get("metrics") or {}, metric_name
        ).selection_value
        if val is not None:
            values.append((str(sib.get("id")), val))

    out: dict[str, Any] = {
        "metric": metric_name,
        "direction": direction,
        "total": len(values),
        "value": run_metric,
        "rank": None,
        "peer_mean": None,
        "delta_pct": None,
    }
    if not values:
        return out

    reverse = (direction == "max")
    sorted_vals = sorted(values, key=lambda t: t[1], reverse=reverse)
    mean_v = sum(v for _, v in values) / len(values)
    out["peer_mean"] = round(mean_v, 6)

    if this_run_id is not None:
        for i, (rid, _v) in enumerate(sorted_vals):
            if rid == this_run_id:
                out["rank"] = i + 1
                break
    if run_metric is not None and mean_v != 0:
        out["delta_pct"] = round((run_metric - mean_v) / abs(mean_v) * 100.0, 2)
    return out


# ---------------------------------------------------------------------------
# Narrative generator — Chinese f-string template
# ---------------------------------------------------------------------------

def _build_narrative(
    run: dict[str, Any],
    overfit: dict[str, Any],
    failure: dict[str, str] | None,
    impacts: list[dict[str, Any]],
    peer: dict[str, Any],
) -> str:
    parts: list[str] = []

    model = (run.get("params") or {}).get("model_type") or run.get("model_type") or "未知模型"
    strategy = (run.get("search_meta") or {}).get("strategy") or run.get("source_experiment_type") or "未知策略"
    metric = peer.get("metric") or "目标指标"
    val = peer.get("value")
    delta = peer.get("delta_pct")
    rank = peer.get("rank")
    total = peer.get("total")

    if val is not None:
        head = f"该 run 为 [{strategy}] 策略下 {model} 模型，{metric} = {val:.4f}"
        if rank and total:
            head += f"，在同批次 {total} 个 SUCCESS run 中排名 {rank}"
        if delta is not None:
            sign = "+" if delta >= 0 else ""
            head += f"，相对同批次平均 {sign}{delta:.2f}%。"
        else:
            head += "。"
        parts.append(head)
    else:
        parts.append(f"该 run 为 [{strategy}] 策略下 {model} 模型，但目标指标未成功记录。")

    # Overfit line
    verdict = overfit.get("verdict")
    gap = overfit.get("gap_pct")
    basis = overfit.get("basis")
    if verdict and verdict != "unknown" and gap is not None:
        if verdict == "overfit":
            parts.append(f"训练 / 验证 gap 为 {gap:+.1f}%（{basis}），判定为**过拟合**，建议增强正则、减小模型容量或增加验证样本。")
        elif verdict == "underfit":
            parts.append(f"训练 / 验证 gap 为 {gap:+.1f}%（{basis}），判定为**欠拟合**，建议增大模型容量或放宽正则。")
        else:
            parts.append(f"训练 / 验证 gap 仅 {gap:+.1f}%（{basis}），拟合状态良好。")

    # Param impact
    if impacts:
        bits = [f"{imp['param']}（{imp['direction']}，r={imp['correlation']:+.2f}）" for imp in impacts]
        parts.append("本批次关键影响参数：" + "、".join(bits) + "。")

    # Failure line (takes precedence if present — shown even with partial metrics)
    if failure:
        parts.append(f"失败归因：**{failure['keyword']}** — {failure['explanation']}")

    return " ".join(parts)


# ---------------------------------------------------------------------------
# Public entry
# ---------------------------------------------------------------------------

def diagnose_run(
    run: dict[str, Any],
    *,
    experiment: dict[str, Any] | None = None,
    siblings: list[dict[str, Any]] | None = None,
    logs: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Entry point. All inputs are plain dicts — caller extracts them from
    the ORM before handing them in (keeps this module DB-agnostic / testable).
    """
    siblings = siblings or []
    logs = logs or []
    metrics = run.get("metrics") or {}

    # 1. Overfit
    basis, train, val = _pick_train_val_pair(metrics)
    verdict, gap = _classify_overfit(train, val, basis)
    overfit = {
        "verdict": verdict,
        "gap_pct": None if gap is None else round(gap, 2),
        "train_metric": None if train is None else round(train, 6),
        "val_metric":   None if val is None else round(val, 6),
        "basis": basis,
    }

    # 2. Failure attribution (only meaningful for FAILED runs but we always scan
    #    — warnings in successful runs are still worth surfacing).
    failure = _scan_failure_logs(logs) if logs else None

    # 3/4. Need metric name + direction to rank and score param impact.
    metric_name = (experiment or {}).get("objective_metric") or "accuracy"
    direction   = (experiment or {}).get("objective_direction") or "max"
    run_metric  = resolve_objective_metrics(metrics, metric_name).selection_value
    run_id      = run.get("id")

    impacts = _param_impact(run_metric, siblings, metric_name, direction, run_id)
    peer    = _peer_comparison(run_metric, siblings, metric_name, direction, run_id)

    # 5. Narrative
    narrative = _build_narrative(run, overfit, failure, impacts, peer)

    return {
        "overfit": overfit,
        "failure_reason": failure,
        "param_impact": impacts,
        "peer_comparison": peer,
        "narrative": narrative,
    }

"""Unit tests for run_diagnosis_service — no DB, pure dict in/out."""
from __future__ import annotations

import pytest

from app.services.run_diagnosis_service import diagnose_run


def _make_run(metrics: dict, params: dict | None = None, run_id: str = "r0") -> dict:
    return {
        "id": run_id,
        "params": params or {"model_type": "xgboost"},
        "metrics": metrics,
        "search_meta": {"strategy": "baseline"},
        "source_experiment_type": "baseline",
        "status": "SUCCESS",
    }


def _exp(metric: str = "accuracy", direction: str = "max") -> dict:
    return {
        "id": "exp0",
        "name": "demo",
        "strategy_type": "baseline",
        "objective_metric": metric,
        "objective_direction": direction,
    }


# ---------------------------------------------------------------------------
# Overfit detection
# ---------------------------------------------------------------------------

def test_overfit_detected_when_train_much_higher_than_val():
    run = _make_run({
        "history": [{"train_acc": 0.99, "val_acc": 0.70}],
        "accuracy": 0.70,
    })
    out = diagnose_run(run=run, experiment=_exp(), siblings=[], logs=[])
    assert out["overfit"]["verdict"] == "overfit"
    assert out["overfit"]["gap_pct"] > 15
    assert "过拟合" in out["narrative"]


def test_underfit_detected_when_val_higher_than_train():
    run = _make_run({
        "history": [{"train_acc": 0.60, "val_acc": 0.85}],
        "accuracy": 0.85,
    })
    out = diagnose_run(run=run, experiment=_exp(), siblings=[], logs=[])
    assert out["overfit"]["verdict"] == "underfit"
    assert "欠拟合" in out["narrative"]


def test_ok_when_gap_is_small():
    run = _make_run({
        "accuracy": 0.9865,
        "cv_avg_accuracy": 0.9845,
    })
    out = diagnose_run(run=run, experiment=_exp(), siblings=[], logs=[])
    assert out["overfit"]["verdict"] == "ok"
    assert out["overfit"]["basis"] == "accuracy vs cv_avg_accuracy"


def test_loss_metrics_invert_direction():
    # train_loss << val_loss → overfit
    run = _make_run({"history": [{"train_loss": 0.05, "val_loss": 0.80}], "loss": 0.05})
    out = diagnose_run(run=run, experiment=_exp(), siblings=[], logs=[])
    assert out["overfit"]["verdict"] == "overfit"


# ---------------------------------------------------------------------------
# Failure attribution
# ---------------------------------------------------------------------------

def test_failure_reason_extracted_from_logs():
    run = _make_run({"accuracy": 0.0})
    logs = [
        {"level": "ERROR", "message": "Traceback: CUDA out of memory on device 0"},
    ]
    out = diagnose_run(run=run, experiment=_exp(), siblings=[], logs=logs)
    assert out["failure_reason"]["keyword"] == "CUDA out of memory"
    assert "显存" in out["failure_reason"]["explanation"]
    assert "失败归因" in out["narrative"]


def test_failure_ignored_when_no_error_logs():
    run = _make_run({"accuracy": 0.9})
    logs = [{"level": "INFO", "message": "training done"}]
    out = diagnose_run(run=run, experiment=_exp(), siblings=[], logs=logs)
    assert out["failure_reason"] is None


# ---------------------------------------------------------------------------
# Peer comparison
# ---------------------------------------------------------------------------

def test_peer_rank_computed_from_siblings():
    run = _make_run({"accuracy": 0.99, "cv_avg_accuracy": 0.85}, run_id="r1")
    siblings = [
        {"id": "r0", "status": "SUCCESS", "metrics": {"accuracy": 0.80, "cv_avg_accuracy": 0.90}, "params": {}},
        {"id": "r1", "status": "SUCCESS", "metrics": {"accuracy": 0.99, "cv_avg_accuracy": 0.85}, "params": {}},
        {"id": "r2", "status": "SUCCESS", "metrics": {"accuracy": 0.90, "cv_avg_accuracy": 0.80}, "params": {}},
    ]
    out = diagnose_run(run=run, experiment=_exp(), siblings=siblings, logs=[])
    assert out["peer_comparison"]["total"] == 3
    assert out["peer_comparison"]["rank"] == 2
    assert out["peer_comparison"]["peer_mean"] == pytest.approx(0.85, abs=1e-3)


def test_peer_comparison_empty_when_no_siblings():
    run = _make_run({"accuracy": 0.95})
    out = diagnose_run(run=run, experiment=_exp(), siblings=[], logs=[])
    assert out["peer_comparison"]["total"] == 0
    assert out["peer_comparison"]["rank"] is None


# ---------------------------------------------------------------------------
# Param impact
# ---------------------------------------------------------------------------

def test_param_impact_correlation_direction():
    # n_estimators correlates positively with accuracy
    siblings = []
    for i, (n, acc) in enumerate([(50, 0.80), (100, 0.85), (150, 0.90), (200, 0.93), (250, 0.96)]):
        siblings.append({
            "id": f"r{i}",
            "status": "SUCCESS",
            "metrics": {"accuracy": acc},
            "params": {"hyperparameters": {"n_estimators": n, "max_depth": 4}},
        })
    run = _make_run(
        {"accuracy": 0.96},
        params={"model_type": "rf", "hyperparameters": {"n_estimators": 250, "max_depth": 4}},
        run_id="r4",
    )
    out = diagnose_run(run=run, experiment=_exp(), siblings=siblings, logs=[])
    impacts = out["param_impact"]
    assert impacts, "should detect n_estimators impact"
    est = next((i for i in impacts if i["param"] == "n_estimators"), None)
    assert est is not None
    assert est["direction"] == "推高"   # positive r + max direction
    assert est["correlation"] > 0.9

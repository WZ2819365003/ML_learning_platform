#!/usr/bin/env python3
"""Seed 3 high-quality V3 demo plans + modeling tasks + experiment batches.

Strategy / family mapping (Phase 1 limit: any DL token forces baseline-only):

  Plan 1 — ML × bayesian_search   (5 ML models, Optuna TPE, max_trials=2 per model)
  Plan 2 — ML × grid_search       (5 ML models, truncated grid → ~10 runs)
  Plan 3 — Mixed × baseline       (3 ML + 3 DL = 6 baseline runs)

Run:
  python3 scripts/seed_v3_demo.py [--api http://127.0.0.1:8000/api]
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any
from urllib import error, request as urlreq


def http(method: str, url: str, body: dict | None = None) -> dict:
    data = None
    headers = {"Accept": "application/json"}
    if body is not None:
        data = json.dumps(body).encode()
        headers["Content-Type"] = "application/json"
    req = urlreq.Request(url, method=method, data=data, headers=headers)
    try:
        with urlreq.urlopen(req, timeout=30) as resp:
            text = resp.read().decode("utf-8") or "{}"
            return json.loads(text)
    except error.HTTPError as e:
        body_text = e.read().decode("utf-8", errors="ignore")[:500]
        sys.exit(f"!! {method} {url} -> HTTP {e.code}: {body_text}")
    except Exception as e:
        sys.exit(f"!! {method} {url} -> {e}")


def step(msg: str) -> None:
    print(f"\n===> {msg}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", default="http://127.0.0.1:8000/api")
    args = ap.parse_args()
    api = args.api.rstrip("/")

    # -----------------------------------------------------------------------
    # 0. Pick the largest classification dataset (predictive maintenance)
    # -----------------------------------------------------------------------
    step("Resolving target dataset")
    ds_resp = http("GET", f"{api}/data/list?page=1&page_size=20")
    pm = next(
        (
            d
            for d in ds_resp.get("items", [])
            if "predictive_maintenance" in d.get("name", "")
        ),
        None,
    )
    if not pm:
        sys.exit("predictive maintenance dataset not found in datasets list")
    print(f"  ✓ dataset_id={pm['id']}  rows={pm.get('row_count')}  target=Target")
    DATASET_ID = pm["id"]
    TARGET = "Target"

    # -----------------------------------------------------------------------
    # 1. Plan 1 — ML × bayesian_search (5 ML, max_trials=2 → 10 runs)
    # -----------------------------------------------------------------------
    step("Plan 1 — ML × bayesian_search")
    ml_models_p1 = ["logistic_regression", "random_forest", "xgboost", "lightgbm", "svm"]
    ml_bayesian_space = {
        "logistic_regression": {
            "C": {"type": "float", "low": 0.01, "high": 10.0, "log": True},
        },
        "random_forest": {
            "n_estimators": {"type": "int", "low": 50, "high": 300, "step": 50},
            "max_depth":    {"type": "int", "low": 4,  "high": 20},
        },
        "xgboost": {
            "n_estimators":  {"type": "int",   "low": 100, "high": 400, "step": 50},
            "learning_rate": {"type": "float", "low": 0.01, "high": 0.3, "log": True},
            "max_depth":     {"type": "int",   "low": 3,   "high": 10},
        },
        "lightgbm": {
            "n_estimators":  {"type": "int",   "low": 100, "high": 400, "step": 50},
            "learning_rate": {"type": "float", "low": 0.01, "high": 0.3, "log": True},
            "num_leaves":    {"type": "int",   "low": 15,  "high": 63},
        },
        "svm": {
            "C": {"type": "float", "low": 0.1, "high": 10.0, "log": True},
        },
    }
    plan1 = http("POST", f"{api}/platform/training-plans", {
        "name": "ML 调参对照（Bayesian）",
        "description": "5 个 ML 模型 × Optuna TPE 贝叶斯采样，max_trials=2/model → 10 runs",
        "task_type": "classification",
        "strategy_type": "bayesian_search",
        "model_family": "ml",
        "selected_models": ml_models_p1,
        "search_space": ml_bayesian_space,
        "eval_metrics": ["accuracy", "f1", "roc_auc"],
        "budget_config": {"max_trials": 2, "cv_folds": 3, "test_size": 0.2},
    })
    print(f"  ✓ plan_id={plan1['id']} name={plan1['name']!r}")

    # -----------------------------------------------------------------------
    # 2. Plan 2 — ML × grid_search (5 ML, truncated grid)
    # -----------------------------------------------------------------------
    step("Plan 2 — ML × grid_search")
    ml_models_p2 = ["logistic_regression", "random_forest", "xgboost", "lightgbm", "svm"]
    # Grid is cartesian — keep tiny per-model so total stays ~10 runs
    ml_grid_space = {
        "logistic_regression": {"C": [0.1, 1.0, 10.0]},                      # 3
        "random_forest":       {"n_estimators": [100, 300]},                 # 2
        "xgboost":             {"n_estimators": [100, 200], "max_depth": [3, 6]},  # 4
        "lightgbm":            {"n_estimators": [100, 300]},                 # 2
        "svm":                 {"C": [0.5, 2.0]},                            # 2
    }                                                                         # 13 total
    plan2 = http("POST", f"{api}/platform/training-plans", {
        "name": "ML 调参对照（Grid）",
        "description": "5 个 ML 模型 × 截断网格，约 13 runs",
        "task_type": "classification",
        "strategy_type": "grid_search",
        "model_family": "ml",
        "selected_models": ml_models_p2,
        "search_space": ml_grid_space,
        "eval_metrics": ["accuracy", "f1", "roc_auc"],
        "budget_config": {"max_trials": 20, "cv_folds": 3, "test_size": 0.2},
    })
    print(f"  ✓ plan_id={plan2['id']} name={plan2['name']!r}")

    # -----------------------------------------------------------------------
    # 3. Plan 3 — ML × baseline (5 ML, no tuning, vanilla defaults)
    # NOTE: original spec called for Mixed (ML+DL), but the deployed backend
    # image has no torch installed → tuning_service raises 501 on any DL token.
    # Pivoted to ML-baseline so all 3 strategies still get demo coverage.
    # -----------------------------------------------------------------------
    step("Plan 3 — ML × baseline (no DL: torch not in image)")
    ml_models_p3 = ["logistic_regression", "random_forest", "xgboost", "lightgbm", "svm"]
    plan3 = http("POST", f"{api}/platform/training-plans", {
        "name": "ML 基线（无调参）",
        "description": "5 个 ML 模型 baseline，无 search_space，默认超参 → 5 runs",
        "task_type": "classification",
        "strategy_type": "baseline",
        "model_family": "ml",
        "selected_models": ml_models_p3,
        "search_space": {},
        "eval_metrics": ["accuracy", "f1", "roc_auc"],
        "budget_config": {"max_trials": 1, "cv_folds": 3, "test_size": 0.2},
    })
    print(f"  ✓ plan_id={plan3['id']} name={plan3['name']!r}")

    # -----------------------------------------------------------------------
    # 4. For each plan, create a modeling_task + launch its experiment batch
    # -----------------------------------------------------------------------
    plans = [
        (plan1, "ml", "bayesian_search", ml_models_p1, ml_bayesian_space, {"max_trials": 2, "cv_folds": 3, "test_size": 0.2}),
        (plan2, "ml", "grid_search",     ml_models_p2, ml_grid_space,     {"max_trials": 20, "cv_folds": 3, "test_size": 0.2}),
        (plan3, "ml", "baseline",        ml_models_p3, {},                {"max_trials": 1, "cv_folds": 3, "test_size": 0.2}),
    ]
    tasks: list[dict] = []
    for plan, family, strategy, models, space, budget in plans:
        step(f"Modeling task for plan {plan['id'][:8]} ({family}/{strategy})")
        task = http("POST", f"{api}/v3/tasks/", {
            "name": f"任务-{plan['name']}",
            "task_type": "classification",
            "objective_metric": "accuracy",
            "objective_direction": "max",
            "training_plan_id": plan["id"],
            "dataset_id": DATASET_ID,
            "target_column": TARGET,
        })
        print(f"  ✓ task_id={task['id']}")

        step(f"Launching experiment batch (strategy={strategy}, models={len(models)})")
        batch = http("POST", f"{api}/v3/tasks/{task['id']}/experiments", {
            "name": f"批次-{plan['name']}",
            "strategy_type": strategy,
            "selected_models": models,
            "search_space": space,
            "budget_config": budget,
            "eval_metrics": ["accuracy", "f1", "roc_auc"],
            "model_family": family,
        })
        exp = batch.get("experiment", batch)
        print(f"  ✓ experiment_id={exp.get('id')} runs_planned={exp.get('runs_planned', 'N/A')}")
        tasks.append({"plan": plan, "task": task, "experiment": exp})

    # -----------------------------------------------------------------------
    # 5. Wait for all runs to finish (poll /v3/tasks/{id}/runs)
    # -----------------------------------------------------------------------
    step("Polling runs until all reach terminal state (max 10 min)")
    deadline = time.time() + 600
    while time.time() < deadline:
        all_done = True
        snapshot = []
        for t in tasks:
            r = http("GET", f"{api}/v3/tasks/{t['task']['id']}/runs")
            runs = r.get("items", []) or r.get("runs", [])
            terminal = sum(1 for x in runs if str(x.get("status", "")).upper() in ("SUCCESS", "FAILED", "CANCELED"))
            running  = sum(1 for x in runs if str(x.get("status", "")).upper() in ("PENDING", "RUNNING"))
            snapshot.append((t["plan"]["name"][:25], terminal, running, len(runs)))
            if running > 0 or len(runs) == 0:
                all_done = False
        line = " | ".join(f"{n}: {d}/{r}+{w}r" for n, d, w, r in snapshot)
        print(f"  ... {line}")
        if all_done:
            break
        time.sleep(5)

    # -----------------------------------------------------------------------
    # 6. Final summary + inspector probe on one run from each plan
    # -----------------------------------------------------------------------
    step("Final state per plan")
    for t in tasks:
        r = http("GET", f"{api}/v3/tasks/{t['task']['id']}/runs")
        runs = r.get("items", []) or r.get("runs", [])
        succ = [x for x in runs if str(x.get("status", "")).upper() == "SUCCESS"]
        failed = [x for x in runs if str(x.get("status", "")).upper() == "FAILED"]
        print(f"  Plan {t['plan']['name']}: total={len(runs)} success={len(succ)} failed={len(failed)}")
        if succ:
            sample_id = succ[0].get("run_id") or succ[0].get("id")
            insp = http("GET", f"{api}/platform/runs/{sample_id}/inspector")
            print(
                f"    inspector probe (run {sample_id[:8]}): "
                f"training_task={'✓' if insp.get('training_task') else '✗'} "
                f"logs={len(insp.get('logs', []))} "
                f"log_task_id={insp.get('log_task_id')}"
            )

    print("\nDone.")


if __name__ == "__main__":
    main()

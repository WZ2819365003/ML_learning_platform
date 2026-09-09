"""A report context shaped like task 0bc692d9 (测试1-电力负荷预测).

The numbers are the ones in doc/report-mock-overview.md, which is the target
output for the overview report: seven runs across five models, four of them
cross-validated tree models and three hold-out deep-learning models, a 35-column
dataset of which 24 columns were engineered, and a target histogram whose
counts and bin edges arrive as JSON text — exactly as they do from the
database, where the profile was stringified at depth three.
"""
from __future__ import annotations

import json
from typing import Any

BASE_COLUMNS = [
    "load", "timestamp", "hour", "month", "day_of_week", "is_weekend",
    "dry_bulb_temp", "wet_bulb_temp", "dew_point", "humidity", "wind_speed",
]
CYCLIC = ["dow_cos", "dow_sin", "doy_cos", "doy_sin", "hour_cos", "hour_sin", "month_cos", "month_sin"]
LAGS = ["load_lag_1", "load_lag_2", "load_lag_48", "load_lag_336"]
ROLLS = ["load_roll_mean_6", "load_roll_std_48", "load_roll_mean_48"]
WEATHER = [
    "temp_x_hour", "cooling_degree", "heating_degree", "weekend_x_hour", "days_since_start",
    "discomfort_index", "dry_bulb_temp_sq", "temp_spread_dry_wet", "dew_point_depression",
]
COLUMNS = BASE_COLUMNS + CYCLIC + LAGS + ROLLS + WEATHER

TARGET_MEAN = 8896.59
TARGET_MIN = 5498.36
TARGET_MAX = 14274.15

# Twenty bins over [min, max]; unimodal, slightly left-skewed body around
# 7900–9900, nine rows in the last bin.
HIST_COUNTS = [
    120, 640, 1980, 4360, 7420, 9950, 11840, 12630, 11710, 9420,
    6910, 4530, 2610, 1510, 860, 470, 220, 96, 25, 9,
]
assert sum(HIST_COUNTS) == 87310  # + 2 rows in under/overflow buckets below

XGB_SHAP = [
    {"feature": "load_lag_1", "mean_abs_shap": 976.4},
    {"feature": "load_lag_2", "mean_abs_shap": 95.6},
    {"feature": "load_roll_mean_6", "mean_abs_shap": 60.1},
    {"feature": "hour_sin", "mean_abs_shap": 40.2},
    {"feature": "load_lag_48", "mean_abs_shap": 35.5},
    {"feature": "temp_x_hour", "mean_abs_shap": 22.3},
    {"feature": "load_roll_mean_48", "mean_abs_shap": 18.7},
    {"feature": "dry_bulb_temp", "mean_abs_shap": 15.2},
]


def _folds(rmses: list[float], r2s: list[float]) -> list[dict[str, Any]]:
    return [
        {"fold": i + 1, "rmse": rmse, "mae": round(rmse * 0.72, 4), "r2": r2}
        for i, (rmse, r2) in enumerate(zip(rmses, r2s))
    ]


def _cv_metrics(rmse: float, std: float, r2: float, folds: list[dict[str, Any]] | None = None,
                shap: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    metrics = {
        "cv_avg_rmse": rmse, "cv_std_rmse": std, "selection_cv_mean_rmse": rmse,
        "cv_avg_mae": round(rmse * 0.72, 4), "cv_avg_r2": r2,
        "validation_strategy": "shuffled_kfold",
    }
    if folds is not None:
        metrics["cv_folds"] = folds
    if shap is not None:
        metrics["top_shap_importances"] = shap
    return metrics


def _history(n: int, best: int, rmse: float, start: float = 4.2e7) -> list[dict[str, Any]]:
    """An MSE curve that falls strictly until `best`, then sits just above it.

    Strictly, because a plateau of equal minima makes "the best epoch" the
    first of them: an earlier version clamped the curve at its floor from
    epoch 14 and reported a run that early-stopped at 38 with patience 10 as
    having peaked at 14. The validation loss at `best` is exactly rmse², so
    val_rmse there equals the score the run reports; the training loss keeps
    easing down afterwards while validation stays within 1% of its minimum.
    """
    target = rmse ** 2
    c = (start / target - 1) / (1 - 1 / best ** 3)
    rows = []
    for i in range(1, n + 1):
        curve = target * (1 + c * (1 / i ** 3 - 1 / best ** 3))
        val = curve if i <= best else target * (1 + 0.0005 * (i - best))
        rows.append({"epoch": i, "train_loss": round(curve * 0.92, 2),
                     "val_loss": round(val, 2), "val_rmse": round(val ** 0.5, 4), "lr": 0.001})
    return rows


def _scatter(n: int, rmse: float, seed: int) -> dict[str, Any]:
    import math
    actual = [TARGET_MEAN + 1800 * math.sin(i / 7.0) + 300 * math.sin(i / 1.3 + seed) for i in range(n)]
    predicted = [a + rmse * math.sin(i * 1.7 + seed) for i, a in enumerate(actual)]
    return {"actual": [round(v, 4) for v in actual], "predicted": [round(v, 4) for v in predicted],
            "ordered": True}


def _dl_metrics(rmse: float, epochs: int, best: int, seed: int) -> dict[str, Any]:
    return {
        "selection_val_rmse": rmse, "val_rmse": rmse, "val_mae": round(rmse * 0.75, 4),
        "val_r2": round(1 - (rmse / 1450) ** 2, 4),
        "history": _history(epochs, best, rmse),
        "val_scatter": _scatter(500, rmse, seed),
    }


def _entry(rank: int, run_id: str, model: str, value: float, metrics: dict[str, Any],
           trial_no: int = 1, params: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "rank": rank, "run_id": run_id, "experiment_name": "baseline", "strategy_type": "baseline",
        "trial_no": trial_no, "objective_value": value,
        "selection_metric_key": next(k for k in metrics if k.startswith("selection_")),
        "selection_value": value, "final_test_metric_key": None, "final_test_value": None,
        "model_type": model, "metrics": metrics,
        "params": {"model_type": model, "hyperparameters": params or {}},
    }


def leaderboard() -> list[dict[str, Any]]:
    dl_params = {"train_config": {"epochs": 50, "batch_size": 64, "early_stopping_patience": 10},
                 "arch_config": {"num_layers": 2, "hidden_size": 128}}
    return [
        # The rank-1 run kept its CV mean but not its folds; the rerun kept both.
        _entry(1, "46479fc1-7cd", "xgboost_regressor", 72.4673,
               _cv_metrics(72.4673, 0.8539, 0.9974, shap=XGB_SHAP)),
        _entry(2, "82afaa82-52d", "xgboost_regressor", 72.4673,
               _cv_metrics(72.4673, 0.8539, 0.9974,
                           _folds([72.9, 71.4, 73.1, 73.6, 71.3], [0.9974, 0.9975, 0.9973, 0.9973, 0.9975]),
                           shap=XGB_SHAP)),
        _entry(3, "89abfdf9-586", "lightgbm_regressor", 72.7856,
               _cv_metrics(72.7856, 0.9319, 0.9973,
                           _folds([72.8147, 71.8756, 73.5246, 74.0774, 71.6359],
                                  [0.9974, 0.9974, 0.9972, 0.9972, 0.9974]),
                           shap=[{"feature": "load_lag_1", "mean_abs_shap": 940.2},
                                 {"feature": "load_lag_2", "mean_abs_shap": 101.3},
                                 {"feature": "hour_sin", "mean_abs_shap": 44.0}]),
               trial_no=2),
        _entry(4, "707870a5-492", "random_forest_regressor", 79.2840,
               _cv_metrics(79.2840, 1.1042, 0.9968,
                           _folds([79.9, 78.1, 80.4, 79.6, 78.4], [0.9968, 0.9969, 0.9967, 0.9968, 0.9969]),
                           shap=[{"feature": "load_lag_1", "mean_abs_shap": 900.1},
                                 {"feature": "load_lag_2", "mean_abs_shap": 120.4}]),
               trial_no=3, params={"max_depth": 12}),
        _entry(5, "4827bf5e-d45", "mlp_dl", 132.0422, _dl_metrics(132.0422, 38, 28, 1),
               trial_no=5, params=dl_params),
        _entry(6, "47e1fe53-ca9", "lstm", 136.3387, _dl_metrics(136.3387, 41, 31, 2),
               trial_no=4, params=dl_params),
        _entry(7, "43c0e10d-2d0", "lstm", 187.3476, _dl_metrics(187.3476, 24, 14, 3),
               trial_no=1, params=dl_params),
    ]


def columns_info() -> dict[str, dict[str, Any]]:
    edges = [TARGET_MIN + i * (TARGET_MAX - TARGET_MIN) / 20 for i in range(21)]
    info: dict[str, dict[str, Any]] = {}
    for name in COLUMNS[:16]:  # the profile is capped at sixteen keys in the real payload
        info[name] = {"dtype": "float64", "missing_count": 0, "missing_rate": 0.0,
                      "unique_count": 366, "histogram": None}
    info["load"] = {
        "dtype": "float64", "missing_count": 0, "null_count": 0, "missing_rate": 0.0,
        "unique_count": 80922, "count": 87312,
        "mean": TARGET_MEAN, "min": TARGET_MIN, "max": TARGET_MAX,
        "quantiles": {"approx": True, "sample_size": 20000, "p25": 8150.4, "p50": 8931.0, "p75": 9648.7},
        # JSON text, as the depth-capped profile delivers them.
        "histogram": {
            "bin_edges": json.dumps([round(e, 2) for e in edges]),
            "counts": json.dumps(HIST_COUNTS),
            "underflow": 1, "overflow": 1, "missing": 0,
        },
    }
    return info


def context() -> dict[str, Any]:
    return {
        "task": {
            "id": "0bc692d9-93ee-4ff9-a7e5-6cdd463a7fd9",
            "name": "测试1-电力负荷预测",
            "dataset_name": "电力负荷预测数据.csv",
            "target_column": "load",
            "task_type": "regression",
            "objective_metric": "rmse",
            "objective_direction": "min",
            "status": "COMPLETED",
            "best_run_id": "46479fc1-7cd",
            "final_evaluation": {"state": "OPEN", "version": 1},
        },
        "dataset": {
            "id": "ds-1", "name": "电力负荷预测数据.csv", "row_count": 87312, "column_count": 35,
            "columns_info": columns_info(), "column_names": list(COLUMNS),
        },
        "experiments": [
            {"id": "e1", "name": "报告验证-ML", "strategy_type": "baseline",
             "selected_models": ["xgboost_regressor", "lightgbm_regressor", "random_forest_regressor"]},
            {"id": "e2", "name": "报告验证-DL", "strategy_type": "baseline", "selected_models": ["mlp_dl", "lstm"]},
            {"id": "e3", "name": "报告验证-混合", "strategy_type": "baseline",
             "selected_models": ["xgboost_regressor", "lstm"]},
        ],
        "run_status_counts": {"SUCCESS": 7},
        "leaderboard": leaderboard(),
        "successful_run_examples": [],
        "failed_run_examples": [],
        "_target_stats": {"mean": TARGET_MEAN, "min": TARGET_MIN, "max": TARGET_MAX, "std": None},
    }

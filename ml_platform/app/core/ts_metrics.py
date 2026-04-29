"""Forecasting-specific metrics — independent of regression metrics module."""
from __future__ import annotations

import numpy as np


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """% error; epsilon prevents div-by-zero on truth==0 elements."""
    eps = 1e-9
    return float(np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + eps))) * 100.0)


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Symmetric MAPE in [0, 200]."""
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    eps = 1e-9
    return float(np.mean(np.abs(y_true - y_pred) / (denom + eps)) * 100.0)


def mase(y_true: np.ndarray, y_pred: np.ndarray,
         train: np.ndarray, season: int = 1) -> float:
    """Mean Absolute Scaled Error — relative to seasonal naive on train."""
    if len(train) <= season:
        return float("nan")
    naive_err = np.mean(np.abs(train[season:] - train[:-season]))
    if naive_err < 1e-12:
        return 0.0 if mae(y_true, y_pred) < 1e-12 else float("inf")
    return float(np.mean(np.abs(y_true - y_pred)) / naive_err)


def coverage(y_true: np.ndarray, low: np.ndarray, high: np.ndarray) -> float:
    """Fraction of true values that fall within [low, high]."""
    return float(np.mean((y_true >= low) & (y_true <= high)))


FORECAST_EVAL_METRICS: list[dict] = [
    {"value": "mae",  "label": "MAE"},
    {"value": "rmse", "label": "RMSE"},
    {"value": "mape", "label": "MAPE (%)"},
    {"value": "smape","label": "sMAPE (%)"},
    {"value": "mase", "label": "MASE"},
    {"value": "coverage_80", "label": "80% 区间覆盖率"},
    {"value": "coverage_95", "label": "95% 区间覆盖率"},
]

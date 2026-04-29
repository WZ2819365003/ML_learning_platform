"""BaseTSTrainer — abstract contract for all ts family trainers."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd


@dataclass
class TSMeta:
    """Static metadata describing the time-series problem.

    Built once by ts_service from TrainingPlan.payload.time_series and
    handed to every trainer's fit() call.
    """
    timestamp_col: str
    target_col: str
    series_id_col: str | None
    exogenous_cols: list[str]
    freq: str
    horizon: int
    lookback: int
    interval_levels: list[int] = field(default_factory=lambda: [80, 95])


@dataclass
class ForecastResult:
    """Trainer.predict() output.

    mean: shape (horizon,) — point forecast
    intervals: dict[level, (low, high)] each shape (horizon,) — None if model has no interval support
    """
    mean: np.ndarray
    intervals: dict[int, tuple[np.ndarray, np.ndarray]] | None = None


class BaseTSTrainer(ABC):
    """Every ts family trainer must satisfy this contract.

    Subclasses MUST override `name` with the registry token (e.g., "arima").
    """

    name: str = ""  # subclasses must override with the registry token
    supports_intervals: bool = False
    supports_exogenous: bool = False

    @abstractmethod
    def fit(self, train_df: pd.DataFrame, meta: TSMeta, params: dict[str, Any]) -> None: ...

    @abstractmethod
    def predict(
        self, horizon: int, exog: pd.DataFrame | None = None
    ) -> ForecastResult: ...

    @abstractmethod
    def save(self, path: Path) -> None: ...

    @classmethod
    @abstractmethod
    def load(cls, path: Path) -> "BaseTSTrainer": ...


class ARIMATrainer(BaseTSTrainer):
    name = "arima"
    supports_intervals = True
    supports_exogenous = True

    def __init__(self):
        self._fitted = None
        self._meta: TSMeta | None = None

    def fit(self, train_df, meta, params):
        from statsmodels.tsa.arima.model import ARIMA
        order = (int(params.get("p", 1)), int(params.get("d", 1)), int(params.get("q", 1)))
        seasonal_p = int(params.get("seasonal_periods", 0))
        seasonal_order = (1, 1, 1, seasonal_p) if seasonal_p > 0 else (0, 0, 0, 0)
        y = train_df[meta.target_col].to_numpy()
        exog = None
        if meta.exogenous_cols:
            exog = train_df[meta.exogenous_cols].to_numpy()
        self._fitted = ARIMA(y, exog=exog, order=order, seasonal_order=seasonal_order).fit()
        self._meta = meta

    def predict(self, horizon, exog=None):
        fc = self._fitted.get_forecast(steps=horizon, exog=exog)
        mean = np.asarray(fc.predicted_mean)
        intervals: dict[int, tuple] = {}
        for level in (self._meta.interval_levels if self._meta else [80, 95]):
            alpha = 1.0 - level / 100.0
            ci = fc.conf_int(alpha=alpha)
            arr = ci if hasattr(ci, "shape") else ci.values
            arr = np.asarray(arr)
            intervals[level] = (arr[:, 0], arr[:, 1])
        return ForecastResult(mean=mean, intervals=intervals)

    def save(self, path):
        joblib.dump({"fitted": self._fitted, "meta": self._meta}, path)

    @classmethod
    def load(cls, path):
        data = joblib.load(path)
        t = cls()
        t._fitted = data["fitted"]
        t._meta = data["meta"]
        return t


class ETSTrainer(BaseTSTrainer):
    name = "ets"
    supports_intervals = True
    supports_exogenous = False

    def __init__(self):
        self._fitted = None
        self._meta: TSMeta | None = None

    def fit(self, train_df, meta, params):
        from statsmodels.tsa.holtwinters import ExponentialSmoothing
        trend = params.get("trend", "add")
        seasonal = params.get("seasonal", "none")
        seasonal_p = int(params.get("seasonal_periods", 0))
        kwargs = {}
        if trend != "none":
            kwargs["trend"] = trend
        if seasonal != "none" and seasonal_p > 0:
            kwargs["seasonal"] = seasonal
            kwargs["seasonal_periods"] = seasonal_p
        y = train_df[meta.target_col].to_numpy()
        self._fitted = ExponentialSmoothing(y, **kwargs).fit()
        self._meta = meta

    def predict(self, horizon, exog=None):
        from scipy.stats import norm
        mean = np.asarray(self._fitted.forecast(steps=horizon))
        # ExponentialSmoothing has no built-in CI; approximate via residual std.
        resid = self._fitted.resid
        sigma = float(np.std(resid)) if len(resid) else 0.0
        intervals: dict[int, tuple] = {}
        for level in (self._meta.interval_levels if self._meta else [80, 95]):
            z = norm.ppf(0.5 + level / 200.0)
            # Forecast uncertainty grows with horizon for additive trend; scale by sqrt(step).
            # This is a coarse approximation — full statsmodels ETSModel would simulate
            # the recursive innovation variance, but is heavier; revisit if needed.
            steps = np.arange(1, horizon + 1, dtype=float)
            half = z * sigma * np.sqrt(steps)
            intervals[level] = (mean - half, mean + half)
        return ForecastResult(mean=mean, intervals=intervals)

    def save(self, path):
        joblib.dump({"fitted": self._fitted, "meta": self._meta}, path)

    @classmethod
    def load(cls, path):
        data = joblib.load(path)
        t = cls()
        t._fitted = data["fitted"]
        t._meta = data["meta"]
        return t

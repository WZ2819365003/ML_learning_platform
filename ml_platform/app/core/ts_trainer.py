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


class LSTMForecasterTrainer(BaseTSTrainer):
    name = "lstm_forecaster"
    supports_intervals = False
    supports_exogenous = True

    def __init__(self):
        self._model = None
        self._meta: TSMeta | None = None
        self._x_last: np.ndarray | None = None
        self._params: dict[str, Any] | None = None

    @staticmethod
    def _build_windows(values: np.ndarray, lookback: int, horizon: int):
        """Slide (lookback+horizon) windows; target at column 0."""
        xs, ys = [], []
        for i in range(len(values) - lookback - horizon + 1):
            xs.append(values[i : i + lookback])
            ys.append(values[i + lookback : i + lookback + horizon, 0])
        return np.asarray(xs), np.asarray(ys)

    def fit(self, train_df, meta, params):
        import torch
        from torch.utils.data import DataLoader, TensorDataset
        from app.core.dl_models.lstm_forecaster import LSTMForecaster

        cols = [meta.target_col] + meta.exogenous_cols
        values = train_df[cols].to_numpy(dtype=np.float32)
        X, Y = self._build_windows(values, meta.lookback, meta.horizon)
        if len(X) == 0:
            raise ValueError(
                f"Not enough rows ({len(values)}) for lookback={meta.lookback}+horizon={meta.horizon}"
            )

        self._model = LSTMForecaster(
            input_size=len(cols),
            hidden_size=int(params["hidden_size"]),
            num_layers=int(params["num_layers"]),
            horizon=meta.horizon,
            dropout=float(params.get("dropout", 0.1)),
        )
        opt = torch.optim.Adam(self._model.parameters(), lr=float(params["learning_rate"]))
        loss_fn = torch.nn.MSELoss()
        ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(Y))
        dl = DataLoader(ds, batch_size=int(params["batch_size"]), shuffle=True)

        self._model.train()
        for _ in range(int(params["epochs"])):
            for xb, yb in dl:
                opt.zero_grad()
                pred = self._model(xb)
                loss = loss_fn(pred, yb)
                loss.backward()
                opt.step()

        self._meta = meta
        self._params = params
        self._x_last = values[-meta.lookback:]

    def predict(self, horizon, exog=None):
        import torch
        self._model.eval()
        with torch.no_grad():
            x = torch.from_numpy(self._x_last).unsqueeze(0)
            out = self._model(x).cpu().numpy()[0]
        return ForecastResult(mean=out[:horizon], intervals=None)

    def save(self, path):
        import torch
        torch.save({
            "state_dict": self._model.state_dict(),
            "meta": self._meta,
            "params": self._params,
            "x_last": self._x_last,
            "input_size": self._model.lstm.input_size,
        }, path)

    @classmethod
    def load(cls, path):
        import torch
        from app.core.dl_models.lstm_forecaster import LSTMForecaster
        data = torch.load(path, weights_only=False)
        t = cls()
        t._meta = data["meta"]
        t._params = data["params"]
        t._x_last = data["x_last"]
        t._model = LSTMForecaster(
            input_size=data["input_size"],
            hidden_size=int(t._params["hidden_size"]),
            num_layers=int(t._params["num_layers"]),
            horizon=t._meta.horizon,
            dropout=float(t._params.get("dropout", 0.1)),
        )
        t._model.load_state_dict(data["state_dict"])
        return t


class TCNForecasterTrainer(BaseTSTrainer):
    name = "tcn_forecaster"
    supports_intervals = False
    supports_exogenous = True

    def __init__(self):
        self._model = None
        self._meta = None
        self._x_last = None
        self._params = None

    # Reuse the LSTM trainer's static windowing logic (same direct multi-step shape)
    _build_windows = staticmethod(LSTMForecasterTrainer._build_windows)

    def fit(self, train_df, meta, params):
        import torch
        from torch.utils.data import DataLoader, TensorDataset
        from app.core.dl_models.tcn_forecaster import TCNForecaster

        cols = [meta.target_col] + meta.exogenous_cols
        values = train_df[cols].to_numpy(dtype=np.float32)
        X, Y = self._build_windows(values, meta.lookback, meta.horizon)
        if len(X) == 0:
            raise ValueError(
                f"Not enough rows ({len(values)}) for lookback={meta.lookback}+horizon={meta.horizon}"
            )

        self._model = TCNForecaster(
            input_size=len(cols),
            channels=int(params["channels"]),
            kernel_size=int(params["kernel_size"]),
            num_layers=int(params["num_layers"]),
            horizon=meta.horizon,
            dropout=float(params.get("dropout", 0.1)),
        )
        opt = torch.optim.Adam(self._model.parameters(), lr=float(params["learning_rate"]))
        loss_fn = torch.nn.MSELoss()
        ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(Y))
        dl = DataLoader(ds, batch_size=int(params["batch_size"]), shuffle=True)

        self._model.train()
        for _ in range(int(params["epochs"])):
            for xb, yb in dl:
                opt.zero_grad()
                pred = self._model(xb)
                loss = loss_fn(pred, yb)
                loss.backward()
                opt.step()

        self._meta = meta
        self._params = params
        self._x_last = values[-meta.lookback:]

    def predict(self, horizon, exog=None):
        import torch
        self._model.eval()
        with torch.no_grad():
            x = torch.from_numpy(self._x_last).unsqueeze(0)
            out = self._model(x).cpu().numpy()[0]
        return ForecastResult(mean=out[:horizon], intervals=None)

    def save(self, path):
        import torch
        torch.save({
            "state_dict": self._model.state_dict(),
            "meta": self._meta,
            "params": self._params,
            "x_last": self._x_last,
            "input_size": self._model.tcn[0].conv1.in_channels,
        }, path)

    @classmethod
    def load(cls, path):
        import torch
        from app.core.dl_models.tcn_forecaster import TCNForecaster
        data = torch.load(path, weights_only=False)
        t = cls()
        t._meta = data["meta"]
        t._params = data["params"]
        t._x_last = data["x_last"]
        t._model = TCNForecaster(
            input_size=data["input_size"],
            channels=int(t._params["channels"]),
            kernel_size=int(t._params["kernel_size"]),
            num_layers=int(t._params["num_layers"]),
            horizon=t._meta.horizon,
            dropout=float(t._params.get("dropout", 0.1)),
        )
        t._model.load_state_dict(data["state_dict"])
        return t

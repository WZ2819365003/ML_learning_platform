"""BaseTSTrainer — abstract contract for all ts family trainers."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

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

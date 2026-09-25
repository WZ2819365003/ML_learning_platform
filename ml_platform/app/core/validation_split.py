"""Validation split selection shared by classic ML and deep learning."""

from __future__ import annotations

import math
import re
from typing import Any, Iterable


_TEMPORAL_FEATURE = re.compile(
    r"(?:_lag_\d+$|_roll_|(?:^|_)(?:hour|day|dow|week|month|year)(?:_|$)|^days_since_)",
    re.IGNORECASE,
)


def has_temporal_features(columns: Iterable[Any]) -> bool:
    """Detect feature sets whose row order carries forecasting semantics.

    Lag and rolling columns are decisive. Calendar/cyclical names also match so
    a prepared forecasting frame without explicit lag columns remains safe.
    """
    return any(_TEMPORAL_FEATURE.search(str(column)) for column in columns)


def is_temporal_feature_frame(frame: Any) -> bool:
    columns = getattr(frame, "columns", None)
    return columns is not None and has_temporal_features(columns)


def chronological_train_test_split(X: Any, y: Any, test_size: float):
    """Keep the newest rows as hold-out and never shuffle temporal samples."""
    size = len(X)
    if len(y) != size:
        raise ValueError("X 与 y 的样本数不一致")
    if not 0 < float(test_size) < 1:
        raise ValueError("test_size 必须在 0 和 1 之间")
    holdout = max(1, int(math.ceil(size * float(test_size))))
    split_at = size - holdout
    if split_at < 2:
        raise ValueError("时间顺序切分后训练样本不足，至少需要保留 2 条训练样本")

    def take(value: Any, start: int | None, end: int | None):
        if hasattr(value, "iloc"):
            return value.iloc[slice(start, end)]
        return value[slice(start, end)]

    return (
        take(X, None, split_at),
        take(X, split_at, None),
        take(y, None, split_at),
        take(y, split_at, None),
    )

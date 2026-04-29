import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from app.core.ts_trainer import TimesFMAdapter, TSMeta


@pytest.fixture
def df():
    n = 100
    return pd.DataFrame({
        "ds": pd.date_range("2024-01-01", periods=n, freq="D"),
        "y":  np.sin(np.arange(n) / 10),
    })


@pytest.fixture
def meta():
    return TSMeta(timestamp_col="ds", target_col="y", series_id_col=None,
                  exogenous_cols=[], freq="D", horizon=14, lookback=64,
                  interval_levels=[])


def test_timesfm_save_load_marker(df, meta, tmp_path):
    """Even when venv missing, save/load round-trip should work via marker.json."""
    t = TimesFMAdapter()
    t.fit(df, meta, params={"context_len": 64})  # fit is a no-op
    p = tmp_path / "timesfm.json"
    t.save(p)
    assert p.exists()
    payload = json.loads(p.read_text())
    assert payload["model"] == "timesfm_1"

    t2 = TimesFMAdapter.load(p)
    assert t2._meta.horizon == 14

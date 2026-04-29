# ml_platform/tests/test_ts_trainer_ets.py
import numpy as np
import pandas as pd
import pytest
from app.core.ts_trainer import ETSTrainer, TSMeta


@pytest.fixture
def trend_df():
    n = 60
    ds = pd.date_range("2024-01-01", periods=n, freq="D")
    y = np.arange(n, dtype=float) + np.random.RandomState(1).randn(n) * 0.5
    return pd.DataFrame({"ds": ds, "y": y})


@pytest.fixture
def meta():
    return TSMeta(
        timestamp_col="ds", target_col="y", series_id_col=None,
        exogenous_cols=[], freq="D", horizon=7, lookback=14,
        interval_levels=[80, 95],
    )


def test_ets_fits_trend(trend_df, meta):
    t = ETSTrainer()
    t.fit(trend_df, meta, params={"trend": "add", "seasonal": "none", "seasonal_periods": 0})
    res = t.predict(horizon=7)
    assert res.mean.shape == (7,)
    # mean should continue trend roughly (linearly increasing)
    assert res.mean[-1] > res.mean[0]


def test_ets_save_load(trend_df, meta, tmp_path):
    t = ETSTrainer()
    t.fit(trend_df, meta, params={"trend": "add", "seasonal": "none", "seasonal_periods": 0})
    p = tmp_path / "ets.joblib"
    t.save(p)
    t2 = ETSTrainer.load(p)
    np.testing.assert_allclose(t.predict(7).mean, t2.predict(7).mean, rtol=1e-6)

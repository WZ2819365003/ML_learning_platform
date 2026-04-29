# ml_platform/tests/test_ts_trainer_arima.py
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from app.core.ts_trainer import ARIMATrainer, TSMeta


@pytest.fixture
def sin_wave_df():
    n = 100
    ds = pd.date_range("2024-01-01", periods=n, freq="D")
    y = np.sin(np.arange(n) * 2 * np.pi / 30) + np.random.RandomState(0).randn(n) * 0.1
    return pd.DataFrame({"ds": ds, "y": y})


@pytest.fixture
def meta():
    return TSMeta(
        timestamp_col="ds", target_col="y", series_id_col=None,
        exogenous_cols=[], freq="D", horizon=10, lookback=30,
        interval_levels=[80, 95],
    )


def test_arima_fit_predict_shape(sin_wave_df, meta):
    t = ARIMATrainer()
    t.fit(sin_wave_df, meta, params={"p": 1, "d": 0, "q": 1, "seasonal_periods": 0})
    res = t.predict(horizon=10)
    assert res.mean.shape == (10,)
    assert res.intervals is not None
    assert 80 in res.intervals
    lo, hi = res.intervals[80]
    assert lo.shape == (10,) and hi.shape == (10,)
    assert np.all(lo <= hi)


def test_arima_save_load_roundtrip(sin_wave_df, meta, tmp_path):
    t = ARIMATrainer()
    t.fit(sin_wave_df, meta, params={"p": 1, "d": 0, "q": 1, "seasonal_periods": 0})
    p = tmp_path / "arima.joblib"
    t.save(p)
    t2 = ARIMATrainer.load(p)
    res1 = t.predict(horizon=5)
    res2 = t2.predict(horizon=5)
    np.testing.assert_allclose(res1.mean, res2.mean, rtol=1e-6)

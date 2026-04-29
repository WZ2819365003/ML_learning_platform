import numpy as np
import pandas as pd
import pytest
from app.core.ts_trainer import TCNForecasterTrainer, TSMeta


@pytest.fixture
def sin_df():
    n = 200
    ds = pd.date_range("2024-01-01", periods=n, freq="D")
    y = np.sin(np.arange(n) * 2 * np.pi / 20)
    return pd.DataFrame({"ds": ds, "y": y})


@pytest.fixture
def meta():
    return TSMeta(timestamp_col="ds", target_col="y", series_id_col=None,
                  exogenous_cols=[], freq="D", horizon=10, lookback=30,
                  interval_levels=[])


def test_tcn_forecaster_fit_predict(sin_df, meta):
    t = TCNForecasterTrainer()
    t.fit(sin_df, meta, params={
        "channels": 16, "kernel_size": 3, "num_layers": 2, "dropout": 0.0,
        "epochs": 5, "batch_size": 16, "learning_rate": 0.01,
    })
    res = t.predict(horizon=10)
    assert res.mean.shape == (10,)
    assert res.intervals is None


def test_tcn_save_load(sin_df, meta, tmp_path):
    t = TCNForecasterTrainer()
    t.fit(sin_df, meta, params={
        "channels": 8, "kernel_size": 3, "num_layers": 2, "dropout": 0.0,
        "epochs": 3, "batch_size": 16, "learning_rate": 0.01,
    })
    p = tmp_path / "tcn.pt"
    t.save(p)
    t2 = TCNForecasterTrainer.load(p)
    np.testing.assert_allclose(t.predict(5).mean, t2.predict(5).mean, atol=1e-5)

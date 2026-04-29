import numpy as np
from app.core.ts_metrics import mae, rmse, mape, smape, mase, coverage


def test_mae_basic():
    y = np.array([1.0, 2.0, 3.0])
    p = np.array([1.0, 2.5, 2.5])
    assert abs(mae(y, p) - (0 + 0.5 + 0.5) / 3) < 1e-9


def test_rmse_basic():
    y = np.array([0.0, 0.0])
    p = np.array([3.0, 4.0])
    assert abs(rmse(y, p) - (np.sqrt((9 + 16) / 2))) < 1e-9


def test_mape_handles_zero_truth():
    y = np.array([0.0, 1.0, 2.0])
    p = np.array([0.5, 1.0, 1.0])
    val = mape(y, p)
    assert val >= 0 and not np.isnan(val)


def test_smape_symmetric():
    y = np.array([2.0, 4.0])
    p = np.array([4.0, 2.0])
    a = smape(y, p)
    b = smape(p, y)
    assert 0 < a <= 200
    assert abs(a - b) < 1e-9, f"sMAPE is not symmetric: {a} vs {b}"


def test_mase_uses_seasonal_naive():
    train = np.array([1, 2, 3, 4, 5, 6])
    y = np.array([7, 8])
    p = np.array([7, 8])
    val = mase(y, p, train, season=1)
    assert abs(val - 0.0) < 1e-9


def test_coverage_full():
    y = np.array([1.0, 2.0, 3.0])
    lo = np.array([0.5, 1.5, 2.5])
    hi = np.array([1.5, 2.5, 3.5])
    assert abs(coverage(y, lo, hi) - 1.0) < 1e-9

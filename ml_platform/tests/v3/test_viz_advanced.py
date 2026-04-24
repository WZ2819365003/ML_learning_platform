"""End-to-end tests for the advanced viz endpoints.

Synthesizes a real RandomForestClassifier artifact on disk + TrainingTask row
so the full viz pipeline runs: resolver → load model → compute metrics.  This
gives us genuine coverage of per_class / pr_curve / calibration / threshold /
distribution rather than just unit-testing the sklearn wrappers.
"""

from __future__ import annotations

import uuid

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from app.services import viz_service


@pytest.fixture
async def binary_rf_task(db, tmp_path_factory):
    """Create a synthetic binary classifier + TrainingTask row + Dataset row.

    Settings is a frozen dataclass so we can't monkeypatch storage dirs;
    instead we write into the real ones under a unique task id and clean up.
    """
    from app.config import get_settings
    from app.models.database import Dataset, TrainingTask

    settings = get_settings()
    settings.ensure_storage_dirs()

    rng = np.random.RandomState(42)
    X = rng.randn(200, 4)
    y = (X[:, 0] + 0.5 * X[:, 1] + rng.randn(200) * 0.3 > 0).astype(int)
    df = pd.DataFrame(X, columns=["feat_a", "feat_b", "feat_c", "feat_d"])
    df["label"] = y

    task_id = "viz_" + uuid.uuid4().hex[:8]
    csv_path = settings.storage_uploads / f"{task_id}.csv"
    df.to_csv(csv_path, index=False)

    model = RandomForestClassifier(n_estimators=30, random_state=0)
    model.fit(X, y)
    model_path = settings.storage_models / f"{task_id}.joblib"
    joblib.dump(model, model_path)

    ds = Dataset(
        id=f"ds-{task_id}", name="viz-binary", file_path=str(csv_path),
        file_size=1024, row_count=200, column_count=5,
    )
    tt = TrainingTask(
        id=task_id, dataset_id=ds.id, name="viz-rf-binary",
        model_type="random_forest", target_column="label",
        hyperparameters={}, eval_metrics=["accuracy"],
        test_size=0.25, status="SUCCESS", progress=100,
        model_path=f"storage/models/{task_id}.joblib",
    )
    db.add_all([ds, tt])
    await db.commit()

    yield task_id

    for p in (csv_path, model_path):
        try:
            p.unlink()
        except FileNotFoundError:
            pass


@pytest.fixture
async def regressor_rf_task(db):
    """Synthetic RandomForestRegressor for regression-only viz paths."""
    from app.config import get_settings
    from app.models.database import Dataset, TrainingTask

    settings = get_settings()
    settings.ensure_storage_dirs()

    rng = np.random.RandomState(7)
    X = rng.randn(150, 3)
    y = 2.0 * X[:, 0] - X[:, 1] + rng.randn(150) * 0.4
    df = pd.DataFrame(X, columns=["x1", "x2", "x3"])
    df["target"] = y

    task_id = "viz_reg_" + uuid.uuid4().hex[:8]
    csv_path = settings.storage_uploads / f"{task_id}.csv"
    df.to_csv(csv_path, index=False)

    model = RandomForestRegressor(n_estimators=20, random_state=0)
    model.fit(X, y)
    model_path = settings.storage_models / f"{task_id}.joblib"
    joblib.dump(model, model_path)

    ds = Dataset(
        id=f"ds-{task_id}", name="viz-reg", file_path=str(csv_path),
        file_size=1024, row_count=150, column_count=4,
    )
    tt = TrainingTask(
        id=task_id, dataset_id=ds.id, name="viz-rf-reg",
        model_type="random_forest_regressor", target_column="target",
        hyperparameters={}, eval_metrics=["rmse"],
        test_size=0.25, status="SUCCESS", progress=100,
        model_path=f"storage/models/{task_id}.joblib",
    )
    db.add_all([ds, tt])
    await db.commit()

    yield task_id

    for p in (csv_path, model_path):
        try:
            p.unlink()
        except FileNotFoundError:
            pass


# ---------------------------------------------------------------------------
# per_class metrics
# ---------------------------------------------------------------------------

async def test_per_class_returns_rows_and_aggregates(binary_rf_task, db):
    payload = await viz_service.get_per_class_metrics(binary_rf_task, db)
    assert len(payload["rows"]) == 2  # binary
    for row in payload["rows"]:
        assert 0.0 <= row["precision"] <= 1.0
        assert 0.0 <= row["recall"] <= 1.0
        assert 0.0 <= row["f1"] <= 1.0
        assert row["support"] > 0
    assert payload["macro_avg"] is not None
    assert payload["weighted_avg"] is not None
    assert 0.0 <= payload["accuracy"] <= 1.0


async def test_per_class_400_for_regression(regressor_rf_task, db):
    from fastapi import HTTPException
    with pytest.raises(HTTPException) as exc:
        await viz_service.get_per_class_metrics(regressor_rf_task, db)
    assert exc.value.status_code == 400


# ---------------------------------------------------------------------------
# PR curve
# ---------------------------------------------------------------------------

async def test_pr_curve_binary(binary_rf_task, db):
    payload = await viz_service.get_pr_curve(binary_rf_task, db)
    assert payload["multiclass"] is False
    assert len(payload["precision"]) == len(payload["recall"])
    assert len(payload["precision"]) == len(payload["thresholds"]) + 1
    assert 0.0 <= payload["average_precision"] <= 1.0
    assert 0.0 <= payload["best_threshold"] <= 1.0
    assert 0.0 <= payload["best_f1"] <= 1.0


async def test_pr_curve_400_for_regression(regressor_rf_task, db):
    from fastapi import HTTPException
    with pytest.raises(HTTPException) as exc:
        await viz_service.get_pr_curve(regressor_rf_task, db)
    assert exc.value.status_code == 400


# ---------------------------------------------------------------------------
# Calibration curve
# ---------------------------------------------------------------------------

async def test_calibration_binary(binary_rf_task, db):
    payload = await viz_service.get_calibration_curve(binary_rf_task, db, n_bins=10)
    assert len(payload["prob_pred"]) == len(payload["prob_true"])
    assert payload["n_bins"] == 10
    assert payload["ece"] >= 0.0
    assert 0.0 <= payload["brier"] <= 1.0


# ---------------------------------------------------------------------------
# Threshold analysis
# ---------------------------------------------------------------------------

async def test_threshold_sweep_binary(binary_rf_task, db):
    payload = await viz_service.get_threshold_analysis(binary_rf_task, db, step=0.1)
    # step=0.1 → thresholds 0.1,0.2,...,0.9 = 9 rows
    assert len(payload["rows"]) == 9
    for row in payload["rows"]:
        assert 0.0 < row["threshold"] < 1.0
        assert 0.0 <= row["f1"] <= 1.0
    # best_threshold is one of the swept thresholds
    swept = [r["threshold"] for r in payload["rows"]]
    assert payload["best_threshold"] in swept


# ---------------------------------------------------------------------------
# Prediction distribution
# ---------------------------------------------------------------------------

async def test_distribution_binary_has_pos_and_neg_counts(binary_rf_task, db):
    payload = await viz_service.get_prediction_distribution(binary_rf_task, db, bins=20)
    assert payload["kind"] == "classification_binary_proba"
    assert len(payload["bin_edges"]) == 21  # bins+1
    assert len(payload["positive_counts"]) == 20
    assert len(payload["negative_counts"]) == 20
    # Total counts should match test set size
    total = sum(payload["positive_counts"]) + sum(payload["negative_counts"])
    assert total == 50  # 200 samples × 0.25 test_size


async def test_distribution_regression_returns_residual_histogram(regressor_rf_task, db):
    payload = await viz_service.get_prediction_distribution(regressor_rf_task, db, bins=15)
    assert payload["kind"] == "regression_residuals"
    assert len(payload["bin_edges"]) == 16
    assert len(payload["counts"]) == 15
    assert isinstance(payload["mean"], float)
    assert payload["std"] > 0

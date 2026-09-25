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


@pytest.fixture
async def categorical_artifact_task(db):
    from app.config import get_settings
    from app.core.model_artifact import fit_tabular_artifact
    from app.models.database import Dataset, TrainingTask

    settings = get_settings()
    settings.ensure_storage_dirs()

    rng = np.random.RandomState(11)
    size = 180
    frame = pd.DataFrame({
        "temperature": rng.normal(300, 8, size),
        "machine_type": np.where(np.arange(size) % 3 == 0, "A", "B"),
    })
    target = ((frame["temperature"] > 300) | (frame["machine_type"] == "A")).astype(int)
    dataset_frame = frame.copy()
    dataset_frame["failed"] = target

    task_id = "viz_artifact_" + uuid.uuid4().hex[:8]
    csv_path = settings.storage_uploads / f"{task_id}.csv"
    model_path = settings.storage_models / f"{task_id}.joblib"
    dataset_frame.to_csv(csv_path, index=False)

    artifact = fit_tabular_artifact(
        RandomForestClassifier(n_estimators=20, random_state=0),
        frame,
        target,
        task_kind="classification",
    )
    joblib.dump(artifact, model_path)

    dataset = Dataset(
        id=f"ds-{task_id}", name="viz-artifact", file_path=str(csv_path),
        file_size=1024, row_count=size, column_count=3,
    )
    task = TrainingTask(
        id=task_id, dataset_id=dataset.id, name="viz-artifact-classifier",
        model_type="random_forest", target_column="failed",
        hyperparameters={}, eval_metrics=["accuracy"], test_size=0.2,
        status="SUCCESS", progress=100,
        model_path=f"storage/models/{task_id}.joblib",
    )
    db.add_all([dataset, task])
    await db.commit()

    yield task_id

    for path in (csv_path, model_path):
        try:
            path.unlink()
        except FileNotFoundError:
            pass


async def test_core_classification_viz_supports_tabular_artifact(categorical_artifact_task, db):
    matrix = await viz_service.get_confusion_matrix(categorical_artifact_task, db)
    curve = await viz_service.get_roc_curve(categorical_artifact_task, db)

    assert len(matrix["matrix"]) == 2
    assert 0.0 <= curve["auc"] <= 1.0


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


async def test_regression_prediction_payload_uses_bounded_tail_window(
    regressor_rf_task, db
):
    payload = await viz_service.get_predicted_vs_actual(
        regressor_rf_task, db, max_samples=10
    )

    assert payload["sample_count"] == 10
    assert payload["total_count"] == 38  # ceil(150 × 0.25)
    assert payload["sample_offset"] == 28
    assert payload["truncated"] is True
    assert len(payload["actual"]) == len(payload["predicted"]) == 10


async def test_regression_distribution_respects_sample_limit(regressor_rf_task, db):
    payload = await viz_service.get_prediction_distribution(
        regressor_rf_task, db, bins=10, max_samples=12
    )

    assert payload["sample_count"] == 12
    assert payload["total_count"] == 38
    assert payload["truncated"] is True
    assert sum(payload["counts"]) == 12

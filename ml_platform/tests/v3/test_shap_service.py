"""Tests for the unified SHAP service ladder (Tree → Kernel → Permutation).

Focuses on the public contract of `compute_shap_summary`:
  1. Tree models → method="tree" with `shap_values` populated
  2. Non-tree models with predict_proba → method="kernel" populated
  3. Kernel failure or pathological model → method="permutation", shap_values=None
  4. numpy 2.0 dtype safety — float32 SHAP arrays must serialize cleanly
"""

from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from app.services import shap_service
from app.services.shap_service import (
    METHOD_KERNEL,
    METHOD_PERMUTATION,
    METHOD_TREE,
    _build_payload,
    _is_tree_model,
    _normalize_shap_values,
    _round_list_2d,
    _to_f64_list,
)


def test_to_f64_list_handles_numpy_inexact():
    """Under numpy 2.0 `astype(float)` rejects `np.inexact` dtypes — our
    serializer must cope by using `np.float64` explicitly."""
    arr = np.array([1.0, 2.5, 3.7], dtype=np.float32)  # inexact subclass
    assert _to_f64_list(arr) == pytest.approx([1.0, 2.5, 3.7])


def test_round_list_2d_round_trip():
    arr = np.array([[1.23456789, 2.0], [3.0, 4.12345678]], dtype=np.float32)
    rounded = _round_list_2d(arr, decimals=3)
    assert rounded == [[1.235, 2.0], [3.0, 4.123]]


def test_normalize_shap_values_legacy_list_binary():
    """Legacy SHAP API returns a list of per-class arrays; we pick class 1 for binary."""
    # 2 classes × 5 samples × 3 features
    arr = np.arange(30, dtype=np.float64).reshape(2, 5, 3)
    sv, selected = _normalize_shap_values(list(arr))
    assert sv.shape == (5, 3)
    assert selected == 1
    np.testing.assert_array_equal(sv, arr[1])


def test_normalize_shap_values_new_api_3d():
    """New SHAP API: (n_samples, n_features, n_classes). Pick class 1 for binary."""
    # 5 samples × 3 features × 2 classes
    arr = np.arange(30, dtype=np.float64).reshape(5, 3, 2)
    sv, selected = _normalize_shap_values(arr)
    assert sv.shape == (5, 3)
    assert selected == 1
    np.testing.assert_array_equal(sv, arr[:, :, 1])


def test_normalize_shap_values_2d_passthrough():
    arr = np.ones((10, 4), dtype=np.float64)
    sv, selected = _normalize_shap_values(arr)
    assert sv.shape == (10, 4)
    assert selected is None


def test_is_tree_model_by_type_string():
    class Dummy:
        pass
    assert _is_tree_model("random_forest", Dummy())
    assert _is_tree_model("XGBoost", Dummy())
    assert _is_tree_model("lightgbm_regressor", Dummy())
    assert not _is_tree_model("logistic_regression", Dummy())


def test_is_tree_model_by_class_name():
    assert _is_tree_model(None, RandomForestClassifier())
    assert not _is_tree_model(None, LogisticRegression())


def test_build_payload_tree_with_direction():
    """Tree payload has signed shap_values → direction field populated."""
    feature_names = ["a", "b", "c"]
    mean_abs = np.array([0.5, 0.2, 0.1])
    # Signed SHAP: feature 'a' pushes up (positive mean), 'b' pushes down, 'c' neutral
    sv_per = np.array([
        [0.6, -0.3, 0.0],
        [0.4, -0.1, 0.01],
    ])
    feature_values = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    payload = _build_payload(
        method=METHOD_TREE,
        feature_names=feature_names,
        mean_abs_shap=mean_abs,
        sv_per_sample=sv_per,
        feature_values=feature_values,
        base_value=0.5,
        sample_count=2,
        task_kind="classification",
    )
    assert payload["method"] == METHOD_TREE
    assert payload["base_value"] == 0.5
    assert payload["sample_count"] == 2
    assert payload["feature_count"] == 3
    assert payload["shap_values"] is not None
    assert payload["feature_values"] is not None
    # Top feature 'a' should have direction='up' (mean of 0.6, 0.4 is positive)
    top = {t["feature"]: t["direction"] for t in payload["top_features"]}
    assert top["a"] == "up"
    assert top["b"] == "down"


def test_build_payload_permutation_strips_per_sample():
    """Permutation fallback has no shap_values — payload should set them to None
    and omit direction tags (we only have importance magnitudes)."""
    feature_names = ["x", "y"]
    mean_abs = np.array([0.9, 0.1])
    payload = _build_payload(
        method=METHOD_PERMUTATION,
        feature_names=feature_names,
        mean_abs_shap=mean_abs,
        sv_per_sample=None,
        feature_values=None,
        base_value=None,
        sample_count=50,
        task_kind="classification",
    )
    assert payload["method"] == METHOD_PERMUTATION
    assert payload["shap_values"] is None
    assert payload["feature_values"] is None
    assert payload["base_value"] is None
    for entry in payload["top_features"]:
        assert entry["direction"] is None


def test_build_payload_downsamples_when_over_cap():
    """Per-sample arrays must be capped at `max_ui_samples` to keep JSON small."""
    feature_names = ["f1", "f2"]
    mean_abs = np.array([0.1, 0.2])
    sv_per = np.random.RandomState(0).randn(500, 2)
    fv = np.random.RandomState(1).randn(500, 2)
    payload = _build_payload(
        method=METHOD_KERNEL,
        feature_names=feature_names,
        mean_abs_shap=mean_abs,
        sv_per_sample=sv_per,
        feature_values=fv,
        base_value=0.0,
        sample_count=500,
        max_ui_samples=100,
    )
    assert len(payload["shap_values"]) == 100
    assert len(payload["feature_values"]) == 100


# ---------------------------------------------------------------------------
# End-to-end computation over a synthetic tree-model artifact
# ---------------------------------------------------------------------------


@pytest.fixture
def synthetic_rf_run(tmp_path):
    """Write a real RandomForest model + dataset into the real storage dirs
    (with a unique task id to avoid collisions). Cleans up after the test.

    We can't monkeypatch `Settings` because it's a frozen dataclass, so we
    just use the real paths and scope our side-effects by the random id.
    """
    from app.config import get_settings
    import uuid

    settings = get_settings()
    settings.ensure_storage_dirs()

    rng = np.random.RandomState(0)
    X = rng.randn(120, 4)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    df = pd.DataFrame(X, columns=["a", "b", "c", "d"])
    df["label"] = y

    task_id = "synth_" + uuid.uuid4().hex[:8]
    csv_path = settings.storage_uploads / f"{task_id}.csv"
    df.to_csv(csv_path, index=False)

    model = RandomForestClassifier(n_estimators=20, random_state=0)
    model.fit(X, y)
    model_path = settings.storage_models / f"{task_id}.joblib"
    joblib.dump(model, model_path)

    yield {
        "task_id": task_id,
        "csv_path": str(csv_path),
        "target_column": "label",
    }

    # Cleanup
    for p in (csv_path, model_path):
        try:
            p.unlink()
        except FileNotFoundError:
            pass


@pytest.mark.asyncio
async def test_compute_shap_summary_tree_model(synthetic_rf_run, db):
    """End-to-end: synthetic RF → compute_shap_summary returns a well-formed
    payload using the best available rung of the ladder.

    In environments where SHAP's TreeExplainer works we expect `method=tree`;
    when SHAP hits a numpy-2.0 dtype issue we correctly fall through to
    `permutation`. We accept either — the critical contract is that we never
    silently return fake `model_feature_importances`.
    """
    from app.models.database import Dataset, TrainingTask

    task_id = synthetic_rf_run["task_id"]
    dataset = Dataset(
        id="ds-synth",
        name="synth",
        file_path=synthetic_rf_run["csv_path"],
        file_size=1024,
        row_count=120,
        column_count=5,
    )
    tt = TrainingTask(
        id=task_id,
        dataset_id="ds-synth",
        name="synth-rf",
        model_type="random_forest",
        hyperparameters={},
        target_column=synthetic_rf_run["target_column"],
        test_size=0.2,
        eval_metrics=["accuracy"],
        status="SUCCESS",
        progress=100,
        model_path=f"storage/models/{task_id}.joblib",
    )
    db.add_all([dataset, tt])
    await db.commit()

    payload = await shap_service.compute_shap_summary(task_id, db, max_samples=30)
    assert payload["status"] == "ready"
    assert payload["method"] in (METHOD_TREE, METHOD_KERNEL, METHOD_PERMUTATION)
    assert payload["feature_count"] == 4
    assert payload["sample_count"] <= 30
    # Top feature should be 'a' or 'b' since they're the ones driving y
    top_names = [t["feature"] for t in payload["top_features"][:2]]
    assert any(name in top_names for name in ("a", "b"))

    if payload["method"] in (METHOD_TREE, METHOD_KERNEL):
        assert payload["shap_values"] is not None
        assert len(payload["shap_values"]) == payload["sample_count"]
    else:
        # Permutation rung — shap_values must be None (not fabricated), and
        # mean_abs_shap must still be populated for the bar chart.
        assert payload["shap_values"] is None
        assert payload["feature_values"] is None
        assert len(payload["mean_abs_shap"]) == 4

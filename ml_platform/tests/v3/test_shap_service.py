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
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from app.core.model_artifact import fit_tabular_artifact
from app.core.dl_trainer import BaseDLTrainer
from app.core.model_artifact import fit_dl_preprocessing_artifact
from app.services import shap_service
from app.services.shap_service import (
    METHOD_KERNEL,
    METHOD_PERMUTATION,
    METHOD_TREE,
    _build_payload,
    _is_tree_model,
    _normalize_shap_values,
    _normalize_base_value,
    _round_list_2d,
    _to_f64_list,
)


def test_installed_shap_imports_with_supported_numpy():
    import shap

    assert tuple(int(part) for part in shap.__version__.split(".")[:2]) >= (0, 49)


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


def test_normalize_base_value_matches_selected_multiclass_output():
    assert _normalize_base_value([0.1, 0.2, 0.3], selected_class_idx=0) == 0.1


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


class TinyDLTrainer(BaseDLTrainer):
    def build_model(self, input_dim: int, output_dim: int, arch_config: dict):
        return nn.Linear(input_dim, output_dim)


@pytest.fixture
async def synthetic_dl_run(db, tmp_path, monkeypatch):
    from app.models.database import (
        Dataset,
        DLTrainingTask,
        ExperimentRun,
        PlatformExperiment,
        PlatformTask,
    )
    from app.services import dl_shap_adapter

    rng = np.random.RandomState(11)
    features = pd.DataFrame(rng.randn(80, 3), columns=["a", "b", "c"])
    labels = pd.Series((features["a"] + features["b"] > 0).astype(int), name="label")
    stored = features.copy()
    stored["label"] = labels
    dataset_path = tmp_path / "dl-shap.csv"
    stored.to_csv(dataset_path, index=False)

    artifact = fit_dl_preprocessing_artifact(
        features.iloc[:64].reset_index(drop=True),
        labels.iloc[:64].reset_index(drop=True),
        task_kind="classification",
    )
    trainer = TinyDLTrainer()
    trainer.model = trainer.build_model(3, 2, {})
    trainer.num_classes = 2
    trainer.scaler = StandardScaler().fit(artifact.transform_features(features.iloc[:64]))
    model_path = tmp_path / "tiny.pt"
    trainer.save(
        str(model_path),
        input_dim=3,
        task_type="classification",
        feature_columns=artifact.feature_names,
        preprocessing_artifact=artifact,
    )

    monkeypatch.setattr(
        dl_shap_adapter, "get_dl_trainer", lambda _model_type: TinyDLTrainer()
    )

    dataset = Dataset(
        name="dl-shap",
        file_path=str(dataset_path),
        file_size=dataset_path.stat().st_size,
        row_count=len(stored),
        column_count=len(stored.columns),
    )
    db.add(dataset)
    await db.flush()
    dl_task = DLTrainingTask(
        dataset_id=dataset.id,
        target_column="label",
        model_type="mlp_dl",
        task_type="classification",
        train_config={"test_size": 0.2},
        status="SUCCESS",
        progress=100,
        model_path=str(model_path),
    )
    db.add(dl_task)
    await db.flush()
    platform_task = PlatformTask(
        kind="dl_train",
        status="SUCCESS",
        progress=1.0,
        payload_ref=f"dl_train:{dl_task.id}",
    )
    db.add(platform_task)
    await db.flush()
    experiment = PlatformExperiment(
        name="dl-shap-exp",
        strategy_type="baseline",
        objective_metric="accuracy",
        objective_direction="max",
        dataset_id=dataset.id,
        status="DONE",
    )
    db.add(experiment)
    await db.flush()
    run = ExperimentRun(
        experiment_id=experiment.id,
        task_id=platform_task.id,
        params={"model_type": "mlp_dl", "family": "dl"},
        metrics={"accuracy": 0.7},
        status="SUCCESS",
        trial_no=1,
    )
    db.add(run)
    await db.commit()
    return {"task": dl_task, "dataset": dataset, "run": run}


async def test_dl_shap_context_replays_persisted_preprocessing(synthetic_dl_run):
    from app.services.dl_shap_adapter import build_dl_shap_context

    context = build_dl_shap_context(
        synthetic_dl_run["task"],
        synthetic_dl_run["dataset"],
        max_background=12,
        max_samples=6,
    )

    assert context.task_kind == "classification"
    assert context.feature_names == ["a", "b", "c"]
    assert context.X_background.shape == (12, 3)
    assert context.X_sample.shape == (6, 3)
    assert context.model.predict_proba(context.X_sample).shape == (6, 2)


async def test_dl_shap_context_rejects_missing_scaler(synthetic_dl_run):
    from pathlib import Path
    from app.services.dl_shap_adapter import build_dl_shap_context

    scaler_path = Path(str(synthetic_dl_run["task"].model_path) + ".scaler.joblib")
    scaler_path.unlink()

    with pytest.raises(ValueError, match="scaler"):
        build_dl_shap_context(
            synthetic_dl_run["task"],
            synthetic_dl_run["dataset"],
            max_background=12,
            max_samples=6,
        )


async def test_compute_shap_summary_supports_dl_run(synthetic_dl_run, db):
    payload = await shap_service.compute_shap_summary(
        synthetic_dl_run["run"].id, db, max_samples=6
    )

    assert payload["status"] == "ready"
    assert payload["method"] == METHOD_KERNEL
    assert payload["task_kind"] == "classification"
    assert payload["feature_names"] == ["a", "b", "c"]
    assert payload["sample_count"] == 6
    assert len(payload["shap_values"]) == 6


async def test_compute_shap_summary_supports_dl_regression(db, tmp_path, monkeypatch):
    from app.models.database import Dataset, DLTrainingTask
    from app.services import dl_shap_adapter

    rng = np.random.RandomState(19)
    features = pd.DataFrame(rng.randn(80, 3), columns=["a", "b", "c"])
    targets = 2.0 * features["a"] - features["b"] + 0.1
    stored = features.copy()
    stored["target"] = targets
    dataset_path = tmp_path / "dl-regression-shap.csv"
    stored.to_csv(dataset_path, index=False)

    artifact = fit_dl_preprocessing_artifact(
        features.iloc[:64].reset_index(drop=True),
        targets.iloc[:64].reset_index(drop=True),
        task_kind="regression",
    )
    trainer = TinyDLTrainer()
    trainer.model = trainer.build_model(3, 1, {})
    trainer.num_classes = 1
    trainer.scaler = StandardScaler().fit(artifact.transform_features(features.iloc[:64]))
    model_path = tmp_path / "tiny-regression.pt"
    trainer.save(
        str(model_path),
        input_dim=3,
        task_type="regression",
        feature_columns=artifact.feature_names,
        preprocessing_artifact=artifact,
    )
    monkeypatch.setattr(
        dl_shap_adapter, "get_dl_trainer", lambda _model_type: TinyDLTrainer()
    )

    dataset = Dataset(
        name="dl-regression-shap",
        file_path=str(dataset_path),
        file_size=dataset_path.stat().st_size,
        row_count=len(stored),
        column_count=len(stored.columns),
    )
    db.add(dataset)
    await db.flush()
    task = DLTrainingTask(
        dataset_id=dataset.id,
        target_column="target",
        model_type="mlp_dl",
        task_type="regression",
        train_config={"test_size": 0.2},
        status="SUCCESS",
        progress=100,
        model_path=str(model_path),
    )
    db.add(task)
    await db.commit()

    payload = await shap_service.compute_shap_summary(task.id, db, max_samples=6)

    assert payload["method"] == METHOD_KERNEL
    assert payload["task_kind"] == "regression"
    assert payload["class_index"] is None
    assert payload["sample_count"] == 6
    assert len(payload["shap_values"]) == 6


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

    The pinned SHAP version supports NumPy 2.x, so a tree model must use the
    tree rung rather than silently degrading to permutation importance.
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
    assert payload["method"] == METHOD_TREE
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


@pytest.mark.asyncio
async def test_compute_shap_summary_with_tabular_artifact(db):
    from app.config import get_settings
    from app.models.database import Dataset, TrainingTask
    import uuid

    settings = get_settings()
    settings.ensure_storage_dirs()
    suffix = uuid.uuid4().hex[:8]
    task_id = f"artifact_{suffix}"
    dataset_id = f"ds_artifact_{suffix}"

    rng = np.random.RandomState(7)
    numeric = rng.randn(120, 2)
    frame = pd.DataFrame(
        {
            "signal": numeric[:, 0],
            "load": numeric[:, 1],
            "kind": np.where(numeric[:, 0] >= 0, "hot", "cold"),
        }
    )
    labels = pd.Series(
        np.where(numeric[:, 0] + numeric[:, 1] >= 0, "fault", "normal")
    )
    stored = frame.copy()
    stored["label"] = labels
    csv_path = settings.storage_uploads / f"{task_id}.csv"
    model_path = settings.storage_models / f"{task_id}.joblib"
    stored.to_csv(csv_path, index=False)

    X_train, _, y_train, _ = train_test_split(
        frame,
        labels,
        test_size=0.2,
        random_state=42,
        stratify=labels,
    )
    artifact = fit_tabular_artifact(
        RandomForestClassifier(n_estimators=20, random_state=0),
        X_train.reset_index(drop=True),
        y_train.reset_index(drop=True),
        task_kind="classification",
    )
    joblib.dump(artifact, model_path)

    dataset = Dataset(
        id=dataset_id,
        name="artifact-shap",
        file_path=str(csv_path),
        file_size=1024,
        row_count=len(stored),
        column_count=len(stored.columns),
    )
    task = TrainingTask(
        id=task_id,
        dataset_id=dataset_id,
        name="artifact-rf",
        model_type="random_forest",
        hyperparameters={},
        target_column="label",
        test_size=0.2,
        eval_metrics=["accuracy"],
        status="SUCCESS",
        progress=100,
        model_path=f"storage/models/{task_id}.joblib",
    )
    db.add_all([dataset, task])
    await db.commit()

    try:
        payload = await shap_service.compute_shap_summary(task_id, db, max_samples=30)
        assert payload["status"] == "ready"
        assert payload["method"] in (METHOD_TREE, METHOD_KERNEL, METHOD_PERMUTATION)
        assert payload["feature_count"] == 3
        assert "kind" in {item["feature"] for item in payload["top_features"]}
    finally:
        for path in (csv_path, model_path):
            path.unlink(missing_ok=True)

"""Visualization service — computes data for charts and plots."""

import json
import logging
from dataclasses import dataclass
from typing import Any

import joblib
import numpy as np
import pandas as pd
from fastapi import HTTPException
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.model_selection import train_test_split
from sqlalchemy import select

from app.config import get_settings
from app.models.database import (
    AsyncSession,
    Dataset,
    ExperimentRun,
    ModelingTask,
    PlatformExperiment,
    PlatformTask,
    TrainingTask,
)
from app.services.prediction_service import load_dataframe, prepare_training_frame
from app.utils.storage_paths import resolve_runtime_path

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Task façade — lets us keep the same get_* function bodies whether the
# backing row lives in TrainingTask (legacy), PlatformTask→TrainingTask
# (v3 wrapper), or ExperimentRun (V3-native, no legacy row).  Without this
# façade, requesting a viz chart for a V3 Run returns 404 "Training task
# not found" even though the model file + metrics log exist on disk.
# ---------------------------------------------------------------------------


@dataclass
class _TaskFacade:
    """Duck-typed subset of TrainingTask used by this file."""
    id: str
    model_type: str | None
    model_path: str | None
    target_column: str | None
    test_size: float
    status: str
    dataset_id: str | None
    # Kept for parity with TrainingTask, not used directly
    result_metrics: dict | None = None


_LEGACY_REGRESSOR_MODEL_TYPES = {
    "linear_regression",
    "random_forest_regressor",
    "xgboost_regressor",
    "lightgbm_regressor",
    "mlp_dl_regressor",
    "lstm_regressor",
    "tcn_regressor",
}


def _is_regressor(model_type: str | None) -> bool:
    if not model_type:
        return False
    lower = model_type.lower()
    if lower in _LEGACY_REGRESSOR_MODEL_TYPES:
        return True
    # Heuristic suffixes used across ML + DL registries
    return lower.endswith("_regressor") or lower.endswith("_regression")


def _load_model(model_path: str):
    """Load a saved sklearn model."""
    path = resolve_runtime_path(model_path)
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Model file not found: {model_path}")
    return joblib.load(path)


def _load_and_split_data(file_path: str, target_column: str, test_size: float = 0.2):
    """Load dataset and split into train/test, mirroring training logic."""
    df = load_dataframe(file_path)
    X, y, _, target_encoder = prepare_training_frame(df, target_column)

    X_train, X_test, y_train, y_test = train_test_split(
        X.values, y.values, test_size=test_size, random_state=42, stratify=y.values
    )

    feature_names = list(X.columns)
    class_labels = (
        target_encoder.classes_.tolist()
        if target_encoder is not None
        else sorted([str(c) for c in np.unique(y.values)])
    )

    return X_train, X_test, y_train, y_test, feature_names, class_labels


def _pick_test_size(container: Any, default: float = 0.2) -> float:
    """Safely coerce a params/snapshot/etc. dict's test_size into a float."""
    if not container:
        return default
    try:
        v = container.get("test_size") if hasattr(container, "get") else None
    except Exception:
        return default
    if v is None:
        return default
    try:
        f = float(v)
        if 0.05 <= f <= 0.5:
            return f
    except (TypeError, ValueError):
        pass
    return default


def _pick_model_type_from_metrics(task_id: str) -> str | None:
    """Peek at storage/logs/{task_id}_metrics.json for the trainer-recorded model_type."""
    settings = get_settings()
    mf = settings.storage_logs / f"{task_id}_metrics.json"
    if not mf.exists():
        return None
    try:
        with open(mf, "r") as f:
            data = json.load(f)
        return data.get("model_type") or None
    except Exception:
        return None


_LOG_DATASET_RE = None
_LOG_TARGET_RE = None


def _parse_legacy_log_context(task_id: str) -> dict:
    """Fallback: parse storage/logs/{task_id}.log for dataset= / target= so orphaned
    legacy tasks (TrainingTask row purged, model file still on disk) can still
    resolve.  Returns {} if the log file is missing or unparseable."""
    import re
    settings = get_settings()
    log_path = settings.storage_logs / f"{task_id}.log"
    if not log_path.exists():
        return {}
    try:
        text = log_path.read_text(errors="ignore")
    except Exception:
        return {}
    out: dict[str, str] = {}
    m = re.search(r"dataset=([^\s|]+)", text)
    if m:
        out["dataset_path"] = m.group(1).strip()
    m = re.search(r"target=([^\s|]+)", text)
    if m:
        out["target_column"] = m.group(1).strip()
    return out


async def _resolve_dataset_by_path(db: AsyncSession, file_path: str) -> Dataset | None:
    """Look up a Dataset by exact file_path match (used for orphan recovery)."""
    result = await db.execute(select(Dataset).where(Dataset.file_path == file_path))
    return result.scalar_one_or_none()


async def _synthesize_facade_from_run(
    run: ExperimentRun, db: AsyncSession
) -> tuple[_TaskFacade, Dataset | None]:
    """Build a TaskFacade for a V3 ExperimentRun.

    V3 runs don't carry model_path/target/test_size on the ORM row — those live
    on the parent PlatformExperiment → ModelingTask + the plan snapshot.  We
    reconstitute them so the legacy viz code can run unchanged.
    """
    params = run.params or {}
    # Model file keyed by run.id OR run.task_id depending on trainer family.
    settings = get_settings()
    candidate_ids = [run.id]
    if run.task_id:
        candidate_ids.append(run.task_id)
    # Also try payload_ref's legacy id if linked PlatformTask still references one
    legacy_id: str | None = None
    if run.task_id:
        pt = (await db.execute(
            select(PlatformTask).where(PlatformTask.id == run.task_id)
        )).scalar_one_or_none()
        if pt and pt.payload_ref and ":" in pt.payload_ref:
            legacy_id = pt.payload_ref.split(":", 1)[1]
            if legacy_id:
                candidate_ids.append(legacy_id)

    model_path: str | None = None
    for cid in candidate_ids:
        candidate = settings.storage_models / f"{cid}.joblib"
        if candidate.exists():
            model_path = f"storage/models/{cid}.joblib"
            break

    # Parent experiment → dataset; ModelingTask → target_column + task_type
    exp = (await db.execute(
        select(PlatformExperiment).where(PlatformExperiment.id == run.experiment_id)
    )).scalar_one_or_none()
    dataset: Dataset | None = None
    target_column: str | None = None
    if exp is not None:
        if exp.dataset_id:
            dataset = (await db.execute(
                select(Dataset).where(Dataset.id == exp.dataset_id)
            )).scalar_one_or_none()
        if exp.modeling_task_id:
            mt = (await db.execute(
                select(ModelingTask).where(ModelingTask.id == exp.modeling_task_id)
            )).scalar_one_or_none()
            if mt:
                target_column = mt.target_column
                if dataset is None and mt.dataset_id:
                    dataset = (await db.execute(
                        select(Dataset).where(Dataset.id == mt.dataset_id)
                    )).scalar_one_or_none()

    # Model type from params, falling back to metrics log
    model_type = params.get("model_type") or params.get("model") if isinstance(params, dict) else None
    if not model_type:
        for cid in candidate_ids:
            mt_guess = _pick_model_type_from_metrics(cid)
            if mt_guess:
                model_type = mt_guess
                break

    test_size = _pick_test_size(params)

    facade = _TaskFacade(
        id=run.id,
        model_type=model_type,
        model_path=model_path,
        target_column=target_column,
        test_size=test_size,
        status="SUCCESS" if run.status == "SUCCESS" else (run.status or "UNKNOWN"),
        dataset_id=(exp.dataset_id if exp else None),
        result_metrics=run.metrics or None,
    )
    return facade, dataset


async def _synthesize_facade_from_orphan(
    task_id: str, db: AsyncSession
) -> tuple[_TaskFacade, Dataset | None] | None:
    """Recover from the on-disk artifacts when the DB row is gone.

    Used when:
      - TrainingTask row has been purged
      - No ExperimentRun references this id
      - PlatformTask may or may not exist (payload_ref=train:<id>)
    Relies on the per-task log file to recover dataset_path + target_column.
    """
    settings = get_settings()
    model_file = settings.storage_models / f"{task_id}.joblib"
    if not model_file.exists():
        return None
    ctx = _parse_legacy_log_context(task_id)
    dataset_path = ctx.get("dataset_path")
    target_column = ctx.get("target_column")
    if not dataset_path or not target_column:
        return None

    dataset = await _resolve_dataset_by_path(db, dataset_path)
    if dataset is None:
        return None

    model_type = _pick_model_type_from_metrics(task_id)
    facade = _TaskFacade(
        id=task_id,
        model_type=model_type,
        model_path=f"storage/models/{task_id}.joblib",
        target_column=target_column,
        test_size=0.2,
        status="SUCCESS",
        dataset_id=dataset.id,
        result_metrics=None,
    )
    return facade, dataset


async def _get_task_and_dataset(task_id: str, db: AsyncSession):
    """Multi-source resolver: TrainingTask → V3 ExperimentRun → on-disk orphan.

    The viz code was originally written for TrainingTask-only lookups, which
    broke two real-world states:
      - V3-native runs that never created a TrainingTask row;
      - legacy tasks whose TrainingTask was purged but whose model+log files
        still live in storage/.
    This resolver recovers both cases so the chart endpoints behave the same
    way `learning_curve` (which reads directly from disk) already does.
    """
    # --- Tier 1: legacy TrainingTask direct -------------------------------
    result = await db.execute(select(TrainingTask).where(TrainingTask.id == task_id))
    task = result.scalar_one_or_none()
    if task is not None:
        if task.status != "SUCCESS":
            raise HTTPException(status_code=400, detail=f"Task not completed (status={task.status})")
        if not task.model_path:
            raise HTTPException(status_code=400, detail="No model saved for this task")
        dataset = (await db.execute(
            select(Dataset).where(Dataset.id == task.dataset_id)
        )).scalar_one_or_none()
        if dataset is None:
            raise HTTPException(status_code=404, detail="Dataset not found")
        return task, dataset

    # --- Tier 2: V3 ExperimentRun synthesis -------------------------------
    run = (await db.execute(
        select(ExperimentRun).where(ExperimentRun.id == task_id)
    )).scalar_one_or_none()
    if run is None:
        # task_id might be the PlatformTask.id instead of the run id
        pt = (await db.execute(
            select(PlatformTask).where(PlatformTask.id == task_id)
        )).scalar_one_or_none()
        if pt is not None:
            run = (await db.execute(
                select(ExperimentRun).where(ExperimentRun.task_id == pt.id)
            )).scalar_one_or_none()
            # Legacy shim: PlatformTask.payload_ref == "train:<legacy_tt_id>"
            if run is None and pt.payload_ref and pt.payload_ref.startswith(("train:", "dl_train:")):
                legacy_id = pt.payload_ref.split(":", 1)[1]
                legacy_tt = (await db.execute(
                    select(TrainingTask).where(TrainingTask.id == legacy_id)
                )).scalar_one_or_none()
                if legacy_tt is not None:
                    return await _get_task_and_dataset(legacy_id, db)

    if run is not None:
        if run.status != "SUCCESS":
            raise HTTPException(
                status_code=400,
                detail=f"Run not completed (status={run.status or 'UNKNOWN'})",
            )
        facade, dataset = await _synthesize_facade_from_run(run, db)
        if not facade.model_path:
            raise HTTPException(status_code=404, detail="Model artifact not found for this run")
        if not facade.target_column:
            raise HTTPException(status_code=400, detail="Target column unavailable for this run")
        if dataset is None:
            raise HTTPException(status_code=404, detail="Dataset not found for this run")
        return facade, dataset

    # --- Tier 3: orphan recovery from on-disk artifacts -------------------
    orphan = await _synthesize_facade_from_orphan(task_id, db)
    if orphan is not None:
        facade, dataset = orphan
        if dataset is None:
            raise HTTPException(status_code=404, detail="Dataset not found for this task")
        return facade, dataset

    # Also honour PlatformTask.payload_ref pointing at an orphan legacy id
    pt2 = (await db.execute(
        select(PlatformTask).where(
            PlatformTask.payload_ref.in_([f"train:{task_id}", f"dl_train:{task_id}"])
        )
    )).scalar_one_or_none()
    if pt2 is not None:
        # Same task id — we already tried orphan recovery above; if we're here
        # the log file was missing.  Fall through to 404 so callers get the
        # existing "Training task not found" error message (preserves clients
        # that special-case this text).
        pass

    raise HTTPException(status_code=404, detail="Training task not found")


async def get_confusion_matrix(task_id: str, db: AsyncSession, normalize: bool = False) -> dict:
    """Compute confusion matrix for a completed training task."""
    task, dataset = await _get_task_and_dataset(task_id, db)
    if _is_regressor(task.model_type):
        raise HTTPException(
            status_code=400,
            detail="该任务为回归任务，不支持混淆矩阵。请查看残差图或预测-真实值散点。",
        )
    model = _load_model(task.model_path)
    _, X_test, _, y_test, _, class_labels = _load_and_split_data(
        dataset.file_path, task.target_column, task.test_size
    )

    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    if normalize:
        cm = cm.astype(float)
        row_sums = cm.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # avoid division by zero
        cm = np.round(cm / row_sums, 4)

    return {
        "labels": class_labels,
        "matrix": cm.tolist(),
        "normalized": normalize,
    }


async def get_roc_curve(task_id: str, db: AsyncSession) -> dict:
    """Compute ROC curve data for a completed training task."""
    task, dataset = await _get_task_and_dataset(task_id, db)
    if _is_regressor(task.model_type):
        raise HTTPException(
            status_code=400,
            detail="该任务为回归任务，不支持 ROC 曲线。请查看残差图或预测-真实值散点。",
        )
    model = _load_model(task.model_path)
    _, X_test, _, y_test, _, class_labels = _load_and_split_data(
        dataset.file_path, task.target_column, task.test_size
    )

    if not hasattr(model, 'predict_proba'):
        raise HTTPException(status_code=400, detail="Model does not support probability prediction for ROC")

    y_proba = model.predict_proba(X_test)

    # Binary classification
    if y_proba.shape[1] == 2:
        fpr, tpr, thresholds = roc_curve(y_test, y_proba[:, 1])
        roc_auc = float(auc(fpr, tpr))
        return {
            "fpr": [round(float(x), 4) for x in fpr],
            "tpr": [round(float(x), 4) for x in tpr],
            "auc": round(roc_auc, 4),
            "thresholds": [round(float(x), 4) if np.isfinite(x) else 1.0 for x in thresholds],
        }

    # Multi-class: one-vs-rest
    from sklearn.preprocessing import label_binarize
    classes = np.unique(y_test)
    y_test_bin = label_binarize(y_test, classes=classes)
    curves = []
    for i, cls in enumerate(classes):
        fpr_i, tpr_i, _ = roc_curve(y_test_bin[:, i], y_proba[:, i])
        auc_i = float(auc(fpr_i, tpr_i))
        curves.append({
            "class": class_labels[i] if i < len(class_labels) else str(cls),
            "fpr": [round(float(x), 4) for x in fpr_i],
            "tpr": [round(float(x), 4) for x in tpr_i],
            "auc": round(auc_i, 4),
        })
    return {"curves": curves, "multiclass": True}


async def get_feature_importance(task_id: str, db: AsyncSession) -> dict:
    """Get feature importance from the trained model."""
    task, dataset = await _get_task_and_dataset(task_id, db)
    model = _load_model(task.model_path)
    # Feature importance only needs column names, not a stratified split.
    # Using _load_task_model_data (no stratify) also handles regression tasks
    # whose continuous targets would crash stratify=y.values.
    _, _, _, _, _, feature_names = _load_task_model_data(
        task, dataset, task.test_size
    )

    importance = None

    # Tree-based models
    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
    # Linear models
    elif hasattr(model, 'coef_'):
        coef = model.coef_
        if coef.ndim > 1:
            importance = np.abs(coef).mean(axis=0)
        else:
            importance = np.abs(coef)
    else:
        raise HTTPException(status_code=400, detail="Model does not provide feature importance")

    # Sort by importance descending
    indices = np.argsort(importance)[::-1]
    sorted_features = [feature_names[i] for i in indices]
    sorted_importance = [round(float(importance[i]), 6) for i in indices]

    return {
        "features": sorted_features,
        "importance": sorted_importance,
    }


async def _resolve_metrics_candidate_ids(task_id: str, db: AsyncSession) -> list[str]:
    """Return every id worth probing for a {id}_metrics.json file.

    Log files are always keyed by the legacy trainer task id.  For V3 runs that
    id only lives on PlatformTask.payload_ref, so we have to walk the chain.
    """
    candidates: list[str] = [task_id]
    # TrainingTask.id might be the legacy id directly — file keyed by same id.
    # ExperimentRun.id → PlatformTask.payload_ref → legacy id.
    run = (await db.execute(
        select(ExperimentRun).where(ExperimentRun.id == task_id)
    )).scalar_one_or_none()
    if run is None:
        # Or task_id might be the PlatformTask.id itself
        pt_by_id = (await db.execute(
            select(PlatformTask).where(PlatformTask.id == task_id)
        )).scalar_one_or_none()
        if pt_by_id is not None:
            run = (await db.execute(
                select(ExperimentRun).where(ExperimentRun.task_id == pt_by_id.id)
            )).scalar_one_or_none()
            if pt_by_id.payload_ref and ":" in pt_by_id.payload_ref:
                candidates.append(pt_by_id.payload_ref.split(":", 1)[1])
    if run is not None and run.task_id:
        candidates.append(run.task_id)
        pt = (await db.execute(
            select(PlatformTask).where(PlatformTask.id == run.task_id)
        )).scalar_one_or_none()
        if pt and pt.payload_ref and ":" in pt.payload_ref:
            candidates.append(pt.payload_ref.split(":", 1)[1])
    # Also honour "task_id is a legacy id whose PlatformTask payload_ref points to it"
    pt_ref = (await db.execute(
        select(PlatformTask).where(PlatformTask.payload_ref.in_(
            [f"train:{task_id}", f"dl_train:{task_id}"]
        ))
    )).scalar_one_or_none()
    if pt_ref is not None:
        candidates.append(pt_ref.id)
    # Deduplicate while preserving order
    seen: set[str] = set()
    ordered: list[str] = []
    for c in candidates:
        if c and c not in seen:
            seen.add(c)
            ordered.append(c)
    return ordered


async def get_learning_curve(task_id: str, db: AsyncSession) -> dict:
    """Get learning curve data from metrics log."""
    settings = get_settings()

    metrics_file = None
    resolved_id = task_id
    for cid in await _resolve_metrics_candidate_ids(task_id, db):
        candidate = settings.storage_logs / f"{cid}_metrics.json"
        if candidate.exists():
            metrics_file = candidate
            resolved_id = cid
            break

    if metrics_file is None:
        raise HTTPException(status_code=404, detail="No metrics data found")

    with open(metrics_file, "r") as f:
        data = json.load(f)

    steps = data.get("steps", [])
    if not steps:
        raise HTTPException(status_code=404, detail="No training steps recorded")

    return {
        "task_id": task_id,
        "resolved_id": resolved_id,
        "model_type": data.get("model_type", ""),
        "steps": steps,
    }


async def get_shap_summary(task_id: str, db: AsyncSession, max_samples: int = 100) -> dict:
    """Compute SHAP values for model explainability."""
    task, dataset = await _get_task_and_dataset(task_id, db)
    model = _load_model(task.model_path)

    # Use the non-stratified split so regression tasks also work here.
    _, X_train, X_test, _, _, feature_names = _load_task_model_data(
        task, dataset, task.test_size
    )

    # Limit samples
    if len(X_test) > max_samples:
        indices = np.random.RandomState(42).choice(len(X_test), max_samples, replace=False)
        X_sample = X_test[indices]
    else:
        X_sample = X_test

    try:
        import shap

        # Choose appropriate explainer
        model_type = (task.model_type or "").lower()
        tree_prefixes = ("random_forest", "xgboost", "lightgbm")
        if any(model_type.startswith(p) for p in tree_prefixes):
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_sample)
        else:
            # Use KernelExplainer — regressors have .predict, classifiers
            # have both .predict and .predict_proba.  Prefer proba when
            # available since that yields per-class SHAP values.
            predict_fn = getattr(model, "predict_proba", None) or model.predict
            background = shap.sample(pd.DataFrame(X_train, columns=feature_names), min(50, len(X_train)))
            explainer = shap.KernelExplainer(predict_fn, background)
            shap_values = explainer.shap_values(X_sample)

        # Handle multi-output (binary classification returns list of 2)
        if isinstance(shap_values, list):
            # Take the positive class (index 1) for binary, or first class for multi
            shap_values = shap_values[1] if len(shap_values) == 2 else shap_values[0]

        # Compute mean absolute SHAP per feature
        mean_abs_shap = np.abs(shap_values).mean(axis=0).tolist()

        return {
            "features": feature_names,
            "mean_abs_shap": [round(float(v), 6) for v in mean_abs_shap],
            "shap_values": [[round(float(v), 4) for v in row] for row in shap_values.tolist()],
            "feature_values": [[round(float(v), 4) for v in row] for row in X_sample.tolist()],
            "num_samples": len(X_sample),
        }

    except ImportError:
        raise HTTPException(status_code=503, detail="SHAP library not installed")
    except Exception as e:
        logger.error("SHAP computation failed: %s", str(e))
        raise HTTPException(status_code=500, detail=f"SHAP computation failed: {str(e)}")


# ---------------------------------------------------------------------------
# Regression visualization helpers
# ---------------------------------------------------------------------------

def _load_task_model_data(task, dataset, test_size: float = 0.2):
    """Load model + split data for regression tasks (no stratify on continuous target).

    This is a separate helper from _load_and_split_data which uses stratify=y.values
    (only valid for classification). Regression targets are continuous so stratify
    must be omitted.
    """
    df = load_dataframe(dataset.file_path)
    X, y, _, target_encoder = prepare_training_frame(df, task.target_column)

    # No stratify — regression targets are continuous
    X_train, X_test, y_train, y_test = train_test_split(
        X.values, y.values, test_size=test_size, random_state=42
    )

    model = _load_model(task.model_path)
    feature_names = list(X.columns)
    return model, X_train, X_test, y_train, y_test, feature_names


async def get_residual_plot(task_id: str, db: AsyncSession) -> dict:
    """Return residuals (y_true - y_pred) and predicted values for residual plot."""
    task, dataset = await _get_task_and_dataset(task_id, db)
    model, _, X_test, _, y_test, _ = _load_task_model_data(task, dataset, task.test_size)

    y_pred = model.predict(X_test)
    residuals = (y_test - y_pred).tolist()
    y_pred_list = y_pred.tolist()

    return {
        "task_id": task_id,
        "predicted": [round(float(v), 4) for v in y_pred_list],
        "residuals": [round(float(v), 4) for v in residuals],
        "mean_residual": round(float(np.mean(residuals)), 4),
        "std_residual": round(float(np.std(residuals)), 4),
    }


async def get_predicted_vs_actual(task_id: str, db: AsyncSession) -> dict:
    """Return predicted vs actual values for scatter plot."""
    task, dataset = await _get_task_and_dataset(task_id, db)
    model, _, X_test, _, y_test, _ = _load_task_model_data(task, dataset, task.test_size)

    y_pred = model.predict(X_test)

    return {
        "task_id": task_id,
        "actual": [round(float(v), 4) for v in y_test.tolist()],
        "predicted": [round(float(v), 4) for v in y_pred.tolist()],
    }

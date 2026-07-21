"""Bounded PyTorch inference adapter for the shared Kernel SHAP pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from app.core.dl_registry import get_dl_trainer
from app.models.database import Dataset, DLTrainingTask
from app.services.dl_service import _outer_split
from app.utils.storage_paths import resolve_runtime_path


class _DLInferenceAdapter:
    def __init__(self, trainer: Any, task_kind: str):
        self._trainer = trainer
        self._task_kind = task_kind

    def predict(self, values) -> np.ndarray:
        predictions, _ = self._trainer.predict(
            np.asarray(values, dtype=np.float32), self._task_kind
        )
        return np.asarray(predictions)


class _DLClassificationAdapter(_DLInferenceAdapter):
    def predict_proba(self, values) -> np.ndarray:
        _, probabilities = self._trainer.predict(
            np.asarray(values, dtype=np.float32), self._task_kind
        )
        if probabilities is None:
            raise ValueError("DL classifier did not return probabilities")
        return np.asarray(probabilities)


@dataclass(frozen=True)
class DLShapContext:
    model: Any
    X_background: np.ndarray
    X_sample: np.ndarray
    y_sample: np.ndarray
    feature_names: list[str]
    task_kind: str


def _bounded_sample(frame, size: int):
    count = min(max(1, int(size)), len(frame))
    return frame.sample(n=count, random_state=42)


def build_dl_shap_context(
    task: DLTrainingTask,
    dataset: Dataset,
    *,
    max_background: int,
    max_samples: int,
) -> DLShapContext:
    """Load a DL checkpoint and replay its persisted train-time transforms."""
    model_path = resolve_runtime_path(task.model_path)
    trainer = get_dl_trainer(task.model_type)
    metadata = trainer.load_for_inference(str(model_path))
    artifact = metadata.get("preprocessing_artifact")
    if artifact is None:
        raise ValueError("DL SHAP requires a persisted preprocessing sidecar")
    if trainer.scaler is None:
        raise ValueError("DL SHAP requires a persisted scaler sidecar")

    configured_test_size = (task.train_config or {}).get("test_size", 0.2)
    raw_train, raw_holdout, _, raw_y_holdout, split_task_kind = _outer_split(
        dataset.file_path,
        task.target_column,
        float(configured_test_size),
        task.task_type or metadata.get("task_type", "classification"),
    )
    task_kind = metadata.get("task_type") or split_task_kind

    background_frame = _bounded_sample(raw_train, max_background)
    sample_frame = _bounded_sample(raw_holdout, max_samples)
    X_background = artifact.transform_features(background_frame)
    X_sample = artifact.transform_features(sample_frame)
    y_sample = artifact.encode_target(raw_y_holdout.loc[sample_frame.index])

    adapter_class = (
        _DLClassificationAdapter if task_kind == "classification" else _DLInferenceAdapter
    )
    return DLShapContext(
        model=adapter_class(trainer, task_kind),
        X_background=np.asarray(X_background, dtype=np.float32),
        X_sample=np.asarray(X_sample, dtype=np.float32),
        y_sample=np.asarray(y_sample),
        feature_names=list(artifact.feature_names),
        task_kind=task_kind,
    )

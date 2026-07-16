"""Versioned preprocessing artifacts for tabular ML and DL models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_object_dtype, is_string_dtype
from sklearn.base import clone
from sklearn.preprocessing import LabelEncoder

ARTIFACT_VERSION = 1
_MISSING_CATEGORY = "__ML_PLATFORM_MISSING__"


@dataclass
class TabularPreprocessor:
    feature_columns: list[str] = field(default_factory=list)
    numeric_fill_values: dict[str, float] = field(default_factory=dict)
    categorical_mappings: dict[str, dict[str, int]] = field(default_factory=dict)
    unknown_category_value: int = -1

    def fit(self, frame: pd.DataFrame) -> "TabularPreprocessor":
        if not isinstance(frame, pd.DataFrame):
            raise TypeError("TabularPreprocessor.fit requires a pandas DataFrame")
        if frame.empty:
            raise ValueError("Cannot fit preprocessing on an empty frame")

        self.feature_columns = [str(column) for column in frame.columns]
        self.numeric_fill_values = {}
        self.categorical_mappings = {}

        for column in self.feature_columns:
            series = frame[column]
            if _is_categorical(series):
                values = series.fillna(_MISSING_CATEGORY).astype(str)
                classes = sorted(values.unique().tolist())
                self.categorical_mappings[column] = {
                    value: index for index, value in enumerate(classes)
                }
            else:
                numeric = pd.to_numeric(series, errors="coerce")
                median = numeric.median()
                self.numeric_fill_values[column] = (
                    float(median) if not pd.isna(median) else 0.0
                )
        return self

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        if not self.feature_columns:
            raise ValueError("TabularPreprocessor has not been fitted")
        if not isinstance(frame, pd.DataFrame):
            frame = pd.DataFrame(frame)

        missing_columns = [
            column for column in self.feature_columns if column not in frame.columns
        ]
        if missing_columns:
            raise ValueError(
                f"Missing required feature columns: {', '.join(missing_columns)}"
            )

        transformed = pd.DataFrame(index=frame.index)
        for column in self.feature_columns:
            if column in self.categorical_mappings:
                mapping = self.categorical_mappings[column]
                values = frame[column].fillna(_MISSING_CATEGORY).astype(str)
                transformed[column] = values.map(mapping).fillna(
                    self.unknown_category_value
                ).astype(float)
            else:
                values = pd.to_numeric(frame[column], errors="coerce")
                transformed[column] = values.fillna(
                    self.numeric_fill_values[column]
                ).astype(float)

        return transformed[self.feature_columns].to_numpy(dtype=float)

    def fit_transform(self, frame: pd.DataFrame) -> np.ndarray:
        return self.fit(frame).transform(frame)


@dataclass
class TabularModelArtifact:
    estimator: Any
    preprocessor: TabularPreprocessor
    task_kind: str
    target_encoder: LabelEncoder | None = None
    artifact_version: int = ARTIFACT_VERSION

    @property
    def feature_names(self) -> list[str]:
        return list(self.preprocessor.feature_columns)

    @property
    def class_labels(self) -> list[Any]:
        if self.target_encoder is not None:
            return self.target_encoder.classes_.tolist()
        classes = getattr(self.estimator, "classes_", [])
        return classes.tolist() if hasattr(classes, "tolist") else list(classes)

    def transform_features(self, frame: pd.DataFrame) -> np.ndarray:
        self._validate_version()
        return self.preprocessor.transform(frame)

    def encode_target(self, values) -> np.ndarray:
        self._validate_version()
        series = pd.Series(values)
        if self.target_encoder is not None:
            return self.target_encoder.transform(series.astype(str))
        return series.to_numpy()

    def predict_encoded(self, frame: pd.DataFrame) -> np.ndarray:
        return np.asarray(self.estimator.predict(self.transform_features(frame)))

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        predictions = self.predict_encoded(frame)
        if self.target_encoder is not None:
            return self.target_encoder.inverse_transform(predictions.astype(int))
        return predictions

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        if not hasattr(self.estimator, "predict_proba"):
            raise AttributeError("Underlying estimator does not support predict_proba")
        return np.asarray(
            self.estimator.predict_proba(self.transform_features(frame))
        )

    def _validate_version(self) -> None:
        if self.artifact_version != ARTIFACT_VERSION:
            raise ValueError(
                "Unsupported tabular model artifact version: "
                f"{self.artifact_version}; expected {ARTIFACT_VERSION}"
            )


@dataclass
class DLPreprocessingArtifact:
    """Persisted tabular preprocessing state shared by DL train and inference."""

    preprocessor: TabularPreprocessor
    task_kind: str
    target_encoder: LabelEncoder | None = None
    artifact_version: int = ARTIFACT_VERSION

    @property
    def feature_names(self) -> list[str]:
        return list(self.preprocessor.feature_columns)

    @property
    def class_labels(self) -> list[Any]:
        if self.target_encoder is None:
            return []
        return self.target_encoder.classes_.tolist()

    def transform_features(self, frame: pd.DataFrame) -> np.ndarray:
        self._validate_version()
        return self.preprocessor.transform(frame).astype(np.float32)

    def encode_target(self, values) -> np.ndarray:
        self._validate_version()
        series = pd.Series(values)
        if self.target_encoder is not None:
            return self.target_encoder.transform(series).astype(np.int64)
        return pd.to_numeric(series, errors="raise").to_numpy(dtype=np.float32)

    def decode_predictions(self, values) -> np.ndarray:
        self._validate_version()
        predictions = np.asarray(values)
        if self.target_encoder is not None:
            return self.target_encoder.inverse_transform(predictions.astype(int))
        return predictions

    def _validate_version(self) -> None:
        if self.artifact_version != ARTIFACT_VERSION:
            raise ValueError(
                "Unsupported DL preprocessing artifact version: "
                f"{self.artifact_version}; expected {ARTIFACT_VERSION}"
            )


def fit_dl_preprocessing_artifact(
    X: pd.DataFrame,
    y,
    *,
    task_kind: str,
) -> DLPreprocessingArtifact:
    if task_kind not in {"classification", "regression"}:
        raise ValueError(f"Unsupported task kind: {task_kind}")

    preprocessor = TabularPreprocessor().fit(X)
    target_encoder: LabelEncoder | None = None
    if task_kind == "classification":
        target_encoder = LabelEncoder().fit(pd.Series(y))

    return DLPreprocessingArtifact(
        preprocessor=preprocessor,
        task_kind=task_kind,
        target_encoder=target_encoder,
    )


def fit_tabular_artifact(
    estimator,
    X: pd.DataFrame,
    y,
    *,
    task_kind: str,
) -> TabularModelArtifact:
    if task_kind not in {"classification", "regression"}:
        raise ValueError(f"Unsupported task kind: {task_kind}")

    preprocessor = TabularPreprocessor()
    transformed_X = preprocessor.fit_transform(X)
    target = pd.Series(y).reset_index(drop=True)
    target_encoder: LabelEncoder | None = None

    if task_kind == "classification" and _is_categorical(target):
        target_encoder = LabelEncoder()
        transformed_y = target_encoder.fit_transform(target.astype(str))
    else:
        transformed_y = target.to_numpy()

    fitted_estimator = clone(estimator)
    fitted_estimator.fit(transformed_X, transformed_y)
    return TabularModelArtifact(
        estimator=fitted_estimator,
        preprocessor=preprocessor,
        task_kind=task_kind,
        target_encoder=target_encoder,
    )


def is_tabular_artifact(value) -> bool:
    return isinstance(value, TabularModelArtifact)


def _is_categorical(series: pd.Series) -> bool:
    dtype = series.dtype
    return bool(
        is_object_dtype(dtype)
        or is_string_dtype(dtype)
        or is_bool_dtype(dtype)
        or isinstance(dtype, pd.CategoricalDtype)
    )

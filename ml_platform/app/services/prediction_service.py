"""Prediction helpers for saved models and reusable preprocessing logic."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from fastapi import HTTPException
from sklearn.preprocessing import LabelEncoder
from sqlalchemy import select

from app.core.model_artifact import is_tabular_artifact
from app.models.database import AsyncSession, Dataset, TrainingTask
from app.services.object_storage import restore_dataset_file, restore_model_bundle
from app.utils.storage_paths import resolve_runtime_path

DERIVED_TARGET_COLUMNS: dict[str, set[str]] = {
    "Target": {"Failure Type"},
    "Failure Type": {"Target"},
}


def _resolve_column_name(df: pd.DataFrame, requested_name: str) -> str:
    """Resolve a user-provided column name against the dataframe case-insensitively."""
    if requested_name in df.columns:
        return requested_name

    normalized_map = {str(column).casefold(): str(column) for column in df.columns}
    resolved = normalized_map.get(str(requested_name).casefold())
    if resolved:
        return resolved

    raise ValueError(f"Target column '{requested_name}' not found. Available: {list(df.columns)}")


def load_dataframe(file_path: str | Path) -> pd.DataFrame:
    """Load a dataframe from a supported file path."""
    path = resolve_runtime_path(file_path)
    ext = path.suffix.lower()

    if ext == ".csv":
        return pd.read_csv(path)
    if ext == ".parquet":
        return pd.read_parquet(path)
    if ext == ".xlsx":
        return pd.read_excel(path)
    raise ValueError(f"Unsupported file format: {ext}")


def prepare_training_frame(
    df: pd.DataFrame,
    target_column: str,
) -> tuple[pd.DataFrame, pd.Series, dict[str, LabelEncoder], LabelEncoder | None]:
    """Mirror training-time preprocessing for reusable inference and viz flows."""
    X, y = prepare_raw_training_frame(df, target_column)

    feature_encoders: dict[str, LabelEncoder] = {}
    for column in X.columns:
        if X[column].dtype == "object" or X[column].dtype.name == "category":
            encoder = LabelEncoder()
            X[column] = encoder.fit_transform(X[column].astype(str))
            feature_encoders[str(column)] = encoder

    target_encoder: LabelEncoder | None = None
    if y.dtype == "object" or y.dtype.name == "category":
        target_encoder = LabelEncoder()
        y = pd.Series(target_encoder.fit_transform(y.astype(str)))

    X = X.fillna(X.median(numeric_only=True)).fillna(0)

    return X, y, feature_encoders, target_encoder


def prepare_raw_training_frame(
    df: pd.DataFrame,
    target_column: str,
) -> tuple[pd.DataFrame, pd.Series]:
    """Return raw model inputs without fitting encoders or fill statistics."""
    target_column = _resolve_column_name(df, target_column)
    X = df.drop(columns=[target_column]).copy()
    y = df[target_column].copy()

    leaked_columns = DERIVED_TARGET_COLUMNS.get(target_column, set()).intersection(X.columns)
    if leaked_columns:
        X = X.drop(columns=list(leaked_columns))
    return X, y


def prepare_prediction_frame(
    training_df: pd.DataFrame,
    rows: list[dict[str, Any]],
    target_column: str,
) -> pd.DataFrame:
    """Coerce raw prediction rows into the exact feature matrix used by training."""
    if not rows:
        raise ValueError("Prediction payload must include at least one row")

    canonical_target_column = _resolve_column_name(training_df, target_column)
    reference_X, _, feature_encoders, _ = prepare_training_frame(training_df, canonical_target_column)
    input_df = pd.DataFrame(rows)

    if input_df.empty:
        raise ValueError("Prediction payload must include at least one row")

    dropped_columns = {canonical_target_column, *DERIVED_TARGET_COLUMNS.get(canonical_target_column, set())}
    input_df = input_df.drop(columns=[column for column in dropped_columns if column in input_df.columns], errors="ignore")

    required_columns = list(reference_X.columns)
    missing_columns = [column for column in required_columns if column not in input_df.columns]
    if missing_columns:
        raise ValueError(f"Missing required feature columns: {', '.join(missing_columns)}")

    input_df = input_df[required_columns].copy()

    for column in required_columns:
        if column in feature_encoders:
            encoder = feature_encoders[column]
            mapping = {value: index for index, value in enumerate(encoder.classes_)}
            series = input_df[column].astype(str)
            unknown_values = sorted({value for value in series if value not in mapping})
            if unknown_values:
                raise ValueError(f"Unknown category for column '{column}': {unknown_values[0]}")
            input_df[column] = series.map(mapping)
        else:
            input_df[column] = pd.to_numeric(input_df[column], errors="coerce")

    input_df = input_df.fillna(reference_X.median(numeric_only=True)).fillna(0)

    return input_df


def _jsonable(value: Any) -> Any:
    """Convert numpy / pandas scalar values into JSON-friendly Python primitives."""
    if isinstance(value, np.generic):
        return value.item()
    return value


def predict_with_model(
    model,
    training_df: pd.DataFrame,
    rows: list[dict[str, Any]],
    target_column: str,
    *,
    include_probabilities: bool = True,
) -> dict[str, Any]:
    """Predict through a self-contained artifact or the legacy preparation path."""
    if not rows:
        raise ValueError("Prediction payload must include at least one row")

    if is_tabular_artifact(model):
        prediction_input = pd.DataFrame(rows)
        raw_predictions = model.predict(prediction_input)
        predictions = [_jsonable(value) for value in raw_predictions.tolist()]
        artifact_labels = model.class_labels
        if not artifact_labels:
            _, target_values = prepare_raw_training_frame(training_df, target_column)
            artifact_labels = sorted(target_values.dropna().unique().tolist())
        class_labels = [str(value) for value in artifact_labels]
        probability_input = prediction_input
    else:
        prediction_input = prepare_prediction_frame(
            training_df,
            rows,
            target_column,
        )
        _, y_reference, _, target_encoder = prepare_training_frame(
            training_df,
            target_column,
        )
        raw_predictions = model.predict(prediction_input.values)
        if target_encoder is not None:
            predictions = target_encoder.inverse_transform(
                np.asarray(raw_predictions, dtype=int)
            ).tolist()
            class_labels = [str(value) for value in target_encoder.classes_.tolist()]
        else:
            predictions = [_jsonable(value) for value in raw_predictions.tolist()]
            class_labels = [
                str(value) for value in sorted(pd.Series(y_reference).unique().tolist())
            ]
        probability_input = prediction_input.values

    probabilities = None
    if include_probabilities:
        try:
            probabilities = np.asarray(
                model.predict_proba(probability_input)
            ).tolist()
        except (AttributeError, ValueError, TypeError):
            probabilities = None

    return {
        "predictions": predictions,
        "class_labels": class_labels,
        "probabilities": probabilities,
    }


async def predict_rows(
    task_id: str,
    rows: list[dict[str, Any]],
    db: AsyncSession,
    include_probabilities: bool = True,
    owner_username: str | None = None,
) -> dict[str, Any]:
    """Run inference against a saved model using inline JSON rows."""
    task_stmt = select(TrainingTask).where(TrainingTask.id == task_id)
    if owner_username:
        task_stmt = task_stmt.where(TrainingTask.owner_username == owner_username)
    task_result = await db.execute(task_stmt)
    task = task_result.scalar_one_or_none()
    if task is None:
        raise HTTPException(status_code=404, detail="Training task not found")
    if task.status != "SUCCESS" or not task.model_path:
        raise HTTPException(status_code=400, detail="Model is not available for prediction")

    dataset_result = await db.execute(select(Dataset).where(Dataset.id == task.dataset_id))
    dataset = dataset_result.scalar_one_or_none()
    if dataset is None:
        raise HTTPException(status_code=404, detail="Dataset not found")

    model_path = restore_model_bundle(task.model_path)
    if model_path is None:
        raise HTTPException(
            status_code=404,
            detail="模型文件不存在，且对象存储中无可恢复副本",
        )

    dataset_path = restore_dataset_file(dataset.id, dataset.file_path)
    if dataset_path is None:
        raise HTTPException(
            status_code=404,
            detail="数据集文件不存在，且对象存储中无可恢复副本",
        )
    training_df = load_dataframe(dataset_path)
    model = joblib.load(model_path)
    try:
        prediction = predict_with_model(
            model,
            training_df,
            rows,
            task.target_column,
            include_probabilities=include_probabilities,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    response: dict[str, Any] = {
        "task_id": task.id,
        "model_type": task.model_type,
        "target_column": task.target_column,
        "rows": len(rows),
        "predictions": prediction["predictions"],
        "class_labels": prediction["class_labels"],
    }

    if prediction["probabilities"] is not None:
        response["probabilities"] = [
            {
                str(
                    prediction["class_labels"][index]
                    if index < len(prediction["class_labels"])
                    else index
                ): round(float(probability), 6)
                for index, probability in enumerate(probabilities)
            }
            for probabilities in prediction["probabilities"]
        ]

    return response

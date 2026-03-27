"""Pydantic v2 schemas for the ML training platform API."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


# ===================================================================
# Dataset schemas
# ===================================================================

class DatasetResponse(BaseModel):
    """Serialised representation of a stored dataset."""

    model_config = ConfigDict(from_attributes=True)

    id: str
    name: str
    file_size: int
    row_count: int | None = None
    column_count: int | None = None
    columns_info: dict[str, Any] | None = None
    created_at: datetime


class DatasetPreview(BaseModel):
    """Dataset metadata together with a sample of rows and statistics."""

    model_config = ConfigDict(from_attributes=True)

    id: str
    name: str
    file_size: int
    row_count: int | None = None
    column_count: int | None = None
    columns_info: dict[str, Any] | None = None
    created_at: datetime
    rows: list[dict[str, Any]] = Field(
        default_factory=list,
        description="First 100 rows of the dataset",
    )
    statistics: dict[str, Any] = Field(
        default_factory=dict,
        description="Descriptive statistics per column",
    )


class DatasetListResponse(BaseModel):
    """Paginated list of datasets."""

    items: list[DatasetResponse]
    total: int
    page: int
    page_size: int


# ===================================================================
# Training schemas
# ===================================================================

class CrossValidationConfig(BaseModel):
    """Cross-validation settings attached to a training request."""

    enabled: bool = True
    folds: int = Field(default=5, ge=2, le=50)


class TrainingRequest(BaseModel):
    """Payload submitted by the client to start a training run."""

    dataset_id: str
    target_column: str
    model_type: str
    hyperparameters: dict[str, Any] = Field(default_factory=dict)
    test_size: float = Field(default=0.2, gt=0.0, lt=1.0)
    eval_metrics: list[str] = Field(default_factory=lambda: ["accuracy"])
    cross_validation: CrossValidationConfig | None = None


class TrainingTaskResponse(BaseModel):
    """Serialised representation of a training task."""

    model_config = ConfigDict(from_attributes=True)

    id: str
    dataset_id: str
    model_type: str
    hyperparameters: dict[str, Any] | None = None
    target_column: str
    status: str
    progress: float
    result_metrics: dict[str, Any] | None = None
    error_message: str | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None
    created_at: datetime


class TrainingListResponse(BaseModel):
    """Paginated list of training tasks."""

    items: list[TrainingTaskResponse]
    total: int
    page: int
    page_size: int


# ===================================================================
# Log / metrics schemas
# ===================================================================

class LogEntry(BaseModel):
    """Single log line emitted during training."""

    model_config = ConfigDict(from_attributes=True)

    level: str
    message: str
    extra: dict[str, Any] | None = None
    created_at: datetime


class LogResponse(BaseModel):
    """Paginated training log for a specific task."""

    task_id: str
    entries: list[LogEntry]
    total: int
    page: int
    page_size: int


class MetricsResponse(BaseModel):
    """Per-step / per-fold metric values for a training task."""

    task_id: str
    model_type: str
    steps: list[dict[str, Any]] = Field(
        default_factory=list,
        description=(
            "Ordered list of dicts, each containing a step or fold number "
            "together with metric values recorded at that point."
        ),
    )

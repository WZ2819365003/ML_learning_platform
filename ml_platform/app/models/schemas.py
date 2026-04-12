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
    class_weight: str | None = Field(
        default=None,
        description="'balanced' or None. Applied only to models with class_weight_support.",
    )


class TrainingTaskResponse(BaseModel):
    """Serialised representation of a training task."""

    model_config = ConfigDict(from_attributes=True)

    id: str
    name: str | None = None
    dataset_id: str
    model_type: str
    hyperparameters: dict[str, Any] | None = None
    target_column: str
    status: str
    progress: float
    result_metrics: dict[str, Any] | None = None
    error_message: str | None = None
    notes: str | None = None
    tags: list[str] | None = None
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


# ===================================================================
# Prediction schemas
# ===================================================================

# ===================================================================
# Model Registry schemas
# ===================================================================

class ParamSpecSchema(BaseModel):
    name: str
    display_name: str
    type: str
    default: Any = None
    min: float | None = None
    max: float | None = None
    step: float | None = None
    options: list[str] | None = None
    required: bool = False
    advanced: bool = False
    description: str = ""


class ModelMetadata(BaseModel):
    id: str
    display_name: str
    category: str
    task_types: list[str]
    description: str
    class_weight_support: bool
    params: list[ParamSpecSchema]


class CategoryMetadata(BaseModel):
    id: str
    display_name: str
    icon: str


class ModelsListResponse(BaseModel):
    categories: list[CategoryMetadata]
    models: list[ModelMetadata]
    common_params: list[ParamSpecSchema]
    classification_metrics: list[dict[str, str]]
    regression_metrics: list[dict[str, str]]


# ===================================================================
# Prediction schemas
# ===================================================================

class PredictionRequest(BaseModel):
    """Inline JSON rows submitted for saved-model inference."""

    rows: list[dict[str, Any]] = Field(default_factory=list)
    include_probabilities: bool = True


class PredictionResponse(BaseModel):
    """Prediction output returned by a saved model endpoint."""

    task_id: str
    model_type: str
    target_column: str
    rows: int
    predictions: list[Any] = Field(default_factory=list)
    class_labels: list[str] = Field(default_factory=list)
    probabilities: list[dict[str, float]] | None = None


# ===================================================================
# Deployment schemas
# ===================================================================

class DeployRequest(BaseModel):
    """Request body for deploying a trained model."""

    name: str = Field(..., min_length=1, max_length=255)
    description: str | None = None
    max_batch_size: int = Field(default=100, ge=1, le=10000)


class DeploymentResponse(BaseModel):
    """Serialised representation of a model deployment."""

    model_config = ConfigDict(from_attributes=True)

    deployment_id: str
    task_id: str
    name: str
    status: str
    request_count: int
    created_at: datetime
    endpoints: dict[str, str] = Field(default_factory=dict)


class DeploymentListResponse(BaseModel):
    """Paginated list of deployments."""

    deployments: list[DeploymentResponse]
    total: int
    page: int
    page_size: int


class MetricsSummaryResponse(BaseModel):
    primary_metric_name: str | None = None
    primary_metric_value: float | None = None
    secondary_metric_name: str | None = None
    secondary_metric_value: float | None = None


class ModelAssetResponse(BaseModel):
    asset_id: str
    runtime_type: str
    task_id: str
    name: str | None = None
    dataset_id: str
    dataset_name: str | None = None
    model_type: str
    task_type: str
    status: str
    created_at: datetime
    finished_at: datetime | None = None
    notes: str | None = None
    tags: list[str] | None = None
    model_path: str | None = None
    deployable: bool
    predictable: bool
    deployment_count: int = 0
    metrics_summary: MetricsSummaryResponse = Field(default_factory=MetricsSummaryResponse)


class ModelAssetListResponse(BaseModel):
    items: list[ModelAssetResponse]
    total: int
    page: int
    page_size: int


class UnifiedDeploymentResponse(BaseModel):
    deployment_id: str
    runtime_type: str
    source_task_id: str
    source_asset_id: str
    source_name: str | None = None
    model_type: str
    task_type: str
    name: str
    description: str | None = None
    status: str
    request_count: int
    created_at: datetime
    predict_url: str
    result_url: str | None = None
    supports_result_polling: bool


class UnifiedDeploymentListResponse(BaseModel):
    deployments: list[UnifiedDeploymentResponse]
    total: int
    page: int
    page_size: int


class InferenceRequest(BaseModel):
    """Request body for submitting an inference (predict) job."""

    rows: list[dict[str, Any]] = Field(default_factory=list)
    include_probabilities: bool = True


class InferenceJobResponse(BaseModel):
    """Serialised representation of an inference job result."""

    job_id: str
    deployment_id: str
    status: str
    predictions: list[Any] | None = None
    probabilities: list[Any] | None = None
    input_rows: int
    error_message: str | None = None
    created_at: datetime
    completed_at: datetime | None = None


# ===================================================================
# Deep-Learning schemas
# ===================================================================

class DLTrainingRequest(BaseModel):
    """Request body for starting a DL training task."""

    dataset_id:    str
    target_column: str
    model_type:    str = Field(..., description="mlp_dl / lstm / cnn1d / transformer")
    task_type:     str = Field(default="auto", description="auto / classification / regression")
    arch_config:   dict[str, Any] = Field(default_factory=dict)
    opt_config:    dict[str, Any] = Field(default_factory=dict)
    train_config:  dict[str, Any] = Field(default_factory=dict)


class TaskRenameRequest(BaseModel):
    """Request body for renaming a task."""
    name: str = Field(..., min_length=1, max_length=100)


class ModelMetaUpdateRequest(BaseModel):
    """Request body for updating model notes/tags."""
    notes: str | None = None
    tags:  list[str] | None = None


class DLTaskResponse(BaseModel):
    """Serialised DLTrainingTask."""

    model_config = ConfigDict(from_attributes=True)

    id:            str
    name:          str | None = None
    dataset_id:    str
    target_column: str
    model_type:    str
    task_type:     str
    status:        str
    progress:      float
    current_epoch: int
    total_epochs:  int
    result_metrics: dict[str, Any] | None = None
    model_path:    str | None = None
    error_message: str | None = None
    notes:         str | None = None
    tags:          list[str] | None = None
    created_at:    datetime
    started_at:    datetime | None = None
    finished_at:   datetime | None = None


class DLTaskListResponse(BaseModel):
    """Paginated list of DL training tasks."""

    items:     list[DLTaskResponse]
    total:     int
    page:      int
    page_size: int


class DLEpochResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    task_id: str
    epoch: int
    total_epochs: int
    train_loss: float | None = None
    val_loss: float | None = None
    val_acc: float | None = None
    val_f1_macro: float | None = None
    val_rmse: float | None = None
    val_mae: float | None = None
    val_r2: float | None = None
    lr: float | None = None
    created_at: datetime


class DLEpochListResponse(BaseModel):
    items: list[DLEpochResponse]
    total: int
    page: int
    page_size: int


class DLModelsResponse(BaseModel):
    """Full DL model registry response from GET /api/dl/models."""

    categories:      list[dict[str, Any]]
    models:          list[dict[str, Any]]
    optimizer_params: list[dict[str, Any]]
    train_params:    list[dict[str, Any]]


# ===================================================================
# DL Deployment schemas
# ===================================================================

class DLDeployRequest(BaseModel):
    """Request body for deploying a DL trained model."""
    name:        str = Field(..., min_length=1, max_length=255)
    description: str | None = None


class DLDeploymentResponse(BaseModel):
    """Serialised DLModelDeployment."""

    model_config = ConfigDict(from_attributes=True)

    id:            str
    dl_task_id:    str
    name:          str
    description:   str | None = None
    status:        str
    request_count: int
    created_at:    datetime


class DLDeploymentListResponse(BaseModel):
    deployments: list[DLDeploymentResponse]
    total:       int


class DLPredictionRequest(BaseModel):
    """Rows submitted for DL model inference."""
    rows: list[dict[str, Any]] = Field(default_factory=list)


class DLPredictionResponse(BaseModel):
    deployment_id: str
    task_type:     str
    rows:          int
    predictions:   list[Any]
    probabilities: list[dict[str, float]] | None = None


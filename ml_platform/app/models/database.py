"""SQLAlchemy async models for the ML training platform."""

from __future__ import annotations

import os
import uuid
from datetime import datetime, timezone
from typing import AsyncGenerator

from sqlalchemy import (
    BigInteger,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
)
from sqlalchemy import JSON
from sqlalchemy import create_engine
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship, sessionmaker


def _uuid() -> str:
    return str(uuid.uuid4())


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


DATABASE_URL: str = os.getenv(
    "DATABASE_URL",
    "mysql+aiomysql://root:123456@localhost:3307/ml_platform",
)

async_engine = create_async_engine(
    DATABASE_URL,
    echo=False,
    future=True,
)

async_session_factory = async_sessionmaker(
    bind=async_engine,
    class_=AsyncSession,
    expire_on_commit=False,
)

# Sync engine — only for code paths that run inside a ThreadPoolExecutor
# (e.g. classical-ML training in `_run_training_sync`) where awaiting the
# async engine would require bouncing back to the event loop per row. The
# URL swaps the aiomysql driver for pymysql.
SYNC_DATABASE_URL: str = DATABASE_URL.replace(
    "mysql+aiomysql://", "mysql+pymysql://"
).replace("postgresql+asyncpg://", "postgresql+psycopg2://")

sync_engine = create_engine(
    SYNC_DATABASE_URL,
    echo=False,
    future=True,
    pool_pre_ping=True,
    pool_recycle=3600,
)

sync_session_factory = sessionmaker(
    bind=sync_engine,
    expire_on_commit=False,
)


# ---------------------------------------------------------------------------
# Declarative base
# ---------------------------------------------------------------------------

class Base(DeclarativeBase):
    pass


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class Dataset(Base):
    __tablename__ = "datasets"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=_uuid
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    file_path: Mapped[str] = mapped_column(String(1024), nullable=False)
    file_size: Mapped[int] = mapped_column(BigInteger, nullable=False, default=0)
    row_count: Mapped[int | None] = mapped_column(default=None)
    column_count: Mapped[int | None] = mapped_column(default=None)
    columns_info: Mapped[dict | None] = mapped_column(JSON, default=None)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, onupdate=_utcnow
    )

    # relationships
    training_tasks: Mapped[list[TrainingTask]] = relationship(
        back_populates="dataset", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<Dataset id={self.id!r} name={self.name!r}>"


# ---------------------------------------------------------------------------
# TrainingTask
# ---------------------------------------------------------------------------

class TrainingTask(Base):
    __tablename__ = "training_tasks"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=_uuid
    )
    name: Mapped[str | None] = mapped_column(String(200), nullable=True, default=None)
    dataset_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("datasets.id", ondelete="CASCADE"), nullable=False
    )
    model_type: Mapped[str] = mapped_column(String(128), nullable=False)
    hyperparameters: Mapped[dict | None] = mapped_column(JSON, default=dict)
    target_column: Mapped[str] = mapped_column(String(255), nullable=False)
    test_size: Mapped[float] = mapped_column(Float, default=0.2)
    cv_folds: Mapped[int] = mapped_column(Integer, default=5)
    eval_metrics: Mapped[list | None] = mapped_column(JSON, default=list)
    status: Mapped[str] = mapped_column(String(32), default="PENDING", nullable=False)
    celery_task_id: Mapped[str | None] = mapped_column(String(255), default=None)
    progress: Mapped[float] = mapped_column(Float, default=0.0)
    result_metrics: Mapped[dict | None] = mapped_column(JSON, default=None)
    model_path: Mapped[str | None] = mapped_column(String(1024), default=None)
    error_message: Mapped[str | None] = mapped_column(Text, default=None)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True, default=None)
    tags: Mapped[list | None] = mapped_column(JSON, default=None)
    started_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), default=None
    )
    finished_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), default=None
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow
    )

    # relationships
    dataset: Mapped[Dataset] = relationship(back_populates="training_tasks")
    logs: Mapped[list[TrainingLog]] = relationship(
        back_populates="task", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return (
            f"<TrainingTask id={self.id!r} model_type={self.model_type!r} "
            f"status={self.status!r}>"
        )


# ---------------------------------------------------------------------------
# TrainingLog
# ---------------------------------------------------------------------------

class TrainingLog(Base):
    __tablename__ = "training_logs"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=_uuid
    )
    task_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("training_tasks.id", ondelete="CASCADE"), nullable=False
    )
    level: Mapped[str] = mapped_column(String(16), nullable=False, default="INFO")
    message: Mapped[str] = mapped_column(Text, nullable=False)
    extra: Mapped[dict | None] = mapped_column(JSON, default=None)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow
    )

    # relationships
    task: Mapped[TrainingTask] = relationship(back_populates="logs")

    def __repr__(self) -> str:
        return f"<TrainingLog id={self.id!r} level={self.level!r}>"


# ---------------------------------------------------------------------------
# ModelDeployment
# ---------------------------------------------------------------------------

class ModelDeployment(Base):
    __tablename__ = "model_deployments"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    task_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("training_tasks.id", ondelete="CASCADE"), nullable=False
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[str | None] = mapped_column(Text, default=None)
    status: Mapped[str] = mapped_column(String(32), default="active", nullable=False)
    max_batch_size: Mapped[int] = mapped_column(Integer, default=100)
    request_count: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, onupdate=_utcnow
    )

    inference_jobs: Mapped[list[InferenceJob]] = relationship(
        back_populates="deployment", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<ModelDeployment id={self.id!r} name={self.name!r} status={self.status!r}>"


# ---------------------------------------------------------------------------
# InferenceJob
# ---------------------------------------------------------------------------

class InferenceJob(Base):
    __tablename__ = "inference_jobs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    deployment_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("model_deployments.id", ondelete="CASCADE"), nullable=False
    )
    status: Mapped[str] = mapped_column(String(32), default="pending", nullable=False)
    input_rows: Mapped[int] = mapped_column(Integer, default=0)
    predictions: Mapped[list | None] = mapped_column(JSON, default=None)
    probabilities: Mapped[list | None] = mapped_column(JSON, default=None)
    error_message: Mapped[str | None] = mapped_column(Text, default=None)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), default=None)

    deployment: Mapped[ModelDeployment] = relationship(back_populates="inference_jobs")

    def __repr__(self) -> str:
        return f"<InferenceJob id={self.id!r} status={self.status!r}>"


# ---------------------------------------------------------------------------
# DLTrainingTask
# ---------------------------------------------------------------------------

class DLTrainingTask(Base):
    __tablename__ = "dl_training_tasks"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    name: Mapped[str | None] = mapped_column(String(200), nullable=True, default=None)
    dataset_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("datasets.id", ondelete="CASCADE"), nullable=False
    )
    target_column: Mapped[str] = mapped_column(String(255), nullable=False)
    model_type: Mapped[str] = mapped_column(String(64), nullable=False)
    task_type: Mapped[str] = mapped_column(String(32), default="classification")

    arch_config:  Mapped[dict | None] = mapped_column(JSON, default=dict)
    opt_config:   Mapped[dict | None] = mapped_column(JSON, default=dict)
    train_config: Mapped[dict | None] = mapped_column(JSON, default=dict)

    status:        Mapped[str]   = mapped_column(String(32), default="PENDING", nullable=False)
    progress:      Mapped[float] = mapped_column(Float, default=0.0)
    current_epoch: Mapped[int]   = mapped_column(Integer, default=0)
    total_epochs:  Mapped[int]   = mapped_column(Integer, default=0)

    result_metrics: Mapped[dict | None] = mapped_column(JSON, default=None)
    model_path:     Mapped[str | None]  = mapped_column(String(1024), default=None)
    error_message:  Mapped[str | None]  = mapped_column(Text, default=None)
    notes:          Mapped[str | None]  = mapped_column(Text, nullable=True, default=None)
    tags:           Mapped[list | None] = mapped_column(JSON, default=None)

    created_at:  Mapped[datetime]      = mapped_column(DateTime(timezone=True), default=_utcnow)
    started_at:  Mapped[datetime | None] = mapped_column(DateTime(timezone=True), default=None)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), default=None)

    dataset: Mapped[Dataset] = relationship("Dataset")
    dl_epochs: Mapped[list["DLTrainingEpoch"]] = relationship(
        back_populates="task", cascade="all, delete-orphan"
    )
    dl_logs: Mapped[list["DLTrainingLog"]] = relationship(
        back_populates="task", cascade="all, delete-orphan"
    )
    dl_deployments: Mapped[list["DLModelDeployment"]] = relationship(
        back_populates="dl_task", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<DLTrainingTask id={self.id!r} model={self.model_type!r} status={self.status!r}>"


# ---------------------------------------------------------------------------
# DLTrainingEpoch
# ---------------------------------------------------------------------------

class DLTrainingEpoch(Base):
    __tablename__ = "dl_training_epochs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    task_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("dl_training_tasks.id", ondelete="CASCADE"), nullable=False
    )
    epoch: Mapped[int] = mapped_column(Integer, nullable=False)
    total_epochs: Mapped[int] = mapped_column(Integer, default=0)
    train_loss: Mapped[float | None] = mapped_column(Float, default=None)
    val_loss: Mapped[float | None] = mapped_column(Float, default=None)
    val_acc: Mapped[float | None] = mapped_column(Float, default=None)
    val_f1_macro: Mapped[float | None] = mapped_column(Float, default=None)
    val_rmse: Mapped[float | None] = mapped_column(Float, default=None)
    val_mae: Mapped[float | None] = mapped_column(Float, default=None)
    val_r2: Mapped[float | None] = mapped_column(Float, default=None)
    lr: Mapped[float | None] = mapped_column(Float, default=None)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    task: Mapped[DLTrainingTask] = relationship(back_populates="dl_epochs")

    def __repr__(self) -> str:
        return f"<DLTrainingEpoch task_id={self.task_id!r} epoch={self.epoch!r}>"


# ---------------------------------------------------------------------------
# DLTrainingLog
# ---------------------------------------------------------------------------

class DLTrainingLog(Base):
    __tablename__ = "dl_training_logs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    task_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("dl_training_tasks.id", ondelete="CASCADE"), nullable=False
    )
    level: Mapped[str] = mapped_column(String(16), nullable=False, default="INFO")
    message: Mapped[str] = mapped_column(Text, nullable=False)
    extra: Mapped[dict | None] = mapped_column(JSON, default=None)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    task: Mapped[DLTrainingTask] = relationship(back_populates="dl_logs")

    def __repr__(self) -> str:
        return f"<DLTrainingLog task_id={self.task_id!r} level={self.level!r}>"


# ---------------------------------------------------------------------------
# DLModelDeployment
# ---------------------------------------------------------------------------

class DLModelDeployment(Base):
    __tablename__ = "dl_model_deployments"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    dl_task_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("dl_training_tasks.id", ondelete="CASCADE"), nullable=False
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[str | None] = mapped_column(Text, default=None)
    status: Mapped[str] = mapped_column(String(32), default="active", nullable=False)
    request_count: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, onupdate=_utcnow
    )

    dl_task: Mapped[DLTrainingTask] = relationship(back_populates="dl_deployments")

    def __repr__(self) -> str:
        return f"<DLModelDeployment id={self.id!r} name={self.name!r} status={self.status!r}>"


# ---------------------------------------------------------------------------
# ModelTagLibrary  – shared tag vocabulary across ML and DL models
# ---------------------------------------------------------------------------

class ModelTagLibrary(Base):
    __tablename__ = "model_tag_library"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    name: Mapped[str] = mapped_column(String(100), nullable=False, unique=True)
    dimension: Mapped[str | None] = mapped_column(String(50), default=None)   # 类别/规模/目的/领域/数据类型
    color: Mapped[str | None] = mapped_column(String(30), default=None)        # Ant Design Tag color token
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    def __repr__(self) -> str:
        return f"<ModelTagLibrary name={self.name!r} dimension={self.dimension!r}>"


# ---------------------------------------------------------------------------
# TimeSeriesForecastTask  – tracks async Chronos forecast jobs
# ---------------------------------------------------------------------------

class TimeSeriesDeployment(Base):
    __tablename__ = "ts_deployments"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[str | None] = mapped_column(Text, default=None)
    backend_label: Mapped[str] = mapped_column(
        String(128),
        nullable=False,
        default="amazon/chronos-t5-small",
    )
    status: Mapped[str] = mapped_column(String(32), default="active", nullable=False)
    request_count: Mapped[int] = mapped_column(Integer, default=0)
    config: Mapped[dict | None] = mapped_column(JSON, default=None)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, onupdate=_utcnow
    )

    def __repr__(self) -> str:
        return f"<TimeSeriesDeployment id={self.id!r} name={self.name!r} status={self.status!r}>"


class TimeSeriesForecastTask(Base):
    __tablename__ = "ts_forecast_tasks"
    __table_args__ = (
        Index("ix_ts_forecast_tasks_created_at", "created_at"),
        Index("ix_ts_forecast_tasks_status", "status"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)

    # Input configuration
    dataset_id: Mapped[str] = mapped_column(String(36), nullable=False)
    dataset_name: Mapped[str | None] = mapped_column(String(255), default=None)
    value_column: Mapped[str] = mapped_column(String(255), nullable=False)
    time_column: Mapped[str | None] = mapped_column(String(255), default=None)
    horizon: Mapped[int] = mapped_column(Integer, default=24)
    frequency: Mapped[str] = mapped_column(String(32), default="high")
    model_name: Mapped[str] = mapped_column(String(128), default="amazon/chronos-t5-small")

    # Execution state
    status: Mapped[str] = mapped_column(String(32), default="PENDING", nullable=False)
    error_message: Mapped[str | None] = mapped_column(Text, default=None)

    # Result payload (stored as JSON)
    result: Mapped[dict | None] = mapped_column(JSON, default=None)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True, default=None)
    tags: Mapped[list | None] = mapped_column(JSON, default=None)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), default=None)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), default=None)

    def __repr__(self) -> str:
        return f"<TimeSeriesForecastTask id={self.id!r} status={self.status!r}>"


# ---------------------------------------------------------------------------
# V3 Platform: DatasetVersion
# ---------------------------------------------------------------------------

class DatasetVersion(Base):
    """Tracks versioned snapshots of a dataset (for incremental training)."""
    __tablename__ = "dataset_versions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    dataset_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("datasets.id", ondelete="CASCADE"), nullable=False
    )
    version: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    file_uri: Mapped[str] = mapped_column(String(1024), nullable=False)
    row_count: Mapped[int | None] = mapped_column(Integer, default=None)
    schema_hash: Mapped[str | None] = mapped_column(String(64), default=None)
    description: Mapped[str | None] = mapped_column(Text, default=None)
    parent_version_id: Mapped[str | None] = mapped_column(
        String(36), ForeignKey("dataset_versions.id", ondelete="SET NULL"), default=None
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    dataset: Mapped["Dataset"] = relationship("Dataset")

    def __repr__(self) -> str:
        return f"<DatasetVersion dataset_id={self.dataset_id!r} version={self.version!r}>"


# ---------------------------------------------------------------------------
# V3 Platform: ModelingTask (top-level modeling workbench entity)
# ---------------------------------------------------------------------------

class ModelingTask(Base):
    """
    A modeling task is the top-level narrative unit of the V3 workbench.

    One ModelingTask = one "modeling goal" (e.g. 'predict churn on customer_v3')
    and owns one or more PlatformExperiments — each being a distinct *strategy*
    (baseline, grid search, Bayesian search).  Experiments in turn own runs.

    Hierarchy:  ModelingTask → PlatformExperiment → ExperimentRun → PlatformTask
    """
    __tablename__ = "modeling_tasks"
    __table_args__ = (
        Index("ix_modeling_tasks_status", "status"),
        Index("ix_modeling_tasks_created_at", "created_at"),
        Index("ix_modeling_tasks_dataset_id", "dataset_id"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[str | None] = mapped_column(Text, default=None)

    # Dataset binding (modeling task is dataset-scoped)
    dataset_id: Mapped[str | None] = mapped_column(String(36), default=None)
    dataset_name: Mapped[str | None] = mapped_column(String(255), default=None)
    dataset_version_id: Mapped[str | None] = mapped_column(
        String(36), ForeignKey("dataset_versions.id", ondelete="SET NULL"), default=None
    )

    # Modeling contract
    target_column: Mapped[str | None] = mapped_column(String(255), default=None)
    task_type: Mapped[str] = mapped_column(String(32), default="classification")
    # classification | regression
    objective_metric: Mapped[str] = mapped_column(String(64), default="accuracy")
    objective_direction: Mapped[str] = mapped_column(String(8), default="max")

    # Lifecycle
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="CREATED")
    # CREATED | RUNNING | COMPLETED | FAILED | ARCHIVED
    best_experiment_id: Mapped[str | None] = mapped_column(String(36), default=None)
    best_run_id: Mapped[str | None] = mapped_column(String(36), default=None)

    # Freeform config + precomputed summary (leaderboard snapshot, counts, etc.)
    config: Mapped[dict | None] = mapped_column(JSON, default=None)
    summary_snapshot: Mapped[dict | None] = mapped_column(JSON, default=None)

    # V3 Phase 2 — TrainingPlan binding.
    # `training_plan_id` is a live FK (ON DELETE SET NULL) for "jump to current
    # plan" UX; `training_plan_snapshot` is the immutable recipe captured at
    # bind-time so historical runs remain reproducible even after the plan is
    # edited or deleted.  Snapshot shape:
    #   { "plan_id", "plan_version", "captured_at",
    #     "payload": {name, task_type, strategy_type, model_family,
    #                 selected_models, search_space, dl_config,
    #                 budget_config, eval_metrics,
    #                 default_objective_metric, default_objective_direction},
    #     "dag_shape": { "nodes": [...] } }
    training_plan_id: Mapped[str | None] = mapped_column(
        String(36),
        ForeignKey("training_plans.id", ondelete="SET NULL"),
        default=None,
        index=True,
    )
    training_plan_snapshot: Mapped[dict | None] = mapped_column(JSON, default=None)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, onupdate=_utcnow
    )
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), default=None)

    experiments: Mapped[list["PlatformExperiment"]] = relationship(
        back_populates="modeling_task",
        cascade="all, delete-orphan",
        foreign_keys="PlatformExperiment.modeling_task_id",
    )

    def __repr__(self) -> str:
        return (
            f"<ModelingTask id={self.id!r} name={self.name!r} status={self.status!r}>"
        )


# ---------------------------------------------------------------------------
# V3 Platform: PlatformTask (unified async work unit)
# ---------------------------------------------------------------------------

class PlatformTask(Base):
    """Unified task record — all async work goes through here."""
    __tablename__ = "platform_tasks"
    __table_args__ = (
        Index("ix_platform_tasks_status", "status"),
        Index("ix_platform_tasks_created_at", "created_at"),
        Index("ix_platform_tasks_kind", "kind"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    kind: Mapped[str] = mapped_column(String(32), nullable=False)
    # train | eval | predict | explain | preprocess | automl
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="PENDING")
    # PENDING | QUEUED | RUNNING | SUCCESS | FAILED | RETRY | CANCELLED
    priority: Mapped[int] = mapped_column(Integer, default=5)

    celery_task_id: Mapped[str | None] = mapped_column(String(255), default=None)
    worker_id: Mapped[str | None] = mapped_column(String(255), default=None)

    retry_count: Mapped[int] = mapped_column(Integer, default=0)
    max_retries: Mapped[int] = mapped_column(Integer, default=3)

    # "kind:id" — e.g. "train:abc123" links to training_tasks.id
    payload_ref: Mapped[str | None] = mapped_column(String(512), default=None)

    # V3 Phase 2 — DAG edge list.  Stores sibling PlatformTask.ids this task
    # waits for; empty / null means "no dependencies, schedule immediately".
    # Phase 2 populates this column but the InProcessScheduler treats all tasks
    # as independent (backwards-compatible with today's behaviour).  Phase 5
    # CeleryScheduler turns on the gate so downstream tasks only submit once
    # all upstream tasks reach SUCCESS.
    depends_on: Mapped[list | None] = mapped_column(JSON, default=None)

    progress: Mapped[float] = mapped_column(Float, default=0.0)

    logs_uri: Mapped[str | None] = mapped_column(String(1024), default=None)
    metrics_uri: Mapped[str | None] = mapped_column(String(1024), default=None)
    artifacts_uri: Mapped[str | None] = mapped_column(String(1024), default=None)
    metrics_snapshot: Mapped[dict | None] = mapped_column(JSON, default=None)

    error_message: Mapped[str | None] = mapped_column(Text, default=None)

    queued_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), default=None)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), default=None)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), default=None)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, onupdate=_utcnow
    )

    runs: Mapped[list["ExperimentRun"]] = relationship(
        back_populates="task", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<PlatformTask id={self.id!r} kind={self.kind!r} status={self.status!r}>"


# ---------------------------------------------------------------------------
# V3 Platform: PlatformExperiment
# ---------------------------------------------------------------------------

class PlatformExperiment(Base):
    """Groups one or more ExperimentRuns (single run or AutoML search)."""
    __tablename__ = "platform_experiments"
    __table_args__ = (
        Index("ix_platform_experiments_status", "status"),
        Index("ix_platform_experiments_created_at", "created_at"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[str | None] = mapped_column(Text, default=None)

    # Parent modeling task (nullable for backward compat; populated by migration)
    modeling_task_id: Mapped[str | None] = mapped_column(
        String(36), ForeignKey("modeling_tasks.id", ondelete="CASCADE"), default=None
    )

    dataset_id: Mapped[str | None] = mapped_column(String(36), default=None)
    dataset_name: Mapped[str | None] = mapped_column(String(255), default=None)
    dataset_version_id: Mapped[str | None] = mapped_column(
        String(36), ForeignKey("dataset_versions.id", ondelete="SET NULL"), default=None
    )

    objective_metric: Mapped[str] = mapped_column(String(64), default="accuracy")
    objective_direction: Mapped[str] = mapped_column(String(8), default="max")

    status: Mapped[str] = mapped_column(String(32), nullable=False, default="CREATED")
    # CREATED | RUNNING | COMPLETED | FAILED
    kind: Mapped[str] = mapped_column(String(32), default="single")
    # single | automl | grid_search  (legacy; use strategy_type for V3 workflows)

    # V3 strategy metadata ──────────────────────────────────────────────────
    # strategy_type: baseline | grid_search | bayesian_search
    strategy_type: Mapped[str] = mapped_column(String(32), default="baseline", nullable=False)
    # Models the user selected for this batch: ["logreg", "random_forest", ...]
    selected_models: Mapped[list | None] = mapped_column(JSON, default=None)
    # Parameter search space (grid/bayesian only).  Shape depends on strategy.
    search_space: Mapped[dict | None] = mapped_column(JSON, default=None)
    # Budget: {max_trials, timeout_minutes, concurrency, random_state, ...}
    budget_config: Mapped[dict | None] = mapped_column(JSON, default=None)

    best_run_id: Mapped[str | None] = mapped_column(String(36), default=None)
    config: Mapped[dict | None] = mapped_column(JSON, default=None)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, onupdate=_utcnow
    )
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), default=None)

    modeling_task: Mapped["ModelingTask | None"] = relationship(
        back_populates="experiments",
        foreign_keys=[modeling_task_id],
    )
    runs: Mapped[list["ExperimentRun"]] = relationship(
        back_populates="experiment",
        cascade="all, delete-orphan",
        foreign_keys="ExperimentRun.experiment_id",
    )

    def __repr__(self) -> str:
        return f"<PlatformExperiment id={self.id!r} name={self.name!r} status={self.status!r}>"


# ---------------------------------------------------------------------------
# V3 Platform: ExperimentRun
# ---------------------------------------------------------------------------

class ExperimentRun(Base):
    """One training attempt within an experiment."""
    __tablename__ = "experiment_runs"
    __table_args__ = (
        Index("ix_experiment_runs_experiment_id", "experiment_id"),
        Index("ix_experiment_runs_status", "status"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    experiment_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("platform_experiments.id", ondelete="CASCADE"), nullable=False
    )
    task_id: Mapped[str | None] = mapped_column(
        String(36), ForeignKey("platform_tasks.id", ondelete="SET NULL"), default=None
    )
    parent_run_id: Mapped[str | None] = mapped_column(
        String(36), ForeignKey("experiment_runs.id", ondelete="SET NULL"), default=None
    )

    params: Mapped[dict | None] = mapped_column(JSON, default=None)
    metrics: Mapped[dict | None] = mapped_column(JSON, default=None)

    status: Mapped[str] = mapped_column(String(32), nullable=False, default="PENDING")
    rank: Mapped[int | None] = mapped_column(Integer, default=None)
    artifacts_uri: Mapped[str | None] = mapped_column(String(1024), default=None)
    notes: Mapped[str | None] = mapped_column(Text, default=None)

    # V3 tuning metadata ────────────────────────────────────────────────────
    # trial_no: sequence within its experiment (1-based).  Useful for grid/optuna.
    trial_no: Mapped[int | None] = mapped_column(Integer, default=None)
    # search_meta: strategy-specific trial info
    #   grid_search:     {"grid_index": 3, "combo": {...}}
    #   bayesian_search: {"optuna_trial_id": 7, "state": "COMPLETE", "value": 0.87}
    #   baseline:        None
    search_meta: Mapped[dict | None] = mapped_column(JSON, default=None)
    # Denormalised strategy_type from parent experiment — lets us filter runs
    # in the Run Inspector without a join.
    source_experiment_type: Mapped[str | None] = mapped_column(String(32), default=None)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), default=None)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), default=None)

    experiment: Mapped["PlatformExperiment"] = relationship(
        back_populates="runs", foreign_keys=[experiment_id]
    )
    task: Mapped["PlatformTask | None"] = relationship(back_populates="runs")

    def __repr__(self) -> str:
        return f"<ExperimentRun id={self.id!r} experiment_id={self.experiment_id!r} status={self.status!r}>"


# ---------------------------------------------------------------------------
# Training Plan — reusable template of (strategy, models, space, budget)
# ---------------------------------------------------------------------------

class TrainingPlan(Base):
    """
    A named, reusable training configuration that can be applied to any
    ModelingTask.  Think of it as a "recipe":

      name          = "Baseline XGB + LogReg 快速筛选"
      task_type     = "classification"
      strategy_type = "baseline"
      selected_models = ["xgboost", "logistic_regression"]
      search_space  = { model_type: { param: override } }
      budget_config = { max_trials: 20, test_size: 0.2 }
      eval_metrics  = ["accuracy", "f1", "roc_auc"]

    When the user creates a new experiment batch on a ModelingTask they can
    pick a plan and have the batch-config form pre-filled.  The plan itself
    has no binding to a dataset — it's the strategic shape of the training,
    reusable across datasets.
    """
    __tablename__ = "training_plans"
    __table_args__ = (
        Index("ix_training_plans_task_type", "task_type"),
        Index("ix_training_plans_created_at", "created_at"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[str | None] = mapped_column(Text, default=None)

    task_type: Mapped[str] = mapped_column(String(32), nullable=False, default="classification")
    strategy_type: Mapped[str] = mapped_column(String(32), nullable=False, default="baseline")
    # 'baseline' | 'grid_search' | 'bayesian_search'

    # V3 Phase 1 — DL integration
    model_family: Mapped[str] = mapped_column(String(16), nullable=False, default="ml")
    # 'ml' | 'dl' | 'mixed' — decides which registry(ies) selected_models are drawn from

    selected_models: Mapped[list | None] = mapped_column(JSON, default=None)
    # list of model_type tokens. For mixed plans, tokens may belong to either the
    # ML registry or the DL registry; the family is resolved per-token at dispatch time.

    search_space: Mapped[dict | None] = mapped_column(JSON, default=None)
    # per-model overrides for ML models: { "xgboost": { "max_depth": {...override...} } }

    dl_config: Mapped[dict | None] = mapped_column(JSON, default=None)
    # per-model DL presets: { "mlp_dl": { "arch": {...}, "opt": {...}, "train": {...} } }
    # Used as baseline config when strategy_type == "baseline". grid/bayesian on DL
    # is downgraded to baseline in Phase 1.

    budget_config: Mapped[dict | None] = mapped_column(JSON, default=None)
    # { "max_trials": 20, "test_size": 0.2, "cv_folds": 5, "timeout_minutes": 60 }

    eval_metrics: Mapped[list | None] = mapped_column(JSON, default=None)
    default_objective_metric: Mapped[str | None] = mapped_column(String(64), default=None)
    default_objective_direction: Mapped[str | None] = mapped_column(String(8), default=None)

    # Usage bookkeeping (helpful for sorting most-used in the picker)
    use_count: Mapped[int] = mapped_column(Integer, default=0)
    last_used_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), default=None)

    # V3 Phase 2 — version bumped on every payload-touching update_plan call.
    # ModelingTask.training_plan_snapshot records the version in effect at binding
    # time, so "which version of this plan did this task run?" stays answerable
    # even after the plan has been edited.
    version: Mapped[int] = mapped_column(Integer, nullable=False, default=1)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, onupdate=_utcnow
    )

    def __repr__(self) -> str:
        return (
            f"<TrainingPlan id={self.id!r} name={self.name!r} v={self.version} "
            f"task_type={self.task_type!r} strategy_type={self.strategy_type!r}>"
        )


# ---------------------------------------------------------------------------
# Dependency injection helper
# ---------------------------------------------------------------------------

async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency that yields an async database session."""
    async with async_session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise

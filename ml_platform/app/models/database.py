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
    Integer,
    String,
    Text,
)
from sqlalchemy.dialects.sqlite import JSON
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def _uuid() -> str:
    return str(uuid.uuid4())


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


DATABASE_URL: str = os.getenv(
    "DATABASE_URL",
    "sqlite+aiosqlite:///./storage/ml_platform.db",
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
    dataset_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("datasets.id", ondelete="CASCADE"), nullable=False
    )
    model_type: Mapped[str] = mapped_column(String(128), nullable=False)
    hyperparameters: Mapped[dict | None] = mapped_column(JSON, default=dict)
    target_column: Mapped[str] = mapped_column(String(255), nullable=False)
    test_size: Mapped[float] = mapped_column(Float, default=0.2)
    eval_metrics: Mapped[list | None] = mapped_column(JSON, default=list)
    status: Mapped[str] = mapped_column(String(32), default="PENDING", nullable=False)
    celery_task_id: Mapped[str | None] = mapped_column(String(255), default=None)
    progress: Mapped[float] = mapped_column(Float, default=0.0)
    result_metrics: Mapped[dict | None] = mapped_column(JSON, default=None)
    model_path: Mapped[str | None] = mapped_column(String(1024), default=None)
    error_message: Mapped[str | None] = mapped_column(Text, default=None)
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

    created_at:  Mapped[datetime]      = mapped_column(DateTime(timezone=True), default=_utcnow)
    started_at:  Mapped[datetime | None] = mapped_column(DateTime(timezone=True), default=None)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), default=None)

    dataset: Mapped[Dataset] = relationship("Dataset")

    def __repr__(self) -> str:
        return f"<DLTrainingTask id={self.id!r} model={self.model_type!r} status={self.status!r}>"


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

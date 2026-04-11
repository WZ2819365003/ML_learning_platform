"""Tests for DL training progress persistence helpers and log formatting."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
import pytest_asyncio
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.models.database import Base, DLTrainingTask, Dataset
from app.services import dl_service
from app.services.dl_service import (
    _build_dl_completion_log_entry,
    _build_dl_epoch_log_entry,
    _store_dl_epoch_progress,
)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


@pytest_asyncio.fixture
async def db_session(tmp_path):
    db_file = tmp_path / "dl_progress.db"
    engine = create_async_engine(f"sqlite+aiosqlite:///{db_file.as_posix()}", echo=False)
    sessionmaker = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    async with sessionmaker() as session:
        dataset = Dataset(
            name="dl-seed.csv",
            file_path=str(tmp_path / "seed.csv"),
            file_size=16,
            row_count=2,
            column_count=3,
            columns_info={"f1": {}, "f2": {}, "target": {}},
        )
        session.add(dataset)
        await session.flush()

        task = DLTrainingTask(
            dataset_id=dataset.id,
            name="dl-task",
            target_column="target",
            model_type="mlp_dl",
            task_type="classification",
            status="RUNNING",
            progress=0.0,
            current_epoch=0,
            total_epochs=20,
            created_at=_utcnow(),
        )
        session.add(task)
        await session.commit()

        yield sessionmaker, task.id

    await engine.dispose()


@pytest.mark.asyncio
async def test_store_dl_epoch_progress_updates_current_epoch_and_percent(db_session, monkeypatch):
    sessionmaker, task_id = db_session
    monkeypatch.setattr(dl_service, "async_session_factory", sessionmaker)

    await _store_dl_epoch_progress(task_id, epoch=5, progress=25.0)

    async with sessionmaker() as session:
        result = await session.execute(select(DLTrainingTask).where(DLTrainingTask.id == task_id))
        task = result.scalar_one()

    assert task.current_epoch == 5
    assert task.progress == 25.0


def test_build_dl_epoch_log_entry_for_classification_contains_validation_metrics():
    message, extra = _build_dl_epoch_log_entry(
        epoch=5,
        total_epochs=20,
        metrics={
            "train_loss": 0.314159,
            "val_loss": 0.271828,
            "val_acc": 0.9625,
            "val_f1_macro": 0.9487,
        },
    )

    assert message == "Epoch 5/20 指标更新"
    assert extra == {
        "train_loss": "0.314159",
        "val_loss": "0.271828",
        "val_acc": "0.962500",
        "val_f1_macro": "0.948700",
    }


def test_build_dl_completion_log_entry_for_regression_prefers_summary_metrics():
    message, extra = _build_dl_completion_log_entry(
        {
            "best_val_loss": 0.112233,
            "val_rmse": 1.234567,
            "val_mae": 0.987654,
            "val_r2": 0.845612,
            "history": [{"epoch": 1, "val_loss": 0.3}],
            "val_scatter": {"actual": [1.0], "predicted": [1.1]},
        }
    )

    assert message == "训练完成，最终验证指标"
    assert extra == {
        "best_val_loss": "0.112233",
        "val_rmse": "1.234567",
        "val_mae": "0.987654",
        "val_r2": "0.845612",
    }


@pytest.mark.asyncio
async def test_list_dl_epoch_history_returns_paginated_rows(db_session, monkeypatch):
    from app.models.database import DLTrainingEpoch

    sessionmaker, task_id = db_session
    monkeypatch.setattr(dl_service, "async_session_factory", sessionmaker)

    async with sessionmaker() as session:
        session.add_all([
            DLTrainingEpoch(
                task_id=task_id,
                epoch=1,
                total_epochs=20,
                train_loss=0.8,
                val_loss=0.7,
                val_acc=0.6,
                lr=0.001,
            ),
            DLTrainingEpoch(
                task_id=task_id,
                epoch=2,
                total_epochs=20,
                train_loss=0.6,
                val_loss=0.5,
                val_acc=0.72,
                lr=0.001,
            ),
        ])
        await session.commit()

    async with sessionmaker() as session:
        payload = await dl_service.list_dl_epoch_history(task_id=task_id, db=session, page=1, page_size=1)

    assert payload["total"] == 2
    assert payload["page"] == 1
    assert payload["page_size"] == 1
    assert len(payload["items"]) == 1
    assert payload["items"][0].epoch == 2

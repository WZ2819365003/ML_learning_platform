"""
Phase 3 — orphan PlatformTask detail resolver tests.

Covers the four known payload_ref kinds (train, dl_train, explain,
ts_forecast) plus three edge cases:
    * unknown PlatformTask id → 404
    * unknown payload_ref prefix → skeleton without domain
    * missing domain row → domain=None but task/skeleton still returned
    * log tail pulls the last N lines from storage/logs/{domain_id}.log
"""
from __future__ import annotations

from pathlib import Path

import pytest
from fastapi import HTTPException

from app.models.database import (
    DLTrainingTask,
    Dataset,
    ExperimentRun,
    PlatformExperiment,
    PlatformTask,
    TimeSeriesForecastTask,
    TrainingTask,
)
from app.services import platform_task_detail_service as svc


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@pytest.fixture
async def dataset(db):
    ds = Dataset(name="orphan_ds", file_path="/tmp/orphan.csv", file_size=0, row_count=10)
    db.add(ds)
    await db.flush()
    return ds


async def _mk_platform_task(db, *, kind: str, payload_ref: str | None) -> PlatformTask:
    t = PlatformTask(kind=kind, status="SUCCESS", payload_ref=payload_ref, progress=1.0)
    db.add(t)
    await db.flush()
    return t


# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------

async def test_detail_for_train_task(db, dataset):
    tt = TrainingTask(
        dataset_id=dataset.id,
        model_type="random_forest",
        target_column="y",
        status="SUCCESS",
        progress=1.0,
        result_metrics={"accuracy": 0.91},
    )
    db.add(tt)
    await db.flush()
    ptask = await _mk_platform_task(db, kind="train", payload_ref=f"train:{tt.id}")

    detail = await svc.get_platform_task_detail(db, ptask.id)

    assert detail["task"]["id"] == ptask.id
    assert detail["source_label"] == "独立训练"
    assert detail["domain_kind"] == "train"
    assert detail["domain_id"] == tt.id
    assert detail["domain"]["model_type"] == "random_forest"
    assert detail["domain"]["result_metrics"] == {"accuracy": 0.91}


# ---------------------------------------------------------------------------
# DL Train
# ---------------------------------------------------------------------------

async def test_detail_for_dl_train_task(db, dataset):
    tt = DLTrainingTask(
        dataset_id=dataset.id,
        target_column="y",
        model_type="mlp_dl",
        task_type="classification",
        status="RUNNING",
        progress=0.4,
        current_epoch=20,
        total_epochs=50,
        arch_config={"hidden": [64, 32]},
    )
    db.add(tt)
    await db.flush()
    ptask = await _mk_platform_task(db, kind="dl_train", payload_ref=f"dl_train:{tt.id}")

    detail = await svc.get_platform_task_detail(db, ptask.id)

    assert detail["source_label"] == "独立 DL 训练"
    assert detail["domain_kind"] == "dl_train"
    assert detail["domain"]["model_type"] == "mlp_dl"
    assert detail["domain"]["current_epoch"] == 20
    assert detail["domain"]["total_epochs"] == 50
    assert detail["domain"]["arch_config"] == {"hidden": [64, 32]}


# ---------------------------------------------------------------------------
# Explain
# ---------------------------------------------------------------------------

async def test_detail_for_explain_task(db, dataset):
    # explain's domain is an ExperimentRun; set up the skeleton chain:
    #   PlatformExperiment → ExperimentRun → (linked training PlatformTask)
    training_task = TrainingTask(
        dataset_id=dataset.id, model_type="xgboost", target_column="y",
        status="SUCCESS", model_path="/tmp/x.joblib",
    )
    db.add(training_task)
    await db.flush()

    training_ptask = await _mk_platform_task(
        db, kind="train", payload_ref=f"train:{training_task.id}"
    )

    exp = PlatformExperiment(name="ex1", strategy_type="baseline")
    db.add(exp)
    await db.flush()

    run = ExperimentRun(
        experiment_id=exp.id,
        task_id=training_ptask.id,
        status="SUCCESS",
        metrics={"accuracy": 0.88},
        params={"model_type": "xgboost", "dataset_id": dataset.id, "target_column": "y"},
    )
    db.add(run)
    await db.flush()

    explain_ptask = await _mk_platform_task(
        db, kind="explain", payload_ref=f"explain:{run.id}"
    )

    detail = await svc.get_platform_task_detail(db, explain_ptask.id)

    assert detail["source_label"] == "独立 SHAP 解释"
    assert detail["domain"]["run_id"] == run.id
    assert detail["domain"]["model_type"] == "xgboost"
    assert detail["domain"]["linked_training_task_id"] == training_task.id


# ---------------------------------------------------------------------------
# TS Forecast
# ---------------------------------------------------------------------------

async def test_detail_for_ts_forecast_task(db):
    tt = TimeSeriesForecastTask(
        dataset_id="ds-ts-1",
        dataset_name="sales.csv",
        value_column="sales",
        horizon=24,
        frequency="high",
        model_name="amazon/chronos-t5-small",
        status="SUCCESS",
        result={
            "summary": "ok",
            "point": list(range(30)),
            "low":   list(range(30)),
            "high":  list(range(30)),
        },
    )
    db.add(tt)
    await db.flush()

    ptask = await _mk_platform_task(
        db, kind="forecast", payload_ref=f"ts_forecast:{tt.id}"
    )

    detail = await svc.get_platform_task_detail(db, ptask.id)

    assert detail["source_label"] == "时序预测"
    assert detail["domain_kind"] == "ts_forecast"
    assert detail["domain"]["dataset_name"] == "sales.csv"
    # Result is trimmed — we should see head arrays + counts, not the full payload
    preview = detail["domain"]["result_preview"]
    assert preview["point_count"] == 30
    assert preview["point_head"] == list(range(10))
    assert "low_head" in preview and "high_head" in preview


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

async def test_detail_404_for_unknown_task(db):
    with pytest.raises(HTTPException) as exc:
        await svc.get_platform_task_detail(db, "nonexistent")
    assert exc.value.status_code == 404


async def test_detail_unknown_payload_prefix_still_returns_skeleton(db):
    ptask = await _mk_platform_task(
        db, kind="mystery", payload_ref="mystery:abc123"
    )
    detail = await svc.get_platform_task_detail(db, ptask.id)
    assert detail["task"]["id"] == ptask.id
    assert detail["domain_kind"] == "mystery"
    assert detail["domain"] is None
    assert detail["source_label"] == "mystery"  # falls back to raw kind


async def test_detail_missing_domain_row_returns_none(db):
    ptask = await _mk_platform_task(
        db, kind="train", payload_ref="train:ghost-id-9999"
    )
    detail = await svc.get_platform_task_detail(db, ptask.id)
    assert detail["domain_kind"] == "train"
    assert detail["domain"] is None  # row was never created / already deleted


def _patch_logs_dir(monkeypatch, path: Path) -> None:
    """Settings is a frozen dataclass — swap ``get_settings`` inside the
    detail service module with a shim returning a simple namespace."""
    from types import SimpleNamespace
    monkeypatch.setattr(
        svc, "get_settings", lambda: SimpleNamespace(storage_logs=path)
    )


async def test_detail_tails_log_file(db, dataset, tmp_path, monkeypatch):
    """The log tail should surface the last N lines of storage/logs/{id}.log."""
    _patch_logs_dir(monkeypatch, tmp_path)

    tt = TrainingTask(
        dataset_id=dataset.id, model_type="random_forest", target_column="y",
        status="SUCCESS",
    )
    db.add(tt)
    await db.flush()

    log_path: Path = tmp_path / f"{tt.id}.log"
    log_path.write_text("\n".join(f"line-{i}" for i in range(250)) + "\n")

    ptask = await _mk_platform_task(db, kind="train", payload_ref=f"train:{tt.id}")

    detail = await svc.get_platform_task_detail(db, ptask.id, log_limit=50)
    assert len(detail["recent_logs"]) == 50
    assert detail["recent_logs"][-1] == "line-249"
    assert detail["recent_logs"][0] == "line-200"


async def test_detail_log_tail_empty_when_file_missing(db, tmp_path, monkeypatch):
    _patch_logs_dir(monkeypatch, tmp_path)

    ptask = await _mk_platform_task(
        db, kind="train", payload_ref="train:no-such-task"
    )
    detail = await svc.get_platform_task_detail(db, ptask.id)
    assert detail["recent_logs"] == []

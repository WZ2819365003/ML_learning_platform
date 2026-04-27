"""Tests for the V3-native experiment_run_logs table introduced in v3.3.0.

Why this exists:
  Inspector logs used to depend on legacy training_tasks/training_logs.
  A perfectly reasonable `DELETE FROM training_tasks` would CASCADE-wipe
  every V3 run's logs. v3.3.0 added experiment_run_logs (FK to
  experiment_runs, NOT training_tasks) so wipes of the legacy table can
  no longer take V3 history with them.

Pytest fixture uses in-memory SQLite without PRAGMA foreign_keys=ON, so
ON DELETE CASCADE behaviour can't be exercised here — that is verified
end-to-end by the playwright milestone gate (15) on the real MySQL
stack.  These tests cover schema + isolation at the model level:
  - the table is registered, columns line up, round-trip works
  - the FK column points at experiment_runs, NOT training_tasks
    (the whole reason this table exists)
"""
from __future__ import annotations

from datetime import datetime, timezone

import pytest
from sqlalchemy import select

from app.models.database import (
    Dataset,
    ExperimentRun,
    ExperimentRunLog,
    PlatformExperiment,
)


@pytest.mark.asyncio
async def test_experiment_run_log_round_trip(session_factory):
    """Model exists, columns line up, basic round-trip works."""
    async with session_factory() as session:
        session.add(Dataset(
            id="ds-1", name="d.csv", file_path="/tmp/d.csv",
            file_size=1, row_count=1, column_count=1, columns_info={},
        ))
        session.add(PlatformExperiment(
            id="exp-1", name="e", objective_metric="accuracy",
            objective_direction="max", status="RUNNING", kind="train",
        ))
        session.add(ExperimentRun(id="run-1", experiment_id="exp-1", status="SUCCESS"))
        await session.commit()

        session.add(ExperimentRunLog(
            run_id="run-1", level="INFO", message="hello v3 native",
            extra={"step": 1}, created_at=datetime.now(timezone.utc),
        ))
        await session.commit()

        rows = (await session.execute(
            select(ExperimentRunLog).where(ExperimentRunLog.run_id == "run-1")
        )).scalars().all()
        assert len(rows) == 1
        assert rows[0].level == "INFO"
        assert rows[0].message == "hello v3 native"
        assert rows[0].extra == {"step": 1}


def test_fk_points_at_experiment_runs_not_training_tasks():
    """Schema-level guard: experiment_run_logs.run_id MUST FK to
    experiment_runs(id), not training_tasks(id).  If somebody ever
    refactors and accidentally re-points the FK back at the legacy
    table we would silently re-introduce the very bug v3.3.0 fixed.
    """
    fks = list(ExperimentRunLog.__table__.foreign_keys)
    assert len(fks) == 1, f"expected exactly one FK, got {len(fks)}"
    target = fks[0].column.table.name
    assert target == "experiment_runs", (
        f"experiment_run_logs.run_id FK points at {target!r} but must "
        f"point at 'experiment_runs' to be isolated from legacy wipes"
    )
    assert fks[0].ondelete == "CASCADE", (
        "FK should cascade so deleting a run deletes its logs"
    )


def test_indexes_present():
    """The query path is `WHERE run_id=? ORDER BY created_at` —
    composite index covers it. Loose run_id index also there for
    point lookups."""
    indexes = {idx.name for idx in ExperimentRunLog.__table__.indexes}
    assert "ix_experiment_run_logs_run_id" in indexes
    assert "ix_experiment_run_logs_run_created" in indexes

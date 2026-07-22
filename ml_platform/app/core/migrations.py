"""
Lightweight idempotent schema migrations for V3 workbench.

The project uses SQLAlchemy ``Base.metadata.create_all`` on startup, which only
creates *new* tables.  When we add columns to existing tables we need small
forward-only migrations.  We keep them here as plain SQL executed through the
async engine and guarded by ``information_schema`` lookups so they are safe to
run repeatedly.

Each migration is an independent function so failures are isolated and the
logs tell us exactly which step rolled back.
"""

from __future__ import annotations

import logging
from typing import Iterable

from sqlalchemy import text
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import AsyncEngine

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _column_exists(conn: Connection, table: str, column: str) -> bool:
    if conn.dialect.name == "sqlite":
        rows = conn.execute(text(f"PRAGMA table_info({table})")).mappings().all()
        return any(row["name"] == column for row in rows)

    row = conn.execute(
        text(
            "SELECT 1 FROM information_schema.COLUMNS "
            "WHERE TABLE_SCHEMA = DATABASE() "
            "  AND TABLE_NAME = :table "
            "  AND COLUMN_NAME = :column"
        ),
        {"table": table, "column": column},
    ).first()
    return row is not None


def _index_exists(conn: Connection, table: str, index: str) -> bool:
    if conn.dialect.name == "sqlite":
        rows = conn.execute(text(f"PRAGMA index_list({table})")).mappings().all()
        return any(row["name"] == index for row in rows)

    row = conn.execute(
        text(
            "SELECT 1 FROM information_schema.STATISTICS "
            "WHERE TABLE_SCHEMA = DATABASE() "
            "  AND TABLE_NAME = :table "
            "  AND INDEX_NAME = :index"
        ),
        {"table": table, "index": index},
    ).first()
    return row is not None


def _table_exists(conn: Connection, table: str) -> bool:
    if conn.dialect.name == "sqlite":
        row = conn.execute(
            text("SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = :table"),
            {"table": table},
        ).first()
        return row is not None

    row = conn.execute(
        text(
            "SELECT 1 FROM information_schema.TABLES "
            "WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = :table"
        ),
        {"table": table},
    ).first()
    return row is not None


def _add_column_if_missing(
    conn: Connection, table: str, column: str, ddl: str
) -> None:
    """Run ``ALTER TABLE ... ADD COLUMN`` if the column does not exist."""
    if not _table_exists(conn, table):
        logger.debug("Skip %s.%s — table not yet created", table, column)
        return
    if _column_exists(conn, table, column):
        return
    logger.info("Adding column %s.%s", table, column)
    conn.execute(text(f"ALTER TABLE {table} ADD COLUMN {column} {ddl}"))


def _add_index_if_missing(
    conn: Connection, table: str, index: str, columns: Iterable[str]
) -> None:
    if not _table_exists(conn, table):
        return
    if _index_exists(conn, table, index):
        return
    cols = ", ".join(columns)
    logger.info("Adding index %s on %s(%s)", index, table, cols)
    conn.execute(text(f"CREATE INDEX {index} ON {table} ({cols})"))


# ---------------------------------------------------------------------------
# Migration steps
# ---------------------------------------------------------------------------

def _migrate_platform_experiments(conn: Connection) -> None:
    """V3 strategy fields on platform_experiments."""
    _add_column_if_missing(
        conn, "platform_experiments", "modeling_task_id", "VARCHAR(36) NULL"
    )
    _add_column_if_missing(
        conn,
        "platform_experiments",
        "strategy_type",
        "VARCHAR(32) NOT NULL DEFAULT 'baseline'",
    )
    _add_column_if_missing(
        conn, "platform_experiments", "selected_models", "JSON NULL"
    )
    _add_column_if_missing(
        conn, "platform_experiments", "search_space", "JSON NULL"
    )
    _add_column_if_missing(
        conn, "platform_experiments", "budget_config", "JSON NULL"
    )
    _add_index_if_missing(
        conn,
        "platform_experiments",
        "ix_platform_experiments_modeling_task_id",
        ["modeling_task_id"],
    )


def _migrate_experiment_runs(conn: Connection) -> None:
    """V3 tuning metadata on experiment_runs."""
    _add_column_if_missing(conn, "experiment_runs", "trial_no", "INT NULL")
    _add_column_if_missing(conn, "experiment_runs", "search_meta", "JSON NULL")
    _add_column_if_missing(
        conn, "experiment_runs", "source_experiment_type", "VARCHAR(32) NULL"
    )
    # M2c terminal failure reason. Production gets this from Alembic 0003, but
    # local/dev databases still bootstrap through this path — without it an
    # existing SQLite file breaks with "no such column: error_message".
    _add_column_if_missing(conn, "experiment_runs", "error_message", "TEXT NULL")


def _backfill_modeling_tasks(conn: Connection) -> None:
    """
    Create one ``ModelingTask`` per existing ``PlatformExperiment`` that doesn't
    already have a parent.  Each legacy experiment becomes a modeling task of
    the same name; the experiment is attached as its only child.

    Strategy type is inferred from ``platform_experiments.kind``:
      - ``grid_search``  → ``grid_search``
      - ``automl``       → ``baseline`` (legacy AutoML is best treated as
                                         a baseline batch over multiple models)
      - anything else    → ``baseline``
    """
    if not _table_exists(conn, "modeling_tasks"):
        return
    if not _table_exists(conn, "platform_experiments"):
        return

    rows = conn.execute(
        text(
            "SELECT id, name, description, dataset_id, dataset_name, "
            "       dataset_version_id, objective_metric, objective_direction, "
            "       status, kind, config, created_at, updated_at, finished_at "
            "  FROM platform_experiments "
            " WHERE modeling_task_id IS NULL"
        )
    ).mappings().all()

    if not rows:
        return

    logger.info("Backfilling %d modeling_tasks from legacy experiments", len(rows))

    for row in rows:
        # Derive task_type and target_column from the first run's params if any
        run = conn.execute(
            text(
                "SELECT params FROM experiment_runs "
                " WHERE experiment_id = :eid ORDER BY created_at ASC LIMIT 1"
            ),
            {"eid": row["id"]},
        ).first()
        target_column = None
        task_type = "classification"
        if run and run[0]:
            import json

            try:
                params = run[0] if isinstance(run[0], dict) else json.loads(run[0])
            except Exception:  # noqa: BLE001
                params = {}
            target_column = params.get("target_column")
            tt = params.get("task_type")
            if tt in ("classification", "regression"):
                task_type = tt

        conn.execute(
            text(
                "INSERT INTO modeling_tasks "
                "  (id, name, description, dataset_id, dataset_name, "
                "   dataset_version_id, target_column, task_type, "
                "   objective_metric, objective_direction, status, config, "
                "   created_at, updated_at, finished_at) "
                "VALUES "
                "  (UUID(), :name, :description, :dataset_id, :dataset_name, "
                "   :dataset_version_id, :target_column, :task_type, "
                "   :objective_metric, :objective_direction, :status, NULL, "
                "   :created_at, :updated_at, :finished_at)"
            ),
            {
                "name": (row["name"] or "Untitled") + " (legacy)",
                "description": row["description"],
                "dataset_id": row["dataset_id"],
                "dataset_name": row["dataset_name"],
                "dataset_version_id": row["dataset_version_id"],
                "target_column": target_column,
                "task_type": task_type,
                "objective_metric": row["objective_metric"] or "accuracy",
                "objective_direction": row["objective_direction"] or "max",
                "status": row["status"] or "CREATED",
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
                "finished_at": row["finished_at"],
            },
        )

        # Fetch the new id (UUID() doesn't round-trip without another query).
        new_id_row = conn.execute(
            text(
                "SELECT id FROM modeling_tasks "
                " WHERE name = :name ORDER BY created_at DESC LIMIT 1"
            ),
            {"name": (row["name"] or "Untitled") + " (legacy)"},
        ).first()
        if new_id_row is None:
            continue
        new_task_id = new_id_row[0]

        # Link experiment to the new modeling task and derive strategy_type.
        legacy_kind = row["kind"] or "single"
        if legacy_kind == "grid_search":
            strategy = "grid_search"
        else:
            strategy = "baseline"

        conn.execute(
            text(
                "UPDATE platform_experiments "
                "   SET modeling_task_id = :tid, "
                "       strategy_type = :strategy "
                " WHERE id = :eid"
            ),
            {"tid": new_task_id, "strategy": strategy, "eid": row["id"]},
        )


def _backfill_experiment_runs(conn: Connection) -> None:
    """Fill source_experiment_type on runs from their parent experiment."""
    if not _table_exists(conn, "experiment_runs"):
        return
    if conn.dialect.name == "sqlite":
        conn.execute(
            text(
                "UPDATE experiment_runs "
                "   SET source_experiment_type = ("
                "       SELECT e.strategy_type FROM platform_experiments e "
                "        WHERE e.id = experiment_runs.experiment_id"
                "   ) "
                " WHERE source_experiment_type IS NULL"
            )
        )
    else:
        conn.execute(
            text(
                "UPDATE experiment_runs r "
                "  JOIN platform_experiments e ON e.id = r.experiment_id "
                "   SET r.source_experiment_type = e.strategy_type "
                " WHERE r.source_experiment_type IS NULL"
            )
        )


def _migrate_training_plans(conn: Connection) -> None:
    """V3 Phase 1 — DL integration fields on training_plans."""
    _add_column_if_missing(
        conn,
        "training_plans",
        "model_family",
        "VARCHAR(16) NOT NULL DEFAULT 'ml'",
    )
    _add_column_if_missing(conn, "training_plans", "dl_config", "JSON NULL")


def _migrate_training_plans_version(conn: Connection) -> None:
    """V3 Phase 2 — version counter for snapshot attribution."""
    _add_column_if_missing(
        conn,
        "training_plans",
        "version",
        "INT NOT NULL DEFAULT 1",
    )


def _migrate_modeling_tasks_plan_binding(conn: Connection) -> None:
    """V3 Phase 2 — bind ModelingTask to a TrainingPlan + capture snapshot."""
    _add_column_if_missing(
        conn, "modeling_tasks", "training_plan_id", "VARCHAR(36) NULL"
    )
    _add_column_if_missing(
        conn, "modeling_tasks", "training_plan_snapshot", "JSON NULL"
    )
    _add_index_if_missing(
        conn,
        "modeling_tasks",
        "ix_modeling_tasks_training_plan_id",
        ["training_plan_id"],
    )


def _migrate_platform_tasks_depends_on(conn: Connection) -> None:
    """V3 Phase 2 — DAG edge column; runtime gate turns on in Phase 5."""
    _add_column_if_missing(conn, "platform_tasks", "depends_on", "JSON NULL")


def _migrate_inference_jobs_batch(conn: Connection) -> None:
    """File-backed batch prediction columns (see Alembic 0005)."""
    _add_column_if_missing(conn, "inference_jobs", "input_path", "VARCHAR(1024) NULL")
    _add_column_if_missing(conn, "inference_jobs", "result_path", "VARCHAR(1024) NULL")
    _add_column_if_missing(conn, "inference_jobs", "processed_rows", "INT NOT NULL DEFAULT 0")


def _migrate_datasets_content_sha256(conn: Connection) -> None:
    """Content hash for dataset dedup/versioning.

    This column shipped as Alembic 0002 but was never mirrored here, so any
    database bootstrapped through the non-production path (create_all + these
    idempotent migrations) never got it. The result is a local/dev environment
    that cannot run the current code at all — every dataset query fails with
    ``Unknown column 'datasets.content_sha256'``. Production is unaffected
    because it goes through Alembic.
    """
    _add_column_if_missing(conn, "datasets", "content_sha256", "VARCHAR(64) NULL")


def _migrate_platform_tasks_attempt_token(conn: Connection) -> None:
    """Per-attempt identity for stalled-task recovery (see Alembic 0004)."""
    _add_column_if_missing(conn, "platform_tasks", "attempt_token", "VARCHAR(36) NULL")


def _migrate_training_tasks_cv_folds(conn: Connection) -> None:
    """Persist cross-validation folds on concrete classical-ML training tasks."""
    _add_column_if_missing(
        conn,
        "training_tasks",
        "cv_folds",
        "INT NOT NULL DEFAULT 5",
    )


# ---------------------------------------------------------------------------
# Public entrypoint
# ---------------------------------------------------------------------------

async def run_startup_migrations(engine: AsyncEngine) -> None:
    """Apply idempotent V3 workbench migrations.  Called from lifespan hook."""
    try:
        async with engine.begin() as conn:
            await conn.run_sync(_migrate_platform_experiments)
            await conn.run_sync(_migrate_experiment_runs)
            await conn.run_sync(_backfill_modeling_tasks)
            await conn.run_sync(_backfill_experiment_runs)
            await conn.run_sync(_migrate_training_plans)
            await conn.run_sync(_migrate_training_plans_version)
            await conn.run_sync(_migrate_modeling_tasks_plan_binding)
            await conn.run_sync(_migrate_datasets_content_sha256)
            await conn.run_sync(_migrate_inference_jobs_batch)
            await conn.run_sync(_migrate_platform_tasks_depends_on)
            await conn.run_sync(_migrate_platform_tasks_attempt_token)
            await conn.run_sync(_migrate_training_tasks_cv_folds)
    except Exception as exc:  # noqa: BLE001
        # Startup must never crash because of a migration hiccup; log and move on.
        logger.warning("V3 workbench migrations skipped (%s): %s", type(exc).__name__, exc)

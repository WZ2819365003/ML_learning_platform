"""Stamp Alembic for legacy databases created before migrations existed.

Early deployments could create tables directly from the SQLAlchemy metadata.
Those databases already contain business tables but have no ``alembic_version``
row, so ``alembic upgrade head`` tries to replay the frozen baseline and fails
with "table already exists". This script detects that state and stamps the
highest revision that is already reflected by the schema before the normal
upgrade runs.
"""

from __future__ import annotations

import sys
from pathlib import Path

import sqlalchemy as sa
from alembic import command
from alembic.config import Config


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.config import get_settings  # noqa: E402
from app.models.database import _to_sync_database_url  # noqa: E402


OWNER_TABLES = (
    "datasets",
    "training_tasks",
    "dl_training_tasks",
    "modeling_tasks",
    "training_plans",
    "ts_forecast_tasks",
    "ts_deployments",
)


def _has_columns(columns_by_table: dict[str, set[str]], table: str, *columns: str) -> bool:
    return set(columns).issubset(columns_by_table.get(table, set()))


def detect_legacy_revision(
    tables: set[str], columns_by_table: dict[str, set[str]]
) -> str | None:
    """Return the revision already represented by an unstamped legacy schema."""
    if "datasets" not in tables:
        return None

    revision = "0001"
    if not _has_columns(columns_by_table, "datasets", "content_sha256"):
        return revision

    revision = "0002"
    if not _has_columns(columns_by_table, "experiment_runs", "error_message"):
        return revision

    revision = "0003"
    if not _has_columns(columns_by_table, "platform_tasks", "attempt_token"):
        return revision

    revision = "0004"
    if not _has_columns(
        columns_by_table,
        "inference_jobs",
        "input_path",
        "result_path",
        "processed_rows",
    ):
        return revision

    revision = "0005"
    if "ai_report_archives" not in tables:
        return revision

    revision = "0006"
    if not all(
        table in tables and _has_columns(columns_by_table, table, "owner_username")
        for table in OWNER_TABLES
    ):
        return revision

    return "0007"


def _alembic_has_version_row(engine: sa.Engine) -> bool:
    inspector = sa.inspect(engine)
    if "alembic_version" not in inspector.get_table_names():
        return False
    with engine.connect() as connection:
        row = connection.execute(
            sa.text("SELECT version_num FROM alembic_version LIMIT 1")
        ).first()
    return row is not None and bool(row[0])


def main() -> int:
    database_url = _to_sync_database_url(get_settings().database_url)
    engine = sa.create_engine(database_url, future=True)
    try:
        if _alembic_has_version_row(engine):
            print("Alembic version already present; baseline stamp skipped.")
            return 0

        inspector = sa.inspect(engine)
        tables = set(inspector.get_table_names())
        columns_by_table = {
            table: {column["name"] for column in inspector.get_columns(table)}
            for table in tables
        }
        revision = detect_legacy_revision(tables, columns_by_table)
        if revision is None:
            print("No legacy business schema found; baseline stamp skipped.")
            return 0
    finally:
        engine.dispose()

    config = Config(str(PROJECT_ROOT / "alembic.ini"))
    print(f"Stamping legacy schema as Alembic revision {revision}.")
    command.stamp(config, revision)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import sqlite3
from pathlib import Path

from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, inspect

from app.config import get_settings
from app.models.database import Base


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _alembic_config() -> Config:
    return Config(str(PROJECT_ROOT / "alembic.ini"))


def test_frozen_baseline_then_incremental_upgrade_preserves_data(tmp_path, monkeypatch):
    database_path = tmp_path / "migration.db"
    monkeypatch.setenv("DATABASE_URL", f"sqlite+aiosqlite:///{database_path}")
    get_settings.cache_clear()

    try:
        command.upgrade(_alembic_config(), "0001")

        connection = sqlite3.connect(database_path)
        columns_at_0001 = {
            row[1] for row in connection.execute("PRAGMA table_info(datasets)")
        }
        assert "content_sha256" not in columns_at_0001, (
            "0001 baseline leaked datasets.content_sha256 from the current ORM"
        )
        connection.execute(
            "INSERT INTO datasets "
            "(id, name, file_path, file_size, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            ("legacy-dataset", "legacy.csv", "/tmp/legacy.csv", 12, "2026-01-01", "2026-01-01"),
        )
        connection.commit()
        connection.close()

        command.upgrade(_alembic_config(), "head")

        engine = create_engine(f"sqlite:///{database_path}")
        db_inspector = inspect(engine)
        actual_tables = set(db_inspector.get_table_names()) - {"alembic_version"}
        expected_tables = set(Base.metadata.tables)
        assert actual_tables == expected_tables
        for table_name, table in Base.metadata.tables.items():
            actual_columns = {column["name"] for column in db_inspector.get_columns(table_name)}
            assert actual_columns == set(table.columns.keys()), f"column mismatch for {table_name}"

        with engine.connect() as connection:
            row = connection.exec_driver_sql(
                "SELECT name, content_sha256 FROM datasets WHERE id = ?",
                ("legacy-dataset",),
            ).one()
        assert row == ("legacy.csv", None)
        engine.dispose()
    finally:
        get_settings.cache_clear()

"""add owner_username to user-owned assets

Revision ID: 0007
Revises: 0006
Create Date: 2026-07-27

Core assets now carry the account that created them. Existing rows are
backfilled to AUTH_USERNAME (or admin) so enabling auth keeps current demo data
visible to the configured release account.
"""

from __future__ import annotations

import os
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "0007"
down_revision: Union[str, None] = "0006"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


OWNER_TABLES = (
    "datasets",
    "training_tasks",
    "dl_training_tasks",
    "modeling_tasks",
    "training_plans",
    "ts_forecast_tasks",
    "ts_deployments",
)


def _table_exists(table_name: str) -> bool:
    return table_name in sa.inspect(op.get_bind()).get_table_names()


def _column_exists(table_name: str, column_name: str) -> bool:
    return column_name in {
        column["name"] for column in sa.inspect(op.get_bind()).get_columns(table_name)
    }


def _index_exists(table_name: str, index_name: str) -> bool:
    return index_name in {
        index["name"] for index in sa.inspect(op.get_bind()).get_indexes(table_name)
    }


def upgrade() -> None:
    owner = os.getenv("AUTH_USERNAME", "admin") or "admin"
    bind = op.get_bind()
    for table in OWNER_TABLES:
        if not _table_exists(table):
            continue
        if not _column_exists(table, "owner_username"):
            op.add_column(table, sa.Column("owner_username", sa.String(length=100), nullable=True))
        index_name = f"ix_{table}_owner_username"
        if not _index_exists(table, index_name):
            op.create_index(index_name, table, ["owner_username"])
        bind.execute(
            sa.text(f"UPDATE {table} SET owner_username = :owner WHERE owner_username IS NULL"),
            {"owner": owner},
        )


def downgrade() -> None:
    for table in reversed(OWNER_TABLES):
        if not _table_exists(table) or not _column_exists(table, "owner_username"):
            continue
        index_name = f"ix_{table}_owner_username"
        if _index_exists(table, index_name):
            op.drop_index(index_name, table_name=table)
        op.drop_column(table, "owner_username")

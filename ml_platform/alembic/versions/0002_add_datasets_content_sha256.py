"""add datasets.content_sha256

Revision ID: 0002
Revises: 0001
Create Date: 2026-07-21 13:51:04.658990
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = '0002'
down_revision: Union[str, None] = '0001'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    columns = {
        column["name"]
        for column in sa.inspect(op.get_bind()).get_columns("datasets")
    }
    if "content_sha256" not in columns:
        op.add_column(
            "datasets",
            sa.Column("content_sha256", sa.String(length=64), nullable=True),
        )


def downgrade() -> None:
    columns = {
        column["name"]
        for column in sa.inspect(op.get_bind()).get_columns("datasets")
    }
    if "content_sha256" in columns:
        op.drop_column("datasets", "content_sha256")

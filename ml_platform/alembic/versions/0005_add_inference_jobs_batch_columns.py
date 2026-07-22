"""add inference_jobs batch columns

Revision ID: 0005
Revises: 0004
Create Date: 2026-07-22

Batch prediction reads and writes files rather than JSON columns. A CSV with
100k rows cannot live in ``predictions`` (JSON) and cannot be streamed back to
the caller from there, so file-backed jobs record paths instead.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "0005"
down_revision: Union[str, None] = "0004"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("inference_jobs", sa.Column("input_path", sa.String(length=1024), nullable=True))
    op.add_column("inference_jobs", sa.Column("result_path", sa.String(length=1024), nullable=True))
    op.add_column(
        "inference_jobs",
        sa.Column("processed_rows", sa.Integer(), nullable=False, server_default="0"),
    )


def downgrade() -> None:
    op.drop_column("inference_jobs", "processed_rows")
    op.drop_column("inference_jobs", "result_path")
    op.drop_column("inference_jobs", "input_path")

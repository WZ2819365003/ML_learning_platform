"""add experiment_runs.error_message

Revision ID: 0003
Revises: 0002
Create Date: 2026-07-21

M2c gives V3 trials a dedicated terminal-failure column. Previously the only
free-text field was ``notes`` (user-facing), so worker failures had nowhere
structured to land.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "0003"
down_revision: Union[str, None] = "0002"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "experiment_runs",
        sa.Column("error_message", sa.Text(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("experiment_runs", "error_message")

"""add platform_tasks.attempt_token

Revision ID: 0004
Revises: 0003
Create Date: 2026-07-21

Stalled-task recovery needs to identify a specific execution attempt so it can
compare-and-set against the row it actually observed. Neither existing column
works: Celery reuses one task id across retries, and ``started_at`` is stamped
only on the first claim. Without a real attempt token, recovery can reset a
task that was legitimately re-claimed after the scan.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "0004"
down_revision: Union[str, None] = "0003"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "platform_tasks",
        sa.Column("attempt_token", sa.String(length=36), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("platform_tasks", "attempt_token")

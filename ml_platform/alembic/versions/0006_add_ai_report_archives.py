"""add ai_report_archives

Revision ID: 0006
Revises: 0005
Create Date: 2026-07-27

AI reports are archived per ModelingTask so the task page can show the latest
generated report and let users open historical generated content.
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "0006"
down_revision: Union[str, None] = "0005"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _table_exists(table_name: str) -> bool:
    return table_name in sa.inspect(op.get_bind()).get_table_names()


def _index_exists(table_name: str, index_name: str) -> bool:
    return index_name in {
        index["name"] for index in sa.inspect(op.get_bind()).get_indexes(table_name)
    }


def upgrade() -> None:
    if not _table_exists("ai_report_archives"):
        op.create_table(
            "ai_report_archives",
            sa.Column("id", sa.String(length=36), nullable=False),
            sa.Column("task_id", sa.String(length=36), nullable=False),
            sa.Column("title", sa.String(length=255), nullable=False),
            sa.Column("model", sa.String(length=128), nullable=True),
            sa.Column("source", sa.String(length=32), nullable=False),
            sa.Column("markdown", sa.Text(), nullable=False),
            sa.Column("payload", sa.JSON(), nullable=True),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
            sa.ForeignKeyConstraint(["task_id"], ["modeling_tasks.id"], ondelete="CASCADE"),
            sa.PrimaryKeyConstraint("id"),
        )
    if not _index_exists("ai_report_archives", "ix_ai_report_archives_task_id"):
        op.create_index(
            "ix_ai_report_archives_task_id",
            "ai_report_archives",
            ["task_id"],
        )
    if not _index_exists("ai_report_archives", "ix_ai_report_archives_created_at"):
        op.create_index(
            "ix_ai_report_archives_created_at",
            "ai_report_archives",
            ["created_at"],
        )


def downgrade() -> None:
    if _table_exists("ai_report_archives"):
        if _index_exists("ai_report_archives", "ix_ai_report_archives_created_at"):
            op.drop_index("ix_ai_report_archives_created_at", table_name="ai_report_archives")
        if _index_exists("ai_report_archives", "ix_ai_report_archives_task_id"):
            op.drop_index("ix_ai_report_archives_task_id", table_name="ai_report_archives")
        op.drop_table("ai_report_archives")

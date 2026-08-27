"""add ensemble_deployments and ensemble_members

Revision ID: 0008
Revises: 0007
Create Date: 2026-08-26

Weighted multi-model deployments. A single deployment cannot represent one:
`model_deployments.task_id` is a NOT NULL FK to training_tasks, and DL models
live in a separate table behind their own deployment table, so no existing row
can point at both an xgboost and an lstm.

Members carry two nullable FKs rather than a (family, id) string pair, so
deleting a trained model cascades its membership away instead of leaving a
dangling row that only fails at inference time.
"""

from __future__ import annotations

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "0008"
down_revision: Union[str, None] = "0007"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "ensemble_deployments",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("owner_username", sa.String(100), nullable=True),
        sa.Column(
            "modeling_task_id",
            sa.String(36),
            sa.ForeignKey("modeling_tasks.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        # Only weighted_average exists today; the column is what lets a
        # per-sample strategy be added later without another migration.
        sa.Column("strategy", sa.String(32), nullable=False, server_default="weighted_average"),
        sa.Column("task_type", sa.String(32), nullable=False),
        sa.Column("status", sa.String(32), nullable=False, server_default="active"),
        sa.Column("request_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index(
        "ix_ensemble_deployments_modeling_task_id",
        "ensemble_deployments",
        ["modeling_task_id"],
    )

    op.create_table(
        "ensemble_members",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column(
            "ensemble_id",
            sa.String(36),
            sa.ForeignKey("ensemble_deployments.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("run_id", sa.String(36), nullable=True),
        sa.Column(
            "ml_task_id",
            sa.String(36),
            sa.ForeignKey("training_tasks.id", ondelete="CASCADE"),
            nullable=True,
        ),
        sa.Column(
            "dl_task_id",
            sa.String(36),
            sa.ForeignKey("dl_training_tasks.id", ondelete="CASCADE"),
            nullable=True,
        ),
        sa.Column("model_type", sa.String(64), nullable=True),
        sa.Column("weight", sa.Float(), nullable=False, server_default="0"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("ix_ensemble_members_ensemble_id", "ensemble_members", ["ensemble_id"])


def downgrade() -> None:
    op.drop_index("ix_ensemble_members_ensemble_id", table_name="ensemble_members")
    op.drop_table("ensemble_members")
    op.drop_index("ix_ensemble_deployments_modeling_task_id", table_name="ensemble_deployments")
    op.drop_table("ensemble_deployments")

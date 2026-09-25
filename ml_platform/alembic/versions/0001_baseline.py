"""Frozen M0 schema baseline (before datasets.content_sha256).

Revision ID: 0001
Revises:
Create Date: 2026-07-21

This migration intentionally contains a literal DDL snapshot.  Importing the
live ORM here would make the baseline change whenever models change and would
silently absorb later revisions on fresh databases.
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "0001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "datasets",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("file_path", sa.String(length=1024), nullable=False),
        sa.Column("file_size", sa.BigInteger(), nullable=False),
        sa.Column("row_count", sa.Integer(), nullable=True),
        sa.Column("column_count", sa.Integer(), nullable=True),
        sa.Column("columns_info", sa.JSON(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_table(
        "model_tag_library",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("name", sa.String(length=100), nullable=False),
        sa.Column("dimension", sa.String(length=50), nullable=True),
        sa.Column("color", sa.String(length=30), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("name"),
    )
    op.create_table(
        "platform_tasks",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("kind", sa.String(length=32), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("priority", sa.Integer(), nullable=False),
        sa.Column("celery_task_id", sa.String(length=255), nullable=True),
        sa.Column("worker_id", sa.String(length=255), nullable=True),
        sa.Column("retry_count", sa.Integer(), nullable=False),
        sa.Column("max_retries", sa.Integer(), nullable=False),
        sa.Column("payload_ref", sa.String(length=512), nullable=True),
        sa.Column("depends_on", sa.JSON(), nullable=True),
        sa.Column("progress", sa.Float(), nullable=False),
        sa.Column("logs_uri", sa.String(length=1024), nullable=True),
        sa.Column("metrics_uri", sa.String(length=1024), nullable=True),
        sa.Column("artifacts_uri", sa.String(length=1024), nullable=True),
        sa.Column("metrics_snapshot", sa.JSON(), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("queued_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_platform_tasks_created_at", "platform_tasks", ["created_at"])
    op.create_index("ix_platform_tasks_kind", "platform_tasks", ["kind"])
    op.create_index("ix_platform_tasks_status", "platform_tasks", ["status"])
    op.create_table(
        "training_plans",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("task_type", sa.String(length=32), nullable=False),
        sa.Column("strategy_type", sa.String(length=32), nullable=False),
        sa.Column("model_family", sa.String(length=16), nullable=False),
        sa.Column("selected_models", sa.JSON(), nullable=True),
        sa.Column("search_space", sa.JSON(), nullable=True),
        sa.Column("dl_config", sa.JSON(), nullable=True),
        sa.Column("budget_config", sa.JSON(), nullable=True),
        sa.Column("eval_metrics", sa.JSON(), nullable=True),
        sa.Column("default_objective_metric", sa.String(length=64), nullable=True),
        sa.Column("default_objective_direction", sa.String(length=8), nullable=True),
        sa.Column("use_count", sa.Integer(), nullable=False),
        sa.Column("last_used_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("version", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_training_plans_created_at", "training_plans", ["created_at"])
    op.create_index("ix_training_plans_task_type", "training_plans", ["task_type"])
    op.create_table(
        "ts_deployments",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("backend_label", sa.String(length=128), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("request_count", sa.Integer(), nullable=False),
        sa.Column("config", sa.JSON(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_table(
        "ts_forecast_tasks",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("dataset_id", sa.String(length=36), nullable=False),
        sa.Column("dataset_name", sa.String(length=255), nullable=True),
        sa.Column("value_column", sa.String(length=255), nullable=False),
        sa.Column("time_column", sa.String(length=255), nullable=True),
        sa.Column("horizon", sa.Integer(), nullable=False),
        sa.Column("frequency", sa.String(length=32), nullable=False),
        sa.Column("model_name", sa.String(length=128), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("result", sa.JSON(), nullable=True),
        sa.Column("notes", sa.Text(), nullable=True),
        sa.Column("tags", sa.JSON(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_ts_forecast_tasks_created_at", "ts_forecast_tasks", ["created_at"])
    op.create_index("ix_ts_forecast_tasks_status", "ts_forecast_tasks", ["status"])
    op.create_table(
        "dataset_versions",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("dataset_id", sa.String(length=36), nullable=False),
        sa.Column("version", sa.Integer(), nullable=False),
        sa.Column("file_uri", sa.String(length=1024), nullable=False),
        sa.Column("row_count", sa.Integer(), nullable=True),
        sa.Column("schema_hash", sa.String(length=64), nullable=True),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("parent_version_id", sa.String(length=36), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["dataset_id"], ["datasets.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(
            ["parent_version_id"], ["dataset_versions.id"], ondelete="SET NULL"
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_table(
        "dl_training_tasks",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("name", sa.String(length=200), nullable=True),
        sa.Column("dataset_id", sa.String(length=36), nullable=False),
        sa.Column("target_column", sa.String(length=255), nullable=False),
        sa.Column("model_type", sa.String(length=64), nullable=False),
        sa.Column("task_type", sa.String(length=32), nullable=False),
        sa.Column("arch_config", sa.JSON(), nullable=True),
        sa.Column("opt_config", sa.JSON(), nullable=True),
        sa.Column("train_config", sa.JSON(), nullable=True),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("progress", sa.Float(), nullable=False),
        sa.Column("current_epoch", sa.Integer(), nullable=False),
        sa.Column("total_epochs", sa.Integer(), nullable=False),
        sa.Column("result_metrics", sa.JSON(), nullable=True),
        sa.Column("model_path", sa.String(length=1024), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("notes", sa.Text(), nullable=True),
        sa.Column("tags", sa.JSON(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(["dataset_id"], ["datasets.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_table(
        "training_tasks",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("name", sa.String(length=200), nullable=True),
        sa.Column("dataset_id", sa.String(length=36), nullable=False),
        sa.Column("model_type", sa.String(length=128), nullable=False),
        sa.Column("hyperparameters", sa.JSON(), nullable=True),
        sa.Column("target_column", sa.String(length=255), nullable=False),
        sa.Column("test_size", sa.Float(), nullable=False),
        sa.Column("cv_folds", sa.Integer(), nullable=False),
        sa.Column("eval_metrics", sa.JSON(), nullable=True),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("celery_task_id", sa.String(length=255), nullable=True),
        sa.Column("progress", sa.Float(), nullable=False),
        sa.Column("result_metrics", sa.JSON(), nullable=True),
        sa.Column("model_path", sa.String(length=1024), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("notes", sa.Text(), nullable=True),
        sa.Column("tags", sa.JSON(), nullable=True),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["dataset_id"], ["datasets.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_table(
        "dl_model_deployments",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("dl_task_id", sa.String(length=36), nullable=False),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("request_count", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["dl_task_id"], ["dl_training_tasks.id"], ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_table(
        "dl_training_epochs",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("task_id", sa.String(length=36), nullable=False),
        sa.Column("epoch", sa.Integer(), nullable=False),
        sa.Column("total_epochs", sa.Integer(), nullable=False),
        sa.Column("train_loss", sa.Float(), nullable=True),
        sa.Column("val_loss", sa.Float(), nullable=True),
        sa.Column("val_acc", sa.Float(), nullable=True),
        sa.Column("val_f1_macro", sa.Float(), nullable=True),
        sa.Column("val_rmse", sa.Float(), nullable=True),
        sa.Column("val_mae", sa.Float(), nullable=True),
        sa.Column("val_r2", sa.Float(), nullable=True),
        sa.Column("lr", sa.Float(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["task_id"], ["dl_training_tasks.id"], ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_table(
        "dl_training_logs",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("task_id", sa.String(length=36), nullable=False),
        sa.Column("level", sa.String(length=16), nullable=False),
        sa.Column("message", sa.Text(), nullable=False),
        sa.Column("extra", sa.JSON(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["task_id"], ["dl_training_tasks.id"], ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_table(
        "model_deployments",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("task_id", sa.String(length=36), nullable=False),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("max_batch_size", sa.Integer(), nullable=False),
        sa.Column("request_count", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["task_id"], ["training_tasks.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_table(
        "modeling_tasks",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("dataset_id", sa.String(length=36), nullable=True),
        sa.Column("dataset_name", sa.String(length=255), nullable=True),
        sa.Column("dataset_version_id", sa.String(length=36), nullable=True),
        sa.Column("target_column", sa.String(length=255), nullable=True),
        sa.Column("task_type", sa.String(length=32), nullable=False),
        sa.Column("objective_metric", sa.String(length=64), nullable=False),
        sa.Column("objective_direction", sa.String(length=8), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("best_experiment_id", sa.String(length=36), nullable=True),
        sa.Column("best_run_id", sa.String(length=36), nullable=True),
        sa.Column("config", sa.JSON(), nullable=True),
        sa.Column("summary_snapshot", sa.JSON(), nullable=True),
        sa.Column("training_plan_id", sa.String(length=36), nullable=True),
        sa.Column("training_plan_snapshot", sa.JSON(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(
            ["dataset_version_id"], ["dataset_versions.id"], ondelete="SET NULL"
        ),
        sa.ForeignKeyConstraint(
            ["training_plan_id"], ["training_plans.id"], ondelete="SET NULL"
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_modeling_tasks_created_at", "modeling_tasks", ["created_at"])
    op.create_index("ix_modeling_tasks_dataset_id", "modeling_tasks", ["dataset_id"])
    op.create_index("ix_modeling_tasks_status", "modeling_tasks", ["status"])
    op.create_index(
        op.f("ix_modeling_tasks_training_plan_id"),
        "modeling_tasks",
        ["training_plan_id"],
    )
    op.create_table(
        "training_logs",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("task_id", sa.String(length=36), nullable=False),
        sa.Column("level", sa.String(length=16), nullable=False),
        sa.Column("message", sa.Text(), nullable=False),
        sa.Column("extra", sa.JSON(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["task_id"], ["training_tasks.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_table(
        "inference_jobs",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("deployment_id", sa.String(length=36), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("input_rows", sa.Integer(), nullable=False),
        sa.Column("predictions", sa.JSON(), nullable=True),
        sa.Column("probabilities", sa.JSON(), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(
            ["deployment_id"], ["model_deployments.id"], ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_table(
        "platform_experiments",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("modeling_task_id", sa.String(length=36), nullable=True),
        sa.Column("dataset_id", sa.String(length=36), nullable=True),
        sa.Column("dataset_name", sa.String(length=255), nullable=True),
        sa.Column("dataset_version_id", sa.String(length=36), nullable=True),
        sa.Column("objective_metric", sa.String(length=64), nullable=False),
        sa.Column("objective_direction", sa.String(length=8), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("kind", sa.String(length=32), nullable=False),
        sa.Column("strategy_type", sa.String(length=32), nullable=False),
        sa.Column("selected_models", sa.JSON(), nullable=True),
        sa.Column("search_space", sa.JSON(), nullable=True),
        sa.Column("budget_config", sa.JSON(), nullable=True),
        sa.Column("best_run_id", sa.String(length=36), nullable=True),
        sa.Column("config", sa.JSON(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(
            ["dataset_version_id"], ["dataset_versions.id"], ondelete="SET NULL"
        ),
        sa.ForeignKeyConstraint(
            ["modeling_task_id"], ["modeling_tasks.id"], ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_platform_experiments_created_at", "platform_experiments", ["created_at"]
    )
    op.create_index(
        "ix_platform_experiments_status", "platform_experiments", ["status"]
    )
    op.create_table(
        "experiment_runs",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("experiment_id", sa.String(length=36), nullable=False),
        sa.Column("task_id", sa.String(length=36), nullable=True),
        sa.Column("parent_run_id", sa.String(length=36), nullable=True),
        sa.Column("params", sa.JSON(), nullable=True),
        sa.Column("metrics", sa.JSON(), nullable=True),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("rank", sa.Integer(), nullable=True),
        sa.Column("artifacts_uri", sa.String(length=1024), nullable=True),
        sa.Column("notes", sa.Text(), nullable=True),
        sa.Column("trial_no", sa.Integer(), nullable=True),
        sa.Column("search_meta", sa.JSON(), nullable=True),
        sa.Column("source_experiment_type", sa.String(length=32), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(
            ["experiment_id"], ["platform_experiments.id"], ondelete="CASCADE"
        ),
        sa.ForeignKeyConstraint(
            ["parent_run_id"], ["experiment_runs.id"], ondelete="SET NULL"
        ),
        sa.ForeignKeyConstraint(["task_id"], ["platform_tasks.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_experiment_runs_experiment_id", "experiment_runs", ["experiment_id"])
    op.create_index("ix_experiment_runs_status", "experiment_runs", ["status"])
    op.create_table(
        "experiment_run_logs",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("run_id", sa.String(length=36), nullable=False),
        sa.Column("level", sa.String(length=16), nullable=False),
        sa.Column("message", sa.Text(), nullable=False),
        sa.Column("extra", sa.JSON(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["run_id"], ["experiment_runs.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_experiment_run_logs_run_created",
        "experiment_run_logs",
        ["run_id", "created_at"],
    )
    op.create_index("ix_experiment_run_logs_run_id", "experiment_run_logs", ["run_id"])


def downgrade() -> None:
    op.drop_index("ix_experiment_run_logs_run_id", table_name="experiment_run_logs")
    op.drop_index("ix_experiment_run_logs_run_created", table_name="experiment_run_logs")
    op.drop_table("experiment_run_logs")
    op.drop_index("ix_experiment_runs_status", table_name="experiment_runs")
    op.drop_index("ix_experiment_runs_experiment_id", table_name="experiment_runs")
    op.drop_table("experiment_runs")
    op.drop_index("ix_platform_experiments_status", table_name="platform_experiments")
    op.drop_index("ix_platform_experiments_created_at", table_name="platform_experiments")
    op.drop_table("platform_experiments")
    op.drop_table("inference_jobs")
    op.drop_table("training_logs")
    op.drop_index(op.f("ix_modeling_tasks_training_plan_id"), table_name="modeling_tasks")
    op.drop_index("ix_modeling_tasks_status", table_name="modeling_tasks")
    op.drop_index("ix_modeling_tasks_dataset_id", table_name="modeling_tasks")
    op.drop_index("ix_modeling_tasks_created_at", table_name="modeling_tasks")
    op.drop_table("modeling_tasks")
    op.drop_table("model_deployments")
    op.drop_table("dl_training_logs")
    op.drop_table("dl_training_epochs")
    op.drop_table("dl_model_deployments")
    op.drop_table("training_tasks")
    op.drop_table("dl_training_tasks")
    op.drop_table("dataset_versions")
    op.drop_index("ix_ts_forecast_tasks_status", table_name="ts_forecast_tasks")
    op.drop_index("ix_ts_forecast_tasks_created_at", table_name="ts_forecast_tasks")
    op.drop_table("ts_forecast_tasks")
    op.drop_table("ts_deployments")
    op.drop_index("ix_training_plans_task_type", table_name="training_plans")
    op.drop_index("ix_training_plans_created_at", table_name="training_plans")
    op.drop_table("training_plans")
    op.drop_index("ix_platform_tasks_status", table_name="platform_tasks")
    op.drop_index("ix_platform_tasks_kind", table_name="platform_tasks")
    op.drop_index("ix_platform_tasks_created_at", table_name="platform_tasks")
    op.drop_table("platform_tasks")
    op.drop_table("model_tag_library")
    op.drop_table("datasets")

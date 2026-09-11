"""日志排序：加 seq 列，created_at 提到微秒精度

Revision ID: 0009
Revises: 0008
Create Date: 2026-09-11

线上现象：一个 3 秒跑完的训练任务，日志面板里 [Step 2/5] 排在 [Step 1/5] 前面、
"Status: SUCCESS" 排在 [Step 4/5] 前面。

根因不在写入端——CV 折循环是单线程顺序 for，日志本来就是按序发出的。问题是
``created_at`` 在 MySQL 上是 ``DATETIME``（fsp=0），Python 端 ``datetime.now()``
的微秒被静默截断。任务只跑了几秒，实测一个时间戳上最多并列 8 条，而
``ORDER BY created_at`` 对并列值不保证任何顺序（主键是随机 UUID，兜不住底）。

两手都要：
  * ``created_at`` 提到 DATETIME(6)，让时间戳本身别再说谎；
  * 加 ``seq``（写入方单调递增）作为权威顺序键——挂钟会被 NTP 往回拨，批量
    flush 失败重试时也会把条目重新拼接，这些都不影响 seq。

历史数据的 seq 一律为 0，读回来仍然只能按秒排序；这是既成事实，不回填。
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import mysql

revision: str = "0009"
down_revision: Union[str, None] = "0008"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

_LOG_TABLES = ("training_logs", "experiment_run_logs", "dl_training_logs")


def upgrade() -> None:
    bind = op.get_bind()
    is_mysql = bind.dialect.name == "mysql"

    for table in _LOG_TABLES:
        # server_default 留着不动：SQLite 不支持 ALTER COLUMN ... DROP DEFAULT，
        # 而这个默认值本身无害——应用写入时一律显式带 seq，0 只是历史行的填充值。
        op.add_column(
            table,
            sa.Column("seq", sa.Integer(), nullable=False, server_default="0"),
        )

        if is_mysql:
            # SQLite 的 DATETIME 存的是 ISO 字符串，本来就带微秒，不需要改。
            op.alter_column(
                table,
                "created_at",
                type_=mysql.DATETIME(fsp=6),
                existing_type=mysql.DATETIME(),
                existing_nullable=False,
            )

    # 日志查询一律是 "按 run/task 过滤 + 按 (created_at, seq) 排序"，
    # 把 seq 带进复合索引，避免排序退化成 filesort。
    op.create_index(
        "ix_experiment_run_logs_run_created_seq",
        "experiment_run_logs",
        ["run_id", "created_at", "seq"],
    )
    op.create_index(
        "ix_training_logs_task_created_seq",
        "training_logs",
        ["task_id", "created_at", "seq"],
    )
    op.create_index(
        "ix_dl_training_logs_task_created_seq",
        "dl_training_logs",
        ["task_id", "created_at", "seq"],
    )


def downgrade() -> None:
    bind = op.get_bind()
    is_mysql = bind.dialect.name == "mysql"

    op.drop_index("ix_dl_training_logs_task_created_seq", table_name="dl_training_logs")
    op.drop_index("ix_training_logs_task_created_seq", table_name="training_logs")
    op.drop_index("ix_experiment_run_logs_run_created_seq", table_name="experiment_run_logs")

    for table in _LOG_TABLES:
        if is_mysql:
            op.alter_column(
                table,
                "created_at",
                type_=mysql.DATETIME(),
                existing_type=mysql.DATETIME(fsp=6),
                existing_nullable=False,
            )
        op.drop_column(table, "seq")

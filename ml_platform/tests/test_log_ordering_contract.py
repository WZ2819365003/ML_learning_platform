"""日志必须能按发出顺序读回来。

线上出现过一次：一个 3 秒跑完的训练任务，日志面板里 [Step 2/5] 排在 [Step 1/5]
前面、"Status: SUCCESS" 排在 [Step 4/5] 前面。

根因不在写入端——CV 的折循环是单线程顺序 for，回调按 1..5 依次触发。问题是
``created_at`` 在 MySQL 上落成了 ``DATETIME``（fsp=0），微秒被静默截断；一个几秒
跑完的任务，几乎所有日志都挤在同一秒里，而 ``ORDER BY created_at`` 对这些并列值
没有任何次序可言（主键还是随机 UUID，兜底也兜不住）。实测一秒内最多 8 条并列。

这组测试锁住两件事：
  1. MySQL 上的 DDL 必须带亚秒精度；
  2. 查询必须有一个确定性的兜底排序键，不能只按时间排。
"""
from __future__ import annotations

import pathlib
import re

import pytest
from sqlalchemy.dialects import mysql
from sqlalchemy.schema import CreateTable

from app.models.database import Base, DLTrainingLog, ExperimentRunLog, TrainingLog

LOG_MODELS = [TrainingLog, ExperimentRunLog, DLTrainingLog]
APP_ROOT = pathlib.Path(__file__).resolve().parents[1] / "app"


def _mysql_ddl(model) -> str:
    return str(CreateTable(model.__table__).compile(dialect=mysql.dialect()))


@pytest.mark.parametrize("model", LOG_MODELS, ids=lambda m: m.__tablename__)
def test_log_created_at_keeps_subsecond_precision_on_mysql(model):
    """DATETIME 不带 fsp 就是秒级，微秒会被 MySQL 悄悄丢掉。"""
    ddl = _mysql_ddl(model)
    created_at_line = next(
        line for line in ddl.splitlines() if "created_at" in line
    )
    assert re.search(r"DATETIME\(\s*[1-6]\s*\)", created_at_line), (
        f"{model.__tablename__}.created_at 在 MySQL 上是秒级精度，"
        f"同一秒内的日志无法排序：{created_at_line.strip()}"
    )


@pytest.mark.parametrize("model", LOG_MODELS, ids=lambda m: m.__tablename__)
def test_log_tables_have_monotonic_sequence_column(model):
    """时间戳再精确也只是时钟读数；顺序要由一个单调递增的列说了算。"""
    assert "seq" in model.__table__.columns, (
        f"{model.__tablename__} 缺少 seq 列，排序只能依赖挂钟时间"
    )


def test_every_log_query_orders_by_the_sequence_column():
    """只按 created_at 排序 = 并列行的顺序由存储引擎随意决定。

    这里扫源码而不是跑查询：排序漏加兜底键是"看着对、跑起来偶尔不对"的那类
    问题，SQLite 上还偏偏能蒙混过关（它按 rowid 稳定返回），只有 MySQL 才暴露。
    """
    offenders: list[str] = []
    pattern = re.compile(
        r"order_by\(\s*(TrainingLog|ExperimentRunLog|DLTrainingLog)\.created_at"
        r"(?:\.(?:asc|desc)\(\))?\s*\)"
    )
    for path in APP_ROOT.rglob("*.py"):
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if pattern.search(line):
                offenders.append(f"{path.relative_to(APP_ROOT)}:{lineno}: {line.strip()}")

    assert not offenders, (
        "这些查询只按 created_at 排序，同一时间戳的日志顺序不确定：\n  "
        + "\n  ".join(offenders)
    )

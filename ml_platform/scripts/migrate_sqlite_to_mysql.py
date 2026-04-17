"""
一次性迁移脚本：将现有 SQLite 数据迁移到 MySQL。

用法（在 ml_platform/ 目录下执行）：
    python scripts/migrate_sqlite_to_mysql.py

依赖：
    pip install aiomysql aiosqlite sqlalchemy

迁移顺序（按外键依赖）：
    model_tag_library → datasets → training_tasks → training_logs
    → model_deployments → inference_jobs → dl_training_tasks
    → dl_training_epochs → dl_training_logs → dl_model_deployments
    → ts_deployments → ts_forecast_tasks
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

# 确保能 import app.*
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import sqlalchemy as sa
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession

SQLITE_URL = "sqlite+aiosqlite:///./storage/ml_platform.db"
MYSQL_URL  = "mysql+aiomysql://root:123456@localhost:3307/ml_platform"

# 迁移顺序：父表在前，子表在后
TABLES = [
    "model_tag_library",
    "datasets",
    "training_tasks",
    "training_logs",
    "model_deployments",
    "inference_jobs",
    "dl_training_tasks",
    "dl_training_epochs",
    "dl_training_logs",
    "dl_model_deployments",
    "ts_deployments",
    "ts_forecast_tasks",
]


def _coerce_row(row: dict) -> dict:
    """将 SQLite 返回的原始值转换为 MySQL 兼容格式。

    aiomysql 不接受 dict/list 参数，JSON 列必须序列化为字符串。
    SQLite 里 JSON 列有时已经是字符串，有时是 Python 对象，统一处理。
    """
    result = {}
    for k, v in row.items():
        if isinstance(v, (dict, list)):
            # Python 对象 → JSON 字符串
            v = json.dumps(v, ensure_ascii=False)
        elif isinstance(v, str):
            # 保持原始字符串（SQLite 已经是 JSON 字符串格式）
            pass
        result[k] = v
    return result


async def migrate() -> None:
    sqlite_engine = create_async_engine(SQLITE_URL, echo=False)
    mysql_engine  = create_async_engine(MYSQL_URL,  echo=False)

    # 先在 MySQL 中建表（使用 SQLAlchemy ORM 模型）
    from app.models.database import Base
    async with mysql_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    print("✅ MySQL 表结构已创建")

    # 用反射从 SQLite 读取表结构
    async with sqlite_engine.connect() as src:
        for table_name in TABLES:
            # 检查 SQLite 表是否存在
            check = await src.execute(
                sa.text("SELECT name FROM sqlite_master WHERE type='table' AND name=:t"),
                {"t": table_name},
            )
            if check.fetchone() is None:
                print(f"  [SKIP] {table_name} — SQLite 中不存在该表")
                continue

            rows_result = await src.execute(sa.text(f"SELECT * FROM `{table_name}`"))
            rows = [dict(zip(rows_result.keys(), row)) for row in rows_result.fetchall()]

            if not rows:
                print(f"  [SKIP] {table_name} — 无数据")
                continue

            rows = [_coerce_row(r) for r in rows]

            # 写入 MySQL（INSERT IGNORE 幂等）
            async with mysql_engine.begin() as dst:
                # 禁用外键检查，避免顺序问题
                await dst.execute(sa.text("SET FOREIGN_KEY_CHECKS=0"))
                cols = list(rows[0].keys())
                placeholders = ", ".join(f":{c}" for c in cols)
                col_list = ", ".join(f"`{c}`" for c in cols)
                stmt = sa.text(
                    f"INSERT IGNORE INTO `{table_name}` ({col_list}) VALUES ({placeholders})"
                )
                await dst.execute(stmt, rows)
                await dst.execute(sa.text("SET FOREIGN_KEY_CHECKS=1"))

            print(f"  [OK]   {table_name} — 迁移 {len(rows)} 行")

    await sqlite_engine.dispose()
    await mysql_engine.dispose()
    print("\n✅ 迁移完成！")


if __name__ == "__main__":
    asyncio.run(migrate())

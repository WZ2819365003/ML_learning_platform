"""The MySQL seed file must not drift from the ORM.

`docker/mysql_init/01_data.sql` is loaded by MySQL on first start and contains
a *snapshot* of the schema plus demo rows. Alembic owns the schema everywhere
else, so the two are independent sources of truth for the same thing — and
nothing tells you when they disagree.

The failure mode is silent and severe: a fresh `docker compose up` builds
tables that the current code cannot query, and the first symptom is a runtime
`Unknown column …` on an endpoint that has nothing to do with deployment.
This test turns that into a loud, early failure.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from app.models.database import Base

SEED = Path(__file__).resolve().parents[2] / "docker" / "mysql_init" / "01_data.sql"


def _seed_tables() -> dict[str, set[str]]:
    """Parse `CREATE TABLE` blocks into {table: {column, ...}}."""
    sql = SEED.read_text(encoding="utf-8")
    tables: dict[str, set[str]] = {}
    # Accept both the hand-written form (backticked names) and DDL generated
    # from the ORM (bare names, optional IF NOT EXISTS) — the parser must not
    # be coupled to one formatting, or it reports "no tables" instead of drift.
    for match in re.finditer(
        r"CREATE TABLE (?:IF NOT EXISTS )?`?(\w+)`?\s*\((.*?)\n\)", sql, re.S,
    ):
        name, body = match.group(1), match.group(2)
        cols = set()
        for line in body.splitlines():
            line = line.strip().rstrip(",")
            col = re.match(r"`?(\w+)`?\s+(?!KEY\b)\w", line)
            if col and col.group(1).upper() not in {
                "PRIMARY", "UNIQUE", "KEY", "INDEX", "CONSTRAINT", "FOREIGN",
            }:
                cols.add(col.group(1))
        tables[name] = cols
    return tables


@pytest.mark.skipif(not SEED.exists(), reason="seed file not present")
def test_seed_schema_matches_orm():
    seed = _seed_tables()
    assert seed, "no CREATE TABLE parsed — the parser or the file shape changed"

    problems: list[str] = []
    for table in Base.metadata.sorted_tables:
        if table.name not in seed:
            problems.append(f"表缺失: {table.name}")
            continue
        orm_cols = {c.name for c in table.columns}
        missing = orm_cols - seed[table.name]
        if missing:
            problems.append(f"{table.name} 缺少列: {sorted(missing)}")

    assert not problems, (
        "docker/mysql_init/01_data.sql 与 ORM 不一致，全新部署会建出跑不了的库:\n  "
        + "\n  ".join(problems)
        + "\n\n每次新增 Alembic 迁移都必须同步这个种子文件，"
        "或改为由 alembic upgrade head 单独负责建表。"
    )

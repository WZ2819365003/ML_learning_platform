"""A0/A1 — engine config comes from Settings and carries pool keepalive.

The MySQL idle-connection 500 (wait_timeout closes the socket, pool hands out
the stale connection) is prevented by ``pool_pre_ping``; these tests pin the
kwargs so a refactor can't silently drop them. Full disconnect/recovery needs
a real MySQL and lives outside the unit suite.
"""
from app.config import get_settings
from app.models.database import DATABASE_URL, _async_engine_kwargs


def test_mysql_url_gets_pool_keepalive():
    kwargs = _async_engine_kwargs("mysql+aiomysql://user:pw@host:3306/db")
    assert kwargs["pool_pre_ping"] is True
    assert kwargs["pool_recycle"] == 1800
    assert kwargs["pool_size"] == 10
    assert kwargs["max_overflow"] == 20


def test_postgres_url_gets_pool_keepalive():
    kwargs = _async_engine_kwargs("postgresql+asyncpg://user:pw@host/db")
    assert kwargs["pool_pre_ping"] is True


def test_sqlite_url_skips_pool_params():
    kwargs = _async_engine_kwargs("sqlite+aiosqlite:///./storage/test.db")
    assert "pool_pre_ping" not in kwargs
    assert "pool_recycle" not in kwargs
    assert "pool_size" not in kwargs


def test_database_url_single_source_of_truth():
    # A0: the module-level URL must be exactly what Settings resolved —
    # no second os.getenv fallback with its own default.
    assert DATABASE_URL == get_settings().database_url

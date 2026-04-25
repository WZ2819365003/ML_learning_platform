from app.models.database import _to_sync_database_url


def test_to_sync_database_url_converts_async_drivers():
    assert (
        _to_sync_database_url("mysql+aiomysql://user:pass@localhost/db")
        == "mysql+pymysql://user:pass@localhost/db"
    )
    assert (
        _to_sync_database_url("postgresql+asyncpg://user:pass@localhost/db")
        == "postgresql+psycopg2://user:pass@localhost/db"
    )
    assert (
        _to_sync_database_url("sqlite+aiosqlite:///./storage/test.db")
        == "sqlite:///./storage/test.db"
    )

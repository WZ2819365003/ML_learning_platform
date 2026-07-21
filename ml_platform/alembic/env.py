"""Alembic environment backed by the application's ORM metadata."""

from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool
from sqlalchemy.engine import make_url

from app.config import get_settings
from app.models import database as database_models


config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

sync_database_url = database_models._to_sync_database_url(
    get_settings().database_url
)
# Alembic stores this value through ConfigParser, where a literal percent sign
# in a database password must be escaped.
config.set_main_option("sqlalchemy.url", sync_database_url.replace("%", "%%"))

target_metadata = database_models.Base.metadata


def run_migrations_offline() -> None:
    """Run migrations without creating a database connection."""
    dialect = make_url(sync_database_url).get_backend_name()
    context.configure(
        url=sync_database_url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
        render_as_batch=dialect == "sqlite",
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations with a synchronous SQLAlchemy connection."""
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=True,
            render_as_batch=connection.dialect.name == "sqlite",
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()

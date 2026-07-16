"""
Configuration management for the ML Training Platform.

Reads settings from .env file using python-dotenv and exposes them
through a Settings dataclass. Uses lru_cache to ensure a single
Settings instance is reused across the application.
"""

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

# Resolve the project root (ml_platform/) relative to this file's location
_PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Load .env from the project root
load_dotenv(_PROJECT_ROOT / ".env")


def _resolve_path(raw: str) -> Path:
    """Resolve a path that may be relative to the project root."""
    p = Path(raw)
    if not p.is_absolute():
        p = _PROJECT_ROOT / p
    return p.resolve()


@dataclass(frozen=True)
class Settings:
    """Application settings populated from environment variables."""

    # Database
    database_url: str = field(
        default_factory=lambda: os.getenv(
            "DATABASE_URL",
            "sqlite+aiosqlite:///./storage/ml_platform.db",
        )
    )

    # Redis
    redis_url: str = field(
        default_factory=lambda: os.getenv(
            "REDIS_URL",
            "redis://localhost:6379/0",
        )
    )

    # Celery
    celery_broker_url: str = field(
        default_factory=lambda: os.getenv(
            "CELERY_BROKER_URL",
            "redis://localhost:6379/0",
        )
    )
    celery_result_backend: str = field(
        default_factory=lambda: os.getenv(
            "CELERY_RESULT_BACKEND",
            "redis://localhost:6379/1",
        )
    )

    # Scheduler strategy — ``celery`` routes PlatformTask submissions through
    # the Celery worker pool (prod default once a Redis+worker pair is part
    # of every deployment); ``inprocess`` runs executors on the FastAPI
    # event loop (unit tests, single-process dev without Redis).
    # Leaving the default at ``inprocess`` keeps existing dev setups working
    # with zero config; ops can opt in via ``SCHEDULER_MODE=celery`` in .env.
    scheduler_mode: str = field(
        default_factory=lambda: os.getenv("SCHEDULER_MODE", "inprocess").lower()
    )

    # Event bus mode — ``memory`` keeps the in-process pub/sub (unit tests +
    # inprocess scheduler); ``redis`` uses Redis channels so Celery workers
    # running in separate processes can publish training progress events
    # that WebSocket clients (living in the FastAPI process) pick up.
    event_bus_mode: str = field(
        default_factory=lambda: os.getenv("EVENT_BUS_MODE", "memory").lower()
    )

    # Storage directories
    storage_uploads: Path = field(
        default_factory=lambda: _resolve_path(
            os.getenv("STORAGE_UPLOADS", "./storage/uploads")
        )
    )
    storage_models: Path = field(
        default_factory=lambda: _resolve_path(
            os.getenv("STORAGE_MODELS", "./storage/models")
        )
    )
    storage_logs: Path = field(
        default_factory=lambda: _resolve_path(
            os.getenv("STORAGE_LOGS", "./storage/logs")
        )
    )

    # MLflow
    mlflow_tracking_uri: str = field(
        default_factory=lambda: os.getenv(
            "MLFLOW_TRACKING_URI",
            "http://localhost:5001",
        )
    )

    # Upload limits (bytes) — default 200 MB
    max_upload_size: int = field(
        default_factory=lambda: int(
            os.getenv("MAX_UPLOAD_SIZE", "209715200")
        )
    )

    # Object storage (MinIO / S3-compatible)
    s3_endpoint_url: str = field(
        default_factory=lambda: os.getenv("S3_ENDPOINT_URL", "http://127.0.0.1:9000")
    )
    s3_access_key: str = field(
        default_factory=lambda: os.getenv("S3_ACCESS_KEY", "mlplatform")
    )
    s3_secret_key: str = field(
        default_factory=lambda: os.getenv("S3_SECRET_KEY", "mlplatform123")
    )
    s3_bucket: str = field(
        default_factory=lambda: os.getenv("S3_BUCKET", "ml-platform")
    )
    # Set to "false" to disable object storage (e.g. local-only dev)
    s3_enabled: bool = field(
        default_factory=lambda: os.getenv("S3_ENABLED", "false").lower() == "true"
    )

    # Convenience: project root
    project_root: Path = field(default_factory=lambda: _PROJECT_ROOT)

    def ensure_storage_dirs(self) -> None:
        """Create storage directories if they do not exist."""
        for directory in (
            self.storage_uploads,
            self.storage_models,
            self.storage_logs,
        ):
            directory.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the cached application settings singleton."""
    settings = Settings()
    settings.ensure_storage_dirs()
    return settings

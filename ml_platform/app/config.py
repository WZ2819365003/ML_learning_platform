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

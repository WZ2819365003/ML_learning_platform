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


# Development-only fallbacks. They are recognisable so production validation
# can refuse to start when one of them leaks into a prod deployment.
_DEV_DATABASE_URL = "sqlite+aiosqlite:///./storage/ml_platform.db"
_DEV_S3_ACCESS_KEY = "mlplatform"
_DEV_S3_SECRET_KEY = "mlplatform123"


@dataclass(frozen=True)
class Settings:
    """Application settings populated from environment variables."""

    # Deployment profile: development | test | production.
    # Production enables strict startup validation (validate_for_production).
    environment: str = field(
        default_factory=lambda: os.getenv("ENVIRONMENT", "development").lower()
    )

    # Database
    database_url: str = field(
        default_factory=lambda: os.getenv("DATABASE_URL", _DEV_DATABASE_URL)
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
    # Comma-separated task kinds enabled for Celery. Empty means all kinds
    # remain in-process; SCHEDULER_MODE=celery stays the global override.
    celery_kinds: str = field(
        default_factory=lambda: os.getenv("CELERY_KINDS", "")
    )
    # Background recovery sweep (see scheduler.recover_stalled_tasks). The
    # stall threshold MUST stay above the training hard time limit: without a
    # heartbeat there is no way to tell a dead worker from a slow trial, so a
    # threshold set too low duplicates live work.
    recovery_sweep_interval_seconds: int = field(
        default_factory=lambda: int(os.getenv("RECOVERY_SWEEP_INTERVAL_SECONDS", "300"))
    )
    stalled_task_timeout_seconds: int = field(
        default_factory=lambda: int(os.getenv("STALLED_TASK_TIMEOUT_SECONDS", str(6 * 3600)))
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
        default_factory=lambda: os.getenv("S3_ACCESS_KEY", _DEV_S3_ACCESS_KEY)
    )
    s3_secret_key: str = field(
        default_factory=lambda: os.getenv("S3_SECRET_KEY", _DEV_S3_SECRET_KEY)
    )
    s3_bucket: str = field(
        default_factory=lambda: os.getenv("S3_BUCKET", "ml-platform")
    )
    # Set to "false" to disable object storage (e.g. local-only dev)
    s3_enabled: bool = field(
        default_factory=lambda: os.getenv("S3_ENABLED", "false").lower() == "true"
    )

    # Authentication (single-admin). Env-configurable with an environment-aware
    # default: production defaults ON, everything else defaults OFF — local
    # dev and the test suite stay unauthenticated with zero config. An
    # explicit AUTH_ENABLED always wins.
    auth_enabled: bool = field(
        default_factory=lambda: (
            os.getenv("AUTH_ENABLED").lower() == "true"
            if os.getenv("AUTH_ENABLED") is not None
            else os.getenv("ENVIRONMENT", "development").lower() == "production"
        )
    )
    auth_username: str = field(
        default_factory=lambda: os.getenv("AUTH_USERNAME", "admin")
    )
    auth_password: str = field(
        default_factory=lambda: os.getenv("AUTH_PASSWORD", "")
    )
    auth_secret_key: str = field(
        default_factory=lambda: os.getenv("AUTH_SECRET_KEY", "dev-secret-not-for-production")
    )

    # User-code executor sandboxes (A3) — wall-clock limits for the
    # 代码配置 / 数据 Pipeline subprocesses. Pipeline gets a longer budget
    # because it crunches real DataFrames.
    user_code_timeout_s: float = field(
        default_factory=lambda: float(os.getenv("USER_CODE_TIMEOUT_S", "5"))
    )
    pipeline_code_timeout_s: float = field(
        default_factory=lambda: float(os.getenv("PIPELINE_CODE_TIMEOUT_S", "60"))
    )

    # Convenience: project root
    project_root: Path = field(default_factory=lambda: _PROJECT_ROOT)

    @property
    def is_production(self) -> bool:
        return self.environment == "production"

    def validate_for_production(self) -> None:
        """Refuse to start production with dev defaults (blueprint A0).

        Raises RuntimeError listing every violation so ops can fix them in
        one pass instead of whack-a-mole restarts. No-op outside production.
        """
        if not self.is_production:
            return
        problems: list[str] = []
        if not os.getenv("DATABASE_URL"):
            problems.append("DATABASE_URL 未显式配置（生产不允许回退到开发 SQLite 默认值）")
        elif self.database_url == _DEV_DATABASE_URL:
            problems.append("DATABASE_URL 仍是开发默认值")
        elif "root:123456" in self.database_url:
            problems.append("DATABASE_URL 使用开发默认凭据 root:123456")
        if self.s3_enabled and (
            self.s3_access_key == _DEV_S3_ACCESS_KEY
            or self.s3_secret_key == _DEV_S3_SECRET_KEY
        ):
            problems.append("S3_ENABLED=true 但 S3_ACCESS_KEY/S3_SECRET_KEY 仍是开发默认密钥")
        # Auth defaults ON in production; explicitly disabling it is allowed
        # (env-configurable by design) but the enabled path must be complete.
        if self.auth_enabled:
            if not self.auth_password:
                problems.append("AUTH_PASSWORD 未配置")
            if len(self.auth_secret_key) < 32 or self.auth_secret_key == "dev-secret-not-for-production":
                problems.append("AUTH_SECRET_KEY 缺失或过短（需 ≥32 位随机串）")
        if problems:
            raise RuntimeError(
                "生产环境配置校验失败（ENVIRONMENT=production）:\n  - "
                + "\n  - ".join(problems)
            )

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

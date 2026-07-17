"""A0 — production startup validation refuses dev defaults."""
import pytest

from app.config import Settings


def _settings(monkeypatch, env: dict[str, str]) -> Settings:
    # Settings reads os.environ in default_factory, so patch then instantiate.
    for key in (
        "ENVIRONMENT", "DATABASE_URL", "S3_ENABLED",
        "S3_ACCESS_KEY", "S3_SECRET_KEY",
    ):
        monkeypatch.delenv(key, raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    return Settings()


def test_development_never_raises(monkeypatch):
    _settings(monkeypatch, {"ENVIRONMENT": "development"}).validate_for_production()


def test_production_missing_database_url_fails(monkeypatch):
    s = _settings(monkeypatch, {"ENVIRONMENT": "production"})
    with pytest.raises(RuntimeError, match="DATABASE_URL"):
        s.validate_for_production()


def test_production_dev_mysql_credentials_fail(monkeypatch):
    s = _settings(monkeypatch, {
        "ENVIRONMENT": "production",
        "DATABASE_URL": "mysql+aiomysql://root:123456@db:3306/ml_platform",
    })
    with pytest.raises(RuntimeError, match="root:123456"):
        s.validate_for_production()


def test_production_dev_s3_secret_fails(monkeypatch):
    s = _settings(monkeypatch, {
        "ENVIRONMENT": "production",
        "DATABASE_URL": "mysql+aiomysql://svc:strongpw@db:3306/ml_platform",
        "S3_ENABLED": "true",
        "S3_ACCESS_KEY": "real-key",
        "S3_SECRET_KEY": "mlplatform123",
    })
    with pytest.raises(RuntimeError, match="S3"):
        s.validate_for_production()


def test_production_with_real_config_passes(monkeypatch):
    s = _settings(monkeypatch, {
        "ENVIRONMENT": "production",
        "DATABASE_URL": "mysql+aiomysql://svc:strongpw@db:3306/ml_platform",
        "S3_ENABLED": "true",
        "S3_ACCESS_KEY": "real-key",
        "S3_SECRET_KEY": "real-secret",
    })
    s.validate_for_production()  # no raise


def test_s3_disabled_skips_s3_checks(monkeypatch):
    s = _settings(monkeypatch, {
        "ENVIRONMENT": "production",
        "DATABASE_URL": "mysql+aiomysql://svc:strongpw@db:3306/ml_platform",
        "S3_ENABLED": "false",
    })
    s.validate_for_production()  # dev S3 keys irrelevant when disabled

"""Auth module — token round trip, login flow, HTTP guard, WS handshake.

Auth is env-configurable with an environment-aware default: production → ON,
everything else → OFF. Tests flip it via env vars + get_settings.cache_clear()
so every module (middleware, routes, core) sees the same settings.
"""
import json
import time

import pytest
from httpx import ASGITransport, AsyncClient

import app.core.auth as auth_core
from app.config import Settings, get_settings
from app.main import app


@pytest.fixture
def auth_on(monkeypatch):
    """Enable auth with a known credential for one test (cache-cleared)."""
    monkeypatch.setenv("AUTH_ENABLED", "true")
    monkeypatch.setenv("AUTH_USERNAME", "userroot")
    monkeypatch.setenv("AUTH_PASSWORD", "test-password-123")
    monkeypatch.setenv("AUTH_SECRET_KEY", "x" * 48)
    get_settings.cache_clear()
    yield get_settings()
    get_settings.cache_clear()


@pytest.fixture
async def client():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


# /api/training/models reads the YAML registry only — no DB, ideal guard probe.
GUARDED_URL = "/api/training/models"


def test_token_round_trip(auth_on):
    token, expires_at = auth_core.issue_token("userroot")
    assert auth_core.verify_token(token) == "userroot"
    assert expires_at > time.time()


def test_configured_auth_users_accepts_multi_account_json(monkeypatch):
    _clean_env(monkeypatch)
    monkeypatch.setenv("AUTH_ENABLED", "true")
    monkeypatch.setenv("AUTH_USERS_JSON", json.dumps({
        "alice": "alice-password",
        "bob": "bob-password",
    }))
    monkeypatch.setenv("AUTH_SECRET_KEY", "x" * 48)
    get_settings.cache_clear()
    try:
        assert auth_core.configured_auth_users() == {
            "alice": "alice-password",
            "bob": "bob-password",
        }
        assert auth_core.authenticate_user("alice", "alice-password") is True
        assert auth_core.authenticate_user("alice", "bob-password") is False
    finally:
        get_settings.cache_clear()


def test_token_tamper_and_expiry(auth_on):
    token, _ = auth_core.issue_token("userroot")
    assert auth_core.verify_token(token[:-2] + "zz") is None
    expired, _ = auth_core.issue_token("userroot", ttl_seconds=-5)
    assert auth_core.verify_token(expired) is None
    assert auth_core.verify_token(None) is None
    assert auth_core.verify_token("garbage") is None


async def test_login_success_and_failure(auth_on, client):
    ok = await client.post("/api/auth/login", json={"username": "userroot", "password": "test-password-123"})
    assert ok.status_code == 200
    body = ok.json()
    assert body["username"] == "userroot" and body["token"]

    bad = await client.post("/api/auth/login", json={"username": "userroot", "password": "wrong"})
    assert bad.status_code == 401
    bad_user = await client.post("/api/auth/login", json={"username": "nobody", "password": "test-password-123"})
    assert bad_user.status_code == 401


async def test_login_accepts_auth_users_json(monkeypatch, client):
    _clean_env(monkeypatch)
    monkeypatch.setenv("AUTH_ENABLED", "true")
    monkeypatch.setenv("AUTH_USERS_JSON", json.dumps({
        "alice": "alice-password",
        "bob": "bob-password",
    }))
    monkeypatch.setenv("AUTH_SECRET_KEY", "x" * 48)
    get_settings.cache_clear()
    try:
        ok = await client.post("/api/auth/login", json={"username": "bob", "password": "bob-password"})
        assert ok.status_code == 200
        token = ok.json()["token"]
        me = await client.get("/api/auth/me", headers={"Authorization": f"Bearer {token}"})
        assert me.json()["username"] == "bob"

        bad = await client.post("/api/auth/login", json={"username": "alice", "password": "bob-password"})
        assert bad.status_code == 401
    finally:
        get_settings.cache_clear()


async def test_api_guard_blocks_and_admits(auth_on, client):
    denied = await client.get(GUARDED_URL)
    assert denied.status_code == 401

    login = await client.post("/api/auth/login", json={"username": "userroot", "password": "test-password-123"})
    token = login.json()["token"]
    admitted = await client.get(GUARDED_URL, headers={"Authorization": f"Bearer {token}"})
    assert admitted.status_code == 200

    me = await client.get("/api/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert me.json()["username"] == "userroot"


async def test_health_stays_public(auth_on, client):
    r = await client.get("/health")
    assert r.status_code == 200
    assert r.json()["environment"] == auth_on.environment


async def test_auth_disabled_passthrough(client):
    # Default (non-production) settings: auth off — unauthenticated works.
    r = await client.get(GUARDED_URL)
    assert r.status_code == 200


def test_ws_gate_uses_query_token(auth_on):
    token, _ = auth_core.issue_token("userroot")
    assert auth_core.request_is_authorized(None, token) is True
    assert auth_core.request_is_authorized(None, "bad") is False
    assert auth_core.request_is_authorized(f"Bearer {token}", None) is True


def _clean_env(monkeypatch):
    for key in ("ENVIRONMENT", "DATABASE_URL", "AUTH_ENABLED", "AUTH_USERNAME",
                "AUTH_PASSWORD", "AUTH_USERS_JSON", "AUTH_SECRET_KEY", "S3_ENABLED"):
        monkeypatch.delenv(key, raising=False)


def test_auth_defaults_on_in_production_off_elsewhere(monkeypatch):
    _clean_env(monkeypatch)
    assert Settings().auth_enabled is False  # development default

    monkeypatch.setenv("ENVIRONMENT", "production")
    assert Settings().auth_enabled is True   # production default

    monkeypatch.setenv("AUTH_ENABLED", "false")
    assert Settings().auth_enabled is False  # explicit override wins


def test_production_validates_enabled_auth_config(monkeypatch):
    _clean_env(monkeypatch)
    monkeypatch.setenv("ENVIRONMENT", "production")
    monkeypatch.setenv("DATABASE_URL", "mysql+aiomysql://svc:strongpw@db:3306/ml_platform")
    # Auth on by default but password/secret missing → boot refused.
    with pytest.raises(RuntimeError, match="AUTH_PASSWORD"):
        Settings().validate_for_production()

    monkeypatch.setenv("AUTH_USERNAME", "userroot")
    monkeypatch.setenv("AUTH_PASSWORD", "pw")
    monkeypatch.setenv("AUTH_SECRET_KEY", "s" * 40)
    Settings().validate_for_production()  # no raise

    monkeypatch.delenv("AUTH_PASSWORD")
    monkeypatch.setenv("AUTH_USERS_JSON", json.dumps({"alice": "pw1"}))
    Settings().validate_for_production()  # no raise

    monkeypatch.setenv("AUTH_USERS_JSON", "not-json")
    with pytest.raises(RuntimeError, match="AUTH_USERS_JSON"):
        Settings().validate_for_production()

    # Explicitly disabling auth in production is allowed (warned, not fatal).
    monkeypatch.setenv("AUTH_ENABLED", "false")
    monkeypatch.delenv("AUTH_PASSWORD", raising=False)
    Settings().validate_for_production()  # no raise

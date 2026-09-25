"""Authentication helpers for the platform cloud deployment gate.

Deliberately dependency-free: PBKDF2 (stdlib hashlib) for the password and an
HMAC-SHA256 signed token (stdlib hmac) instead of a JWT library. The platform
supports the legacy single-admin environment pair:

    AUTH_ENABLED=true
    AUTH_USERNAME=...
    AUTH_PASSWORD=...          # plaintext lives ONLY in the server-side .env
    AUTH_SECRET_KEY=...        # >=32 chars, signs tokens

Cloud deployments can define multiple accounts with:

    AUTH_USERS_JSON='{"alice":"password1","bob":"password2"}'

With AUTH_ENABLED=false (the dev/test default) every check passes through, so
local workflows and the 328-test suite run unauthenticated exactly as before.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import secrets
import time
from typing import Any

from fastapi import Header, HTTPException

from app.config import Settings
from app.config import get_settings

TOKEN_TTL_SECONDS = 24 * 3600


def _b64(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode().rstrip("=")


def _unb64(data: str) -> bytes:
    return base64.urlsafe_b64decode(data + "=" * (-len(data) % 4))


def verify_password(candidate: str, expected: str) -> bool:
    """Constant-time comparison via PBKDF2 with a fixed app-level salt derived
    from the secret key — enough for a single env-configured credential."""
    settings = get_settings()
    salt = hashlib.sha256(settings.auth_secret_key.encode()).digest()

    def _kdf(pw: str) -> bytes:
        return hashlib.pbkdf2_hmac("sha256", pw.encode(), salt, 60_000)

    return hmac.compare_digest(_kdf(candidate), _kdf(expected))


def configured_auth_users(settings: Settings | None = None) -> dict[str, str]:
    """Return username -> plaintext password from env-backed settings.

    Passwords intentionally stay server-side only.  The legacy
    AUTH_USERNAME/AUTH_PASSWORD pair is used when AUTH_USERS_JSON is absent.
    """
    settings = settings or get_settings()
    raw = (settings.auth_users_json or "").strip()
    if raw:
        try:
            parsed: Any = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise RuntimeError("AUTH_USERS_JSON 必须是 JSON 对象") from exc
        if not isinstance(parsed, dict):
            raise RuntimeError("AUTH_USERS_JSON 必须是 username -> password 的 JSON 对象")
        users = {
            str(username).strip(): str(password)
            for username, password in parsed.items()
            if str(username).strip() and str(password)
        }
        if not users:
            raise RuntimeError("AUTH_USERS_JSON 至少需要一个非空账号")
        return users

    if settings.auth_username and settings.auth_password:
        return {settings.auth_username: settings.auth_password}
    return {}


def authenticate_user(username: str, password: str) -> bool:
    """Return True when the supplied credential matches a configured account."""
    expected = configured_auth_users().get(username)
    return bool(expected) and verify_password(password, expected)


def issue_token(username: str, ttl_seconds: int = TOKEN_TTL_SECONDS) -> tuple[str, int]:
    """Return (token, expires_at_epoch). Token = b64(payload).b64(hmac)."""
    settings = get_settings()
    expires_at = int(time.time()) + ttl_seconds
    payload = _b64(json.dumps(
        {"u": username, "exp": expires_at, "n": secrets.token_hex(4)},
        separators=(",", ":"),
    ).encode())
    sig = _b64(hmac.new(settings.auth_secret_key.encode(), payload.encode(), hashlib.sha256).digest())
    return f"{payload}.{sig}", expires_at


def verify_token(token: str | None) -> str | None:
    """Return the username for a valid unexpired token, else None."""
    if not token or "." not in token:
        return None
    settings = get_settings()
    payload, _, sig = token.rpartition(".")
    expected = _b64(hmac.new(settings.auth_secret_key.encode(), payload.encode(), hashlib.sha256).digest())
    if not hmac.compare_digest(sig, expected):
        return None
    try:
        data = json.loads(_unb64(payload))
    except (ValueError, UnicodeDecodeError):
        return None
    if int(data.get("exp") or 0) < time.time():
        return None
    return data.get("u") or None


def username_from_authorization(
    authorization_header: str | None,
    query_token: str | None = None,
) -> str | None:
    """Resolve a username from Authorization or query token.

    When auth is disabled, callers get the configured local username so newly
    created development rows can still carry an owner without filtering lists.
    """
    settings = get_settings()
    if not settings.auth_enabled:
        return settings.auth_username or "dev"
    token = query_token
    if authorization_header and authorization_header.lower().startswith("bearer "):
        token = authorization_header[7:].strip()
    return verify_token(token)


def current_username_from_authorization(
    authorization: str | None = Header(default=None),
) -> str:
    """FastAPI dependency returning the current username or raising 401."""
    username = username_from_authorization(authorization)
    if username is None:
        raise HTTPException(status_code=401, detail="未登录或登录已过期")
    return username


def owner_scope_username(username: str | None) -> str | None:
    """Return an owner filter only when auth is enabled.

    Dev/test mode keeps historical global visibility; production auth scopes
    rows to the logged-in account.
    """
    return username if get_settings().auth_enabled else None


def request_is_authorized(authorization_header: str | None, query_token: str | None = None) -> bool:
    """Shared gate for HTTP middleware and WebSocket handshakes.

    Always True when auth is disabled. Accepts `Authorization: Bearer <token>`
    or (for WebSocket, where headers are awkward from browsers) `?token=`.
    """
    settings = get_settings()
    if not settings.auth_enabled:
        return True
    return username_from_authorization(authorization_header, query_token) is not None

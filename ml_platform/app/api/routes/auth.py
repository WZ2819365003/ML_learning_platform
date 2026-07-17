"""Authentication routes — POST /api/auth/login, GET /api/auth/me."""
from __future__ import annotations

from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel

from app.config import get_settings
from app.core.auth import issue_token, verify_password, verify_token

router = APIRouter(prefix="/auth", tags=["Auth"])


class LoginRequest(BaseModel):
    username: str
    password: str


@router.post("/login", summary="Exchange credentials for a bearer token")
async def login(body: LoginRequest) -> dict:
    settings = get_settings()
    if not settings.auth_enabled:
        # Keep the frontend flow working in dev: hand out a token nobody checks.
        token, expires_at = issue_token(body.username or "dev")
        return {"token": token, "username": body.username or "dev", "expires_at": expires_at}
    ok = (
        body.username == settings.auth_username
        and bool(settings.auth_password)
        and verify_password(body.password, settings.auth_password)
    )
    if not ok:
        raise HTTPException(status_code=401, detail="用户名或密码错误")
    token, expires_at = issue_token(body.username)
    return {"token": token, "username": body.username, "expires_at": expires_at}


@router.get("/me", summary="Who am I (validates the bearer token)")
async def me(authorization: str | None = Header(default=None)) -> dict:
    settings = get_settings()
    if not settings.auth_enabled:
        return {"username": "dev", "auth_enabled": False}
    token = authorization[7:].strip() if authorization and authorization.lower().startswith("bearer ") else None
    username = verify_token(token)
    if username is None:
        raise HTTPException(status_code=401, detail="未登录或登录已过期")
    return {"username": username, "auth_enabled": True}

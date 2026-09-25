"""WebSocket endpoints for real-time training metrics and log streaming."""

import asyncio

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from fastapi import HTTPException

from app.core.auth import (
    owner_scope_username,
    request_is_authorized,
    username_from_authorization,
)
from app.core.logger import event_bus
from app.core.ownership import ensure_task_owner
from app.models.database import async_session_factory

router = APIRouter()


async def ws_authorized(websocket: WebSocket) -> bool:
    """Handshake auth for WebSocket routes (HTTP middleware can't see them).

    Browsers can't set Authorization headers on WebSocket, so the token rides
    a ``?token=`` query param. Rejects with 4401 before accepting.
    """
    if request_is_authorized(
        websocket.headers.get("authorization"),
        websocket.query_params.get("token"),
    ):
        return True
    await websocket.close(code=4401, reason="unauthorized")
    return False


async def ws_task_authorized(websocket: WebSocket, task_id: str) -> bool:
    """Authorize a WebSocket token and verify the requested task ownership."""
    username = username_from_authorization(
        websocket.headers.get("authorization"),
        websocket.query_params.get("token"),
    )
    if username is None:
        await websocket.close(code=4401, reason="unauthorized")
        return False
    async with async_session_factory() as db:
        try:
            await ensure_task_owner(db, task_id, owner_scope_username(username))
        except HTTPException:
            await websocket.close(code=4404, reason="task_not_found")
            return False
    return True


@router.websocket("/ws/training/{task_id}")
async def training_ws(websocket: WebSocket, task_id: str):
    """Stream real-time training metrics for a specific task."""
    if not await ws_task_authorized(websocket, task_id):
        return
    await websocket.accept()

    queue = event_bus.subscribe(f"training:{task_id}")
    try:
        while True:
            message = await queue.get()
            await websocket.send_json(message)
    except WebSocketDisconnect:
        pass
    except Exception:
        pass
    finally:
        event_bus.unsubscribe(f"training:{task_id}", queue)


@router.websocket("/ws/logs/{task_id}")
async def logs_ws(websocket: WebSocket, task_id: str):
    """Stream real-time training logs for a specific task."""
    if not await ws_task_authorized(websocket, task_id):
        return
    await websocket.accept()

    queue = event_bus.subscribe(f"logs:{task_id}")
    try:
        while True:
            message = await queue.get()
            await websocket.send_json(message)
    except WebSocketDisconnect:
        pass
    except Exception:
        pass
    finally:
        event_bus.unsubscribe(f"logs:{task_id}", queue)

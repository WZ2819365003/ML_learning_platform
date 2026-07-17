"""WebSocket endpoints for real-time training metrics and log streaming."""

import asyncio

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.core.auth import request_is_authorized
from app.core.logger import event_bus

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


@router.websocket("/ws/training/{task_id}")
async def training_ws(websocket: WebSocket, task_id: str):
    """Stream real-time training metrics for a specific task."""
    if not await ws_authorized(websocket):
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
    if not await ws_authorized(websocket):
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

"""WebSocket endpoints for real-time training metrics and log streaming."""

import asyncio

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.core.logger import event_bus

router = APIRouter()


@router.websocket("/ws/training/{task_id}")
async def training_ws(websocket: WebSocket, task_id: str):
    """Stream real-time training metrics for a specific task."""
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

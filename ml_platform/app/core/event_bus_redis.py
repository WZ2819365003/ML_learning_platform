"""Redis-backed event bus with a local asyncio fan-out bridge."""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import threading
from collections import defaultdict
from typing import Any

import redis


logger = logging.getLogger(__name__)


class RedisEventBus:
    """Publish synchronously to Redis and bridge pubsub into local queues.

    ``publish`` deliberately remains a normal synchronous method: sklearn
    training invokes it directly from ThreadPoolExecutor workers.  Only the
    subscriber bridge is asynchronous and belongs to the FastAPI lifespan.
    """

    def __init__(
        self,
        redis_url: str | None = None,
        *,
        redis_client: Any | None = None,
        queue_maxsize: int = 256,
        reconnect_delay: float = 1.0,
        namespace: str = "ml-platform",
    ) -> None:
        if queue_maxsize < 1:
            raise ValueError("queue_maxsize must be at least 1")
        if redis_client is None and not redis_url:
            raise ValueError("redis_url is required when redis_client is not supplied")

        self._owns_client = redis_client is None
        self._redis = redis_client or redis.Redis.from_url(redis_url)
        self._queue_maxsize = queue_maxsize
        self._reconnect_delay = reconnect_delay
        self._namespace = namespace.rstrip(":")
        self._subscribers: dict[str, list[asyncio.Queue]] = defaultdict(list)
        self._publish_lock = threading.Lock()
        self._stop_event = asyncio.Event()
        self._bridge_task: asyncio.Task | None = None
        self._active_pubsub: Any | None = None

    def subscribe(self, channel: str) -> asyncio.Queue:
        local_queue: asyncio.Queue = asyncio.Queue(maxsize=self._queue_maxsize)
        self._subscribers[channel].append(local_queue)
        return local_queue

    def unsubscribe(self, channel: str, queue: asyncio.Queue) -> None:
        if channel not in self._subscribers:
            return
        self._subscribers[channel] = [
            candidate
            for candidate in self._subscribers[channel]
            if candidate is not queue
        ]
        if not self._subscribers[channel]:
            del self._subscribers[channel]

    def publish(self, channel: str, message: dict) -> None:
        """Synchronously publish JSON; safe to call from worker threads."""
        payload = json.dumps(
            {"channel": channel, "message": message},
            ensure_ascii=False,
            default=str,
        )
        physical_channel = f"{self._namespace}:{channel}"
        with self._publish_lock:
            self._redis.publish(physical_channel, payload)

    async def start(self) -> None:
        """Start the Redis pubsub-to-local-queue bridge once."""
        if self._bridge_task is not None and not self._bridge_task.done():
            return
        self._stop_event.clear()
        self._bridge_task = asyncio.create_task(
            self._bridge_loop(), name="redis-event-bus-bridge"
        )
        await asyncio.sleep(0)

    async def stop(self) -> None:
        """Stop the bridge and close its active pubsub connection."""
        self._stop_event.set()
        task = self._bridge_task
        self._bridge_task = None
        if task is not None:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
        if self._owns_client:
            with contextlib.suppress(Exception):
                await asyncio.to_thread(self._redis.close)

    async def _bridge_loop(self) -> None:
        while not self._stop_event.is_set():
            pubsub = None
            try:
                pubsub = self._redis.pubsub(ignore_subscribe_messages=True)
                self._active_pubsub = pubsub
                await asyncio.to_thread(
                    pubsub.psubscribe, f"{self._namespace}:*"
                )
                while not self._stop_event.is_set():
                    event = await asyncio.to_thread(
                        pubsub.get_message,
                        ignore_subscribe_messages=True,
                        timeout=1.0,
                    )
                    if event is None or event.get("type") not in {"message", "pmessage"}:
                        continue
                    self._handle_event(event.get("data"))
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # Redis disconnects are retried indefinitely
                logger.warning("Redis event bus disconnected; retrying: %s", exc)
                try:
                    await asyncio.wait_for(
                        self._stop_event.wait(), timeout=self._reconnect_delay
                    )
                except asyncio.TimeoutError:
                    pass
            finally:
                self._active_pubsub = None
                if pubsub is not None:
                    with contextlib.suppress(Exception):
                        await asyncio.to_thread(pubsub.close)

    def _handle_event(self, raw_data: Any) -> None:
        try:
            payload = json.loads(raw_data)
            channel = payload["channel"]
            message = payload["message"]
        except (TypeError, ValueError, KeyError) as exc:
            logger.warning("Dropping malformed Redis event: %s", exc)
            return

        for local_queue in list(self._subscribers.get(channel, [])):
            if local_queue.full():
                with contextlib.suppress(asyncio.QueueEmpty):
                    local_queue.get_nowait()
            with contextlib.suppress(asyncio.QueueFull):
                local_queue.put_nowait(message)

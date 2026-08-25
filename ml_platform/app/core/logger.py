"""Training logger — per-task file + metrics logging with event bus."""

import asyncio
import json
import logging
import threading
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.config import get_settings

logger = logging.getLogger(__name__)


def _parse_extra_fields(raw_extra: str) -> dict[str, str] | None:
    extra: dict[str, str] = {}
    for item in raw_extra.split(" | "):
        if "=" not in item:
            continue
        key, value = item.split("=", 1)
        extra[key] = value
    return extra or None


class EventBus:
    """Simple in-memory pub/sub for bridging training workers to WebSocket clients.

    Works without Redis — suitable for single-process dev mode.

    ``publish`` is called from ThreadPoolExecutor workers (sklearn training runs
    there), while subscribers are coroutines parked on ``await queue.get()``.
    ``asyncio.Queue`` is not thread-safe: a bare ``put_nowait`` from a worker
    thread wakes the consumer through ``loop.call_soon``, which neither is
    thread-safe nor interrupts a sleeping event loop — so entries sat in the
    queue until some unrelated request happened to wake it. Measured worst-case
    delivery on an otherwise-idle loop was ~2.9s. Each queue therefore remembers
    the loop it was created on, and off-loop publishes hop back via
    ``call_soon_threadsafe``.
    """

    def __init__(self):
        # channel -> list of (queue, owning event loop)
        self._subscribers: dict[
            str, list[tuple[asyncio.Queue, asyncio.AbstractEventLoop]]
        ] = defaultdict(list)

    def subscribe(self, channel: str) -> asyncio.Queue:
        queue = asyncio.Queue()
        # subscribe() is only ever called from a coroutine (the WebSocket
        # handler), so the running loop here is the one that will consume.
        self._subscribers[channel].append((queue, asyncio.get_event_loop()))
        return queue

    def unsubscribe(self, channel: str, queue: asyncio.Queue):
        if channel in self._subscribers:
            self._subscribers[channel] = [
                entry for entry in self._subscribers[channel] if entry[0] is not queue
            ]
            if not self._subscribers[channel]:
                del self._subscribers[channel]

    def publish(self, channel: str, message: dict):
        for queue, loop in list(self._subscribers.get(channel, [])):
            try:
                running = asyncio.get_running_loop()
            except RuntimeError:
                running = None
            if running is loop:
                # Already on the consumer's loop — a direct put is correct.
                try:
                    queue.put_nowait(message)
                except asyncio.QueueFull:
                    pass  # drop if consumer is too slow
                continue
            try:
                loop.call_soon_threadsafe(_put_nowait_dropping_full, queue, message)
            except RuntimeError:
                pass  # loop already closed; the subscriber is gone


def _put_nowait_dropping_full(queue: asyncio.Queue, message: dict) -> None:
    """Queue put that mirrors publish()'s drop-on-full policy. Runs on the loop."""
    try:
        queue.put_nowait(message)
    except asyncio.QueueFull:
        pass  # drop if consumer is too slow


def _build_event_bus():
    settings = get_settings()
    if settings.event_bus_mode == "redis":
        from app.core.event_bus_redis import RedisEventBus

        return RedisEventBus(redis_url=settings.redis_url)
    if settings.event_bus_mode != "memory":
        logger.warning(
            "Unknown EVENT_BUS_MODE=%r; falling back to memory",
            settings.event_bus_mode,
        )
    return EventBus()


# Global event bus singleton. The default branch is the original in-memory
# implementation, preserving synchronous publish semantics and call paths.
event_bus = _build_event_bus()


class TrainingLogger:
    """Per-task logger that writes to files, publishes to event bus, and
    buffers entries for periodic persistence to the `training_logs` table.

    Persistence is batched rather than per-line: sklearn trials emit ~10-50
    log lines, and one DB round-trip each would be wasteful. It used to be
    deferred entirely to a single `flush_to_db()` at the end of the Run, which
    meant `training_logs` stayed empty for the whole run — so opening the log
    panel mid-training showed nothing at all, and a crash left the rows only in
    the on-disk .log file. Now a flush also happens once the buffer reaches
    ``_FLUSH_EVERY_N_ENTRIES`` or ``_FLUSH_INTERVAL_SECONDS`` have passed,
    whichever comes first, and `_run_training_sync` still flushes at the end to
    drain the tail.

    ``log()`` runs on a ThreadPoolExecutor worker while `flush_to_db()` may also
    be called from the owning coroutine's thread at the end of a run, so buffer
    handoff is guarded by a lock.

    Set ``persist_to_db=False`` for task families that are not rows in
    ``training_tasks``. ``training_logs.task_id`` is a FK onto that table, so a
    DL task id (which lives in ``dl_training_tasks``) cannot be inserted there —
    every flush would fail the constraint, get pushed back, and be retried
    forever. DL already persists each line through ``_store_dl_log_record`` on
    its own path, so its buffer was pure waste even before batching existed.
    """

    # Small enough that a run's logs show up while it is still running, large
    # enough that a chatty trial does not turn into one INSERT per line.
    _FLUSH_EVERY_N_ENTRIES = 25
    _FLUSH_INTERVAL_SECONDS = 2.0
    # Bound the retry buffer: a persistently failing flush must not grow without
    # limit and turn an observability feature into an OOM.
    _MAX_BUFFERED_ENTRIES = 1000

    def __init__(self, task_id: str, model_type: str = "", *, persist_to_db: bool = True):
        self.task_id = task_id
        self.model_type = model_type
        settings = get_settings()
        self.log_dir = settings.storage_logs
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.log_file = self.log_dir / f"{task_id}.log"
        self.metrics_file = self.log_dir / f"{task_id}_metrics.json"

        # In-memory buffer of (level, message, extra, created_at) tuples
        # awaiting a DB flush, plus the bookkeeping that decides when to flush.
        self._db_buffer: list[dict[str, Any]] = []
        self._buffer_lock = threading.Lock()
        self._last_flush_at = time.monotonic()
        self._persist_to_db = persist_to_db

        # Initialize metrics JSON
        self._metrics_data: dict[str, Any] = {
            "task_id": task_id,
            "model_type": model_type,
            "steps": [],
        }
        self._save_metrics()

    def log(self, level: str, message: str, **extra):
        """Write a log entry to file, buffer for DB, and publish to bus."""
        timestamp_dt = datetime.now(timezone.utc)
        timestamp = timestamp_dt.isoformat()

        # Format log line
        extra_str = (
            " | ".join(f"{k}={v}" for k, v in extra.items()) if extra else ""
        )
        line = f"{timestamp} | {level:5s} | {message}"
        if extra_str:
            line += f" | {extra_str}"

        # Append to file
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(line + "\n")

        # Buffer for DB flush (skipped when this task family has no row in
        # training_tasks — see the class docstring).
        if self._persist_to_db:
            with self._buffer_lock:
                self._db_buffer.append(
                    {
                        "level": level,
                        "message": message,
                        "extra": dict(extra) if extra else None,
                        "created_at": timestamp_dt,
                    }
                )

        # Publish to event bus
        event_bus.publish(
            f"logs:{self.task_id}",
            {
                "type": "log",
                "task_id": self.task_id,
                "level": level,
                "message": message,
                "extra": extra if extra else None,
                "timestamp": timestamp,
            },
        )

        self._maybe_flush()

    def _maybe_flush(self) -> None:
        """Flush if the buffer is big enough or old enough. Never raises."""
        if not self._persist_to_db:
            return
        with self._buffer_lock:
            pending = len(self._db_buffer)
            if pending == 0:
                return
            due = (
                pending >= self._FLUSH_EVERY_N_ENTRIES
                or (time.monotonic() - self._last_flush_at) >= self._FLUSH_INTERVAL_SECONDS
            )
        if due:
            # Outside the lock: flush_to_db takes it itself for the handoff,
            # and the INSERT must not block other threads appending log lines.
            self.flush_to_db()

    def flush_to_db(self) -> int:
        """Persist all buffered log entries to `training_logs`.

        Idempotent (drains the buffer), safe to call multiple times. Returns
        the number of rows inserted. Designed to be called from sync code
        (e.g. a ThreadPoolExecutor worker) — uses the sync SQLAlchemy engine.
        Any exception is logged but NOT re-raised: losing observability
        data must never break a successful Run.
        """
        if not self._persist_to_db:
            return 0
        with self._buffer_lock:
            if not self._db_buffer:
                return 0
            buffered, self._db_buffer = self._db_buffer, []
            self._last_flush_at = time.monotonic()
        try:
            from app.models.database import TrainingLog, sync_session_factory

            with sync_session_factory() as session:
                session.bulk_insert_mappings(
                    TrainingLog,
                    [
                        {
                            "task_id": self.task_id,
                            "level": entry["level"],
                            "message": entry["message"],
                            "extra": entry["extra"],
                            "created_at": entry["created_at"],
                        }
                        for entry in buffered
                    ],
                )
                session.commit()
            return len(buffered)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "TrainingLogger.flush_to_db failed for task %s: %s",
                self.task_id,
                exc,
            )
            # Push entries back so a later flush can retry, keeping the most
            # recent entries if the backlog has grown past the cap.
            with self._buffer_lock:
                merged = buffered + self._db_buffer
                self._db_buffer = merged[-self._MAX_BUFFERED_ENTRIES:]
            return 0

    def log_metrics(self, step: int, total_steps: int, metrics: dict):
        """Record metrics for a training step/fold and publish to event bus."""
        timestamp = datetime.now(timezone.utc).isoformat()

        step_data = {
            "step": step,
            "total_steps": total_steps,
            "metrics": metrics,
            "timestamp": timestamp,
        }

        self._metrics_data["steps"].append(step_data)
        self._save_metrics()

        # Also write to text log
        metrics_str = " ".join(
            f"{k}={v}" for k, v in metrics.items() if k != "fold"
        )
        self.log("INFO", f"[Step {step}/{total_steps}] {metrics_str}")

        # Publish to event bus for WebSocket
        progress = round((step / total_steps) * 100, 1)
        event_bus.publish(
            f"training:{self.task_id}",
            {
                "type": "metrics",
                "task_id": self.task_id,
                "step": step,
                "total_steps": total_steps,
                "progress": progress,
                "metrics": metrics,
                "timestamp": timestamp,
            },
        )

    def log_status(
        self,
        status: str,
        message: str = "",
        result_metrics: dict | None = None,
    ):
        """Publish a status change event."""
        self.log("INFO", f"Status: {status} — {message}")
        event_bus.publish(
            f"training:{self.task_id}",
            {
                "type": "status",
                "task_id": self.task_id,
                "status": status,
                "message": message,
                "result_metrics": result_metrics,
            },
        )

    def _save_metrics(self):
        with open(self.metrics_file, "w", encoding="utf-8") as f:
            json.dump(self._metrics_data, f, indent=2, default=str)

    def get_log_content(
        self,
        level_filter: str | None = None,
        page: int = 1,
        page_size: int = 100,
    ) -> dict:
        """Read log file and return paginated entries."""
        entries: list[dict[str, Any]] = []
        if self.log_file.exists():
            with open(self.log_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    # Parse: timestamp | LEVEL | message | extras
                    parts = line.split(" | ", 3)
                    if len(parts) >= 3:
                        entry_level = parts[1].strip()
                        if (
                            level_filter
                            and entry_level != level_filter.upper()
                        ):
                            continue
                        extra = (
                            _parse_extra_fields(parts[3])
                            if len(parts) == 4 and parts[3].strip()
                            else None
                        )
                        entries.append(
                            {
                                "level": entry_level,
                                "message": parts[2],
                                "extra": extra,
                                "created_at": parts[0],
                            }
                        )

        total = len(entries)
        start = (page - 1) * page_size
        end = start + page_size

        return {
            "task_id": self.task_id,
            "entries": entries[start:end],
            "total": total,
            "page": page,
            "page_size": page_size,
        }

    def get_metrics(self) -> dict:
        """Read metrics JSON file."""
        if self.metrics_file.exists():
            with open(self.metrics_file, "r", encoding="utf-8") as f:
                return json.load(f)
        return self._metrics_data

    def export(self, fmt: str = "txt") -> Path:
        """Return the path to the log/metrics file for download."""
        if fmt == "json":
            return self.metrics_file
        return self.log_file

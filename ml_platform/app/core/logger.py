"""Training logger — per-task file + metrics logging with event bus."""

import asyncio
import json
import logging
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
    """

    def __init__(self):
        self._subscribers: dict[str, list[asyncio.Queue]] = defaultdict(list)

    def subscribe(self, channel: str) -> asyncio.Queue:
        queue = asyncio.Queue()
        self._subscribers[channel].append(queue)
        return queue

    def unsubscribe(self, channel: str, queue: asyncio.Queue):
        if channel in self._subscribers:
            self._subscribers[channel] = [
                q for q in self._subscribers[channel] if q is not queue
            ]
            if not self._subscribers[channel]:
                del self._subscribers[channel]

    def publish(self, channel: str, message: dict):
        for queue in self._subscribers.get(channel, []):
            try:
                queue.put_nowait(message)
            except asyncio.QueueFull:
                pass  # drop if consumer is too slow


# Global event bus singleton
event_bus = EventBus()


class TrainingLogger:
    """Per-task logger that writes to files, publishes to event bus, and
    buffers entries for eventual persistence to the `training_logs` table.

    Persistence is deliberately deferred: sklearn trials emit ~10-50 log
    lines, so we accumulate in memory and flush once via `flush_to_db()`
    at the end of the Run (called from `_run_training_sync`). That avoids
    one DB round-trip per log line without sacrificing durability for
    anything short of a worker crash mid-training — in which case the
    .log file on disk still has the entries.
    """

    def __init__(self, task_id: str, model_type: str = ""):
        self.task_id = task_id
        self.model_type = model_type
        settings = get_settings()
        self.log_dir = settings.storage_logs
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.log_file = self.log_dir / f"{task_id}.log"
        self.metrics_file = self.log_dir / f"{task_id}_metrics.json"

        # In-memory buffer of (level, message, extra, created_at) tuples
        # awaiting a DB flush.
        self._db_buffer: list[dict[str, Any]] = []

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

        # Buffer for eventual DB flush
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

    def flush_to_db(self) -> int:
        """Persist all buffered log entries to `training_logs`.

        Idempotent (drains the buffer), safe to call multiple times. Returns
        the number of rows inserted. Designed to be called from sync code
        (e.g. a ThreadPoolExecutor worker) — uses the sync SQLAlchemy engine.
        Any exception is logged but NOT re-raised: losing observability
        data must never break a successful Run.
        """
        if not self._db_buffer:
            return 0
        buffered, self._db_buffer = self._db_buffer, []
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
            # Push entries back so a later flush can retry.
            self._db_buffer = buffered + self._db_buffer
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

"""Celery queue names and kind routing contract."""

from __future__ import annotations

from collections.abc import Iterable, Mapping


TRAIN = "train"
EXPLAIN = "explain"
FORECAST = "forecast"
DEFAULT = "default"

DECLARED_QUEUES = frozenset({TRAIN, EXPLAIN, FORECAST, DEFAULT})

KIND_TO_QUEUE: dict[str, str] = {
    "train": TRAIN,
    "dl_train": TRAIN,
    "explain": EXPLAIN,
    "ts_forecast": FORECAST,
    "forecast": FORECAST,
}


def assert_queue_contract(
    declared_queues: Iterable[str] = DECLARED_QUEUES,
    kind_to_queue: Mapping[str, str] = KIND_TO_QUEUE,
) -> None:
    """Fail fast when one or more task kinds route to undeclared queues."""
    declared = set(declared_queues)
    missing = {
        kind: queue for kind, queue in kind_to_queue.items() if queue not in declared
    }
    if missing:
        details = ", ".join(
            f"{kind!r} -> {queue!r}" for kind, queue in sorted(missing.items())
        )
        raise RuntimeError(
            f"Celery queue contract violation: {details}; "
            f"declared queues={sorted(declared)!r}"
        )

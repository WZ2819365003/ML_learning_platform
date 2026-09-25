"""Process-local serialization for task dispatch and final evaluation."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from weakref import WeakValueDictionary


_locks: WeakValueDictionary[str, asyncio.Lock] = WeakValueDictionary()


def _task_lock(task_id: str) -> asyncio.Lock:
    lock = _locks.get(task_id)
    if lock is None:
        lock = asyncio.Lock()
        _locks[task_id] = lock
    return lock


@asynccontextmanager
async def task_lifecycle_guard(task_id: str):
    lock = _task_lock(task_id)
    async with lock:
        yield

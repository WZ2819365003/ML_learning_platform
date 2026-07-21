"""
Executor Registry — unified dispatch point for all PlatformTask kinds.

V3 Phase 2 lays the abstraction so the Scheduler layer has one way to invoke
work, independent of whether the work is ML training, DL training, SHAP, or
future DAG node kinds (prepare_data / evaluate / aggregate).  Before this
registry existed, callers branched on ``kind`` and imported the relevant
service function directly — which meant adding a new kind required patching
every call site.

Contract
========
An executor is an ``async`` callable with the signature::

    async def executor(domain_id: str, platform_task_id: str) -> dict[str, Any]

The return dict should carry at least ``{"metrics": {...}}`` for the caller
to write back to ``PlatformTask.metrics_snapshot`` — but additional keys are
allowed and will be preserved as-is.

Services that own an executor register it at import time by calling
``register_executor(kind, fn)`` from a module-level statement.  This keeps
the registry population side-effectful but implicit (just importing the
service is enough), which matches how FastAPI routers are wired.
"""
from __future__ import annotations

import logging
from importlib import import_module
from typing import Any, Awaitable, Callable, Protocol, runtime_checkable

logger = logging.getLogger(__name__)


@runtime_checkable
class Executor(Protocol):
    """Protocol documenting the (domain_id, platform_task_id) → dict contract."""

    async def __call__(
        self, domain_id: str, platform_task_id: str
    ) -> dict[str, Any]: ...


_EXECUTORS: dict[str, Executor] = {}

_EXECUTOR_MODULES = (
    "app.services.training_service",
    "app.services.dl_service",
    "app.services.explain_service",
)


def load_executor_modules() -> None:
    """Import every service module that registers scheduler executors."""
    for module_name in _EXECUTOR_MODULES:
        import_module(module_name)


def register_executor(kind: str, fn: Callable[..., Awaitable[dict[str, Any]]]) -> None:
    """
    Register an executor for a given PlatformTask.kind.

    Idempotent — re-registering the same kind overwrites silently with a
    debug log.  Services call this at module import so by the time any
    scheduler or tuning logic runs the registry is fully populated.
    """
    if kind in _EXECUTORS:
        logger.debug("Overwriting executor for kind=%r (prev=%r, new=%r)",
                     kind, _EXECUTORS[kind], fn)
    _EXECUTORS[kind] = fn  # type: ignore[assignment]


def get_executor(kind: str) -> Executor:
    """
    Look up the executor for a kind.

    Raises
    ------
    KeyError
        If no executor is registered.  Callers should treat this as a
        programming error — either the service module that owns the kind
        wasn't imported yet, or the kind is misspelled.
    """
    try:
        return _EXECUTORS[kind]
    except KeyError as exc:
        raise KeyError(
            f"No executor registered for kind={kind!r}. "
            f"Known kinds: {sorted(_EXECUTORS.keys())!r}. "
            f"Check that the owning service module is imported at startup."
        ) from exc


def list_registered_kinds() -> list[str]:
    """Return a sorted copy of currently-registered kinds — useful for diagnostics."""
    return sorted(_EXECUTORS.keys())


def has_executor(kind: str) -> bool:
    """Cheap ``kind in registry`` check without raising."""
    return kind in _EXECUTORS


# ---------------------------------------------------------------------------
# Convenience: wrapper that adds PlatformTask status write-back around any
# executor.  Used by the Scheduler so every executor doesn't have to repeat
# the RUNNING → SUCCESS / FAILED transition.
# ---------------------------------------------------------------------------

async def run_with_status(
    kind: str, domain_id: str, platform_task_id: str
) -> dict[str, Any]:
    """
    Invoke the executor for ``kind`` with standard status write-back.

    Transitions the PlatformTask through RUNNING → SUCCESS/FAILED and
    persists ``metrics`` from the executor's return value.  Re-raises the
    underlying exception so the caller can log / retry / propagate as
    appropriate — status has already been set to FAILED before the raise.
    """
    # Local import: task_runner imports executors eagerly, which would
    # otherwise create a cycle at import time.
    from app.scheduler.task_runner import update_platform_task_status

    executor = get_executor(kind)
    try:
        await update_platform_task_status(platform_task_id, "RUNNING", progress=0.0)
        result = await executor(domain_id, platform_task_id)
        await update_platform_task_status(
            platform_task_id, "SUCCESS", metrics=result.get("metrics", {}),
        )
        return result
    except Exception as exc:  # noqa: BLE001
        await update_platform_task_status(
            platform_task_id, "FAILED", error=str(exc),
        )
        raise

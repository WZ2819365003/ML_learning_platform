"""SHAP summaries are cached on the task row.

SHAP is the most expensive thing viz_service does — a TreeExplainer on an
unbounded-depth forest took six minutes in production — and the result cannot
change while the model does not. Recomputing on every tab open made an already
slow operation look broken.
"""
from types import SimpleNamespace
from unittest import mock

import pytest

from app.services import viz_service


class _Task:
    """Stands in for a TrainingTask row: just the JSON column we write to."""

    def __init__(self, result_metrics=None):
        self.id = "task-1"
        self.result_metrics = result_metrics


class _Db:
    def __init__(self):
        self.flushes = 0

    async def flush(self):
        self.flushes += 1


@pytest.fixture
def patched(monkeypatch):
    """Patch the resolver and the expensive computation."""
    state = SimpleNamespace(task=_Task(), compute_calls=0, payload={"method": "tree", "top": [1]})

    async def _resolve(task_id, db):
        return state.task, None

    async def _compute(task_id, db, max_samples=200):
        state.compute_calls += 1
        return state.payload

    monkeypatch.setattr(viz_service.shap_service, "resolve_task_and_dataset", _resolve)
    monkeypatch.setattr(viz_service.shap_service, "compute_shap_summary", _compute)
    return state


async def test_first_call_computes_and_stores(patched):
    db = _Db()
    out = await viz_service.get_shap_summary("task-1", db, max_samples=100)

    assert patched.compute_calls == 1
    assert out["cached"] is False
    stored = patched.task.result_metrics["shap_cache"]
    assert stored["max_samples"] == 100
    assert stored["payload"] == patched.payload


async def test_second_call_returns_the_cache_without_recomputing(patched):
    db = _Db()
    await viz_service.get_shap_summary("task-1", db, max_samples=100)
    out = await viz_service.get_shap_summary("task-1", db, max_samples=100)

    assert patched.compute_calls == 1, "the cached summary should not be recomputed"
    assert out["cached"] is True
    assert out["method"] == "tree"


async def test_a_different_sample_size_is_a_different_question(patched):
    """A summary over 100 samples is not the answer to a request for 300."""
    db = _Db()
    await viz_service.get_shap_summary("task-1", db, max_samples=100)
    await viz_service.get_shap_summary("task-1", db, max_samples=300)
    assert patched.compute_calls == 2


async def test_refresh_forces_a_recomputation(patched):
    db = _Db()
    await viz_service.get_shap_summary("task-1", db, max_samples=100)
    out = await viz_service.get_shap_summary("task-1", db, max_samples=100, refresh=True)
    assert patched.compute_calls == 2
    assert out["cached"] is False


async def test_existing_metrics_are_preserved(patched):
    """The cache shares result_metrics with the run's real metrics."""
    patched.task.result_metrics = {"rmse": 72.4673}
    db = _Db()
    await viz_service.get_shap_summary("task-1", db, max_samples=100)
    assert patched.task.result_metrics["rmse"] == 72.4673
    assert "shap_cache" in patched.task.result_metrics


async def test_a_storage_failure_still_returns_the_result(patched, monkeypatch):
    """The user just waited minutes; a cache write problem must not lose that."""
    async def _boom(self):
        raise RuntimeError("database is down")

    db = _Db()
    monkeypatch.setattr(_Db, "flush", _boom)
    out = await viz_service.get_shap_summary("task-1", db, max_samples=100)
    assert out["method"] == "tree"
    assert out["cached"] is False

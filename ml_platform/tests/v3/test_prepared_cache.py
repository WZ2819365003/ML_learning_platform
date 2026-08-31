"""The prepared model+split bundle is cached across visualization calls.

Every chart endpoint calls resolve_and_load, and each call used to re-read the
model off disk and re-split the whole dataset. On an 87k-row set that is most
of the wait when a result page opens six charts at once.
"""
import pytest

from app.services import resolver


class _Task:
    id = "t1"
    model_path = "/tmp/model.joblib"
    target_column = "y"
    test_size = 0.2
    model_type = "xgboost_regressor"


@pytest.fixture(autouse=True)
def _clean_cache():
    resolver.clear_prepared_cache()
    yield
    resolver.clear_prepared_cache()


@pytest.fixture
def counted(monkeypatch):
    calls = {"load_model": 0, "split": 0}

    async def _resolve(task_id, db):
        return _Task(), type("DS", (), {"file_path": "/tmp/d.csv"})()

    def _load_model(path):
        calls["load_model"] += 1
        return object()

    def _split(*args, **kwargs):
        calls["split"] += 1
        return {"X_test": [[1]], "y_test": [1]}

    monkeypatch.setattr(resolver, "resolve_task_and_dataset", _resolve)
    monkeypatch.setattr(resolver, "load_model", _load_model)
    monkeypatch.setattr(resolver, "load_and_split_data_for_model", _split)
    monkeypatch.setattr(resolver, "is_regressor", lambda _m: True)
    return calls


async def test_second_call_reuses_the_bundle(counted):
    await resolver.resolve_and_load("t1", db=None)
    await resolver.resolve_and_load("t1", db=None)
    assert counted["load_model"] == 1, "the model should be read from disk once"
    assert counted["split"] == 1, "the dataset should be split once"


async def test_a_different_split_is_a_different_entry(counted):
    # A stratified split is a different hold-out; serving one for the other
    # would quietly change which rows a chart is drawn from.
    await resolver.resolve_and_load("t1", db=None, stratified=False)
    await resolver.resolve_and_load("t1", db=None, stratified=True)
    assert counted["split"] == 2


async def test_callers_cannot_contaminate_each_other(counted):
    """Chart code adds its own keys to the returned dict."""
    first = await resolver.resolve_and_load("t1", db=None)
    first["injected_by_caller"] = True
    second = await resolver.resolve_and_load("t1", db=None)
    assert "injected_by_caller" not in second


async def test_the_cache_stays_bounded(counted, monkeypatch):
    """Each entry holds a model plus hold-out arrays; it must not grow forever."""
    async def _resolve_many(task_id, db):
        t = _Task()
        t.id = task_id
        return t, type("DS", (), {"file_path": "/tmp/d.csv"})()

    monkeypatch.setattr(resolver, "resolve_task_and_dataset", _resolve_many)
    for i in range(resolver._PREPARED_CACHE_MAX + 3):
        await resolver.resolve_and_load(f"task-{i}", db=None)
    assert len(resolver._PREPARED_CACHE) == resolver._PREPARED_CACHE_MAX


async def test_evicts_the_least_recently_used(counted, monkeypatch):
    async def _resolve_many(task_id, db):
        t = _Task()
        t.id = task_id
        return t, type("DS", (), {"file_path": "/tmp/d.csv"})()

    monkeypatch.setattr(resolver, "resolve_task_and_dataset", _resolve_many)
    for i in range(resolver._PREPARED_CACHE_MAX):
        await resolver.resolve_and_load(f"task-{i}", db=None)

    await resolver.resolve_and_load("task-0", db=None)      # refresh the oldest
    await resolver.resolve_and_load("overflow", db=None)    # forces one eviction

    keys = {k[0] for k in resolver._PREPARED_CACHE}
    assert "task-0" in keys, "a just-used entry should survive"
    assert "task-1" not in keys, "the least recently used one should go"

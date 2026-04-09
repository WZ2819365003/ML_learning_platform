"""Tests for deploy service LRU cache logic."""
import pytest
from app.services.deploy_service import _ModelCache


def test_model_cache_lru_eviction(tmp_path):
    """LRU cache evicts the least-recently-used entry when full."""
    import joblib
    from sklearn.linear_model import LinearRegression
    import numpy as np

    paths = []
    for i in range(3):
        m = LinearRegression()
        m.fit([[1]], [1])
        p = str(tmp_path / f"m{i}.joblib")
        joblib.dump(m, p)
        paths.append(p)

    cache = _ModelCache(max_size=2)
    cache.get("d0", paths[0])
    cache.get("d1", paths[1])
    assert "d0" in cache._cache
    assert "d1" in cache._cache

    # Adding d2 should evict d0 (LRU)
    cache.get("d2", paths[2])
    assert "d0" not in cache._cache
    assert "d1" in cache._cache
    assert "d2" in cache._cache


def test_model_cache_evict():
    cache = _ModelCache(max_size=5)
    cache._cache["fake_id"] = "fake_model"
    cache.evict("fake_id")
    assert "fake_id" not in cache._cache


def test_model_cache_miss_returns_model(tmp_path):
    import joblib
    from sklearn.linear_model import LinearRegression

    m = LinearRegression()
    m.fit([[1], [2]], [1, 2])
    p = str(tmp_path / "model.joblib")
    joblib.dump(m, p)

    cache = _ModelCache(max_size=5)
    loaded = cache.get("dep1", p)
    assert hasattr(loaded, "predict")


def test_model_cache_hit_reuses_object(tmp_path):
    import joblib
    from sklearn.linear_model import LinearRegression

    m = LinearRegression()
    m.fit([[1], [2]], [1, 2])
    p = str(tmp_path / "model.joblib")
    joblib.dump(m, p)

    cache = _ModelCache(max_size=5)
    first = cache.get("dep1", p)
    second = cache.get("dep1", p)
    assert first is second  # same object, no re-load

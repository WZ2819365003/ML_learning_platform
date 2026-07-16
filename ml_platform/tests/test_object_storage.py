from types import SimpleNamespace

from app.services import object_storage


def _settings():
    return SimpleNamespace(
        s3_enabled=True,
        s3_endpoint_url="http://127.0.0.1:9000",
        s3_access_key="test",
        s3_secret_key="test",
        s3_bucket="test-bucket",
    )


def test_s3_client_is_configured_to_fail_fast(monkeypatch):
    captured = {}
    sentinel = object()

    def fake_client(*args, **kwargs):
        captured.update(kwargs)
        return sentinel

    monkeypatch.setattr(object_storage, "get_settings", _settings)
    monkeypatch.setattr(object_storage.boto3, "client", fake_client)

    assert object_storage._get_client() is sentinel
    config = captured["config"]
    assert config.connect_timeout <= 2
    assert config.read_timeout <= 3
    assert config.retries["total_max_attempts"] == 1


def test_upload_failure_opens_process_circuit(monkeypatch, tmp_path):
    object_storage._reset_circuit_for_tests()
    calls = {"head": 0}

    class FailingClient:
        def head_bucket(self, **_kwargs):
            calls["head"] += 1
            raise RuntimeError("storage unavailable")

    artifact = tmp_path / "model.joblib"
    artifact.write_bytes(b"model")
    monkeypatch.setattr(object_storage, "get_settings", _settings)
    monkeypatch.setattr(object_storage, "_get_client", lambda: FailingClient())

    assert object_storage.upload_file(artifact, "models/one.joblib") is None
    assert object_storage.upload_file(artifact, "models/two.joblib") is None
    assert calls["head"] == 1

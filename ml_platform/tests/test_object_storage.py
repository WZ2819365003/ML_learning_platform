import io
from types import SimpleNamespace

from botocore.exceptions import ClientError

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


def test_restore_file_downloads_missing_artifact_atomically(monkeypatch, tmp_path):
    class Body:
        def __init__(self):
            self.payload = io.BytesIO(b"persisted-artifact")
            self.read_sizes = []

        def read(self, size=-1):
            self.read_sizes.append(size)
            return self.payload.read(size)

    body = Body()

    class Client:
        def get_object(self, **kwargs):
            if kwargs["Key"] == "models/missing.joblib":
                raise ClientError(
                    {"Error": {"Code": "NoSuchKey", "Message": "missing"}},
                    "GetObject",
                )
            assert kwargs == {"Bucket": "test-bucket", "Key": "models/task.joblib"}
            return {"Body": body}

    destination = tmp_path / "nested" / "task.joblib"
    monkeypatch.setattr(object_storage, "get_settings", _settings)
    monkeypatch.setattr(object_storage, "_get_client", lambda: Client())

    restored_key = object_storage.restore_file(
        destination,
        ["models/missing.joblib", "models/task.joblib"],
    )

    assert restored_key == "models/task.joblib"
    assert destination.read_bytes() == b"persisted-artifact"
    assert list(destination.parent.glob("*.tmp")) == []
    assert body.read_sizes and set(body.read_sizes) == {object_storage._DOWNLOAD_CHUNK_SIZE}


def test_dataset_object_key_is_stable_across_runtime_roots():
    assert object_storage.dataset_object_key(
        "dataset-123",
        "/app/storage/uploads/abc-source.csv",
    ) == "datasets/dataset-123/original/abc-source.csv"


def test_restore_dataset_file_uses_dataset_scoped_key(monkeypatch, tmp_path):
    local_path = tmp_path / "storage" / "uploads" / "source.csv"

    def restore_file(path, keys):
        assert path == local_path
        assert keys == ["datasets/dataset-123/original/source.csv"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("feature,target\n1,0\n")
        return keys[0]

    monkeypatch.setattr(object_storage, "resolve_runtime_path", lambda _path: local_path)
    monkeypatch.setattr(object_storage, "restore_file", restore_file)

    assert object_storage.restore_dataset_file(
        "dataset-123",
        "storage/uploads/source.csv",
    ) == local_path


def test_restore_model_bundle_fetches_dl_checkpoint_and_sidecars(monkeypatch, tmp_path):
    model_path = tmp_path / "storage" / "models" / "dl_task.pt"
    requested = []

    def restore_file(path, keys):
        requested.append((path.name, keys[0]))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"artifact")
        return keys[0]

    monkeypatch.setattr(object_storage, "resolve_runtime_path", lambda _path: model_path)
    monkeypatch.setattr(object_storage, "restore_file", restore_file)

    assert object_storage.restore_model_bundle("storage/models/dl_task.pt") == model_path
    assert requested == [
        ("dl_task.pt", "models/dl_task.pt"),
        ("dl_task.pt.scaler.joblib", "models/dl_task.pt.scaler.joblib"),
        ("dl_task.pt.preprocessor.joblib", "models/dl_task.pt.preprocessor.joblib"),
    ]

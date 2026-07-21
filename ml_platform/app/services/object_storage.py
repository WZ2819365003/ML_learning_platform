"""Object storage service — S3-compatible (MinIO) client.

Convention:
    models/  {task_id}.joblib          ← ML model artifacts
    models/  dl_{task_id}.pt           ← DL model checkpoints
    models/  dl_{task_id}.pt.scaler.joblib
    logs/    {task_id}.log             ← human-readable training log
    logs/    {task_id}_metrics.json    ← step-by-step metrics

This module exposes a synchronous helper ``upload_file`` and an async
helper ``upload_file_async``.  Training services run in a ThreadPoolExecutor
so they use the sync version directly; async callers use the async wrapper.

S3_ENABLED=false (in .env) disables uploads silently — useful for
local-only development without a running MinIO instance.
"""

from __future__ import annotations

import logging
import os
import tempfile
import threading
import time
from pathlib import Path

import boto3
from botocore.client import Config
from botocore.exceptions import ClientError

from app.config import get_settings
from app.utils.storage_paths import resolve_runtime_path

logger = logging.getLogger(__name__)

_CIRCUIT_COOLDOWN_SECONDS = 30.0
_circuit_open_until = 0.0
_circuit_lock = threading.Lock()

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_client():
    """Return a boto3 S3 client pointed at MinIO."""
    s = get_settings()
    return boto3.client(
        "s3",
        endpoint_url=s.s3_endpoint_url,
        aws_access_key_id=s.s3_access_key,
        aws_secret_access_key=s.s3_secret_key,
        config=Config(
            signature_version="s3v4",
            connect_timeout=2,
            read_timeout=3,
            retries={"mode": "standard", "total_max_attempts": 1},
        ),
        region_name="us-east-1",  # MinIO ignores region but boto3 requires it
    )


def _ensure_bucket(client, bucket: str) -> None:
    """Create the bucket if it does not already exist."""
    try:
        client.head_bucket(Bucket=bucket)
    except ClientError as e:
        if e.response["Error"]["Code"] in ("404", "NoSuchBucket"):
            client.create_bucket(Bucket=bucket)
            logger.info("Created MinIO bucket: %s", bucket)
        else:
            raise


def _circuit_is_open() -> bool:
    with _circuit_lock:
        return time.monotonic() < _circuit_open_until


def _open_circuit() -> None:
    global _circuit_open_until
    with _circuit_lock:
        _circuit_open_until = time.monotonic() + _CIRCUIT_COOLDOWN_SECONDS


def _close_circuit() -> None:
    global _circuit_open_until
    with _circuit_lock:
        _circuit_open_until = 0.0


def _reset_circuit_for_tests() -> None:
    _close_circuit()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def upload_file(local_path: str | Path, object_key: str) -> str | None:
    """Upload a local file to MinIO.

    Returns the object key on success, or None if S3 is disabled / upload fails.

    Args:
        local_path: Absolute path to the file on disk.
        object_key:  Destination key inside the bucket, e.g. ``models/abc.joblib``.
    """
    settings = get_settings()
    if not settings.s3_enabled:
        return None
    if _circuit_is_open():
        logger.debug("Object storage circuit open; skipping upload %s", object_key)
        return None

    local_path = Path(local_path)
    if not local_path.exists():
        logger.warning("upload_file: %s does not exist, skipping", local_path)
        return None

    try:
        client = _get_client()
        _ensure_bucket(client, settings.s3_bucket)
        client.upload_file(str(local_path), settings.s3_bucket, object_key)
        _close_circuit()
        logger.info("Uploaded %s → s3://%s/%s", local_path.name, settings.s3_bucket, object_key)
        return object_key
    except Exception as exc:
        _open_circuit()
        logger.warning("Object storage upload failed (%s): %s", object_key, exc)
        return None


def upload_training_artifacts(task_id: str, model_files: list[Path], log_files: list[Path]) -> None:
    """Upload all model and log artifacts for a completed training task.

    Designed to be called from a training thread after the task finishes.
    Failures are logged as warnings and never propagate to the caller.

    Args:
        task_id:     The training task UUID.
        model_files: List of model file paths (e.g. .joblib, .pt, .pt.scaler.joblib).
        log_files:   List of log file paths (e.g. .log, _metrics.json).
    """
    for path in model_files:
        if path.exists():
            upload_file(path, f"models/{path.name}")

    for path in log_files:
        if path.exists():
            upload_file(path, f"logs/{path.name}")


def dataset_object_key(dataset_id: str, local_path: str | Path) -> str:
    """Return the stable object key for a dataset's original uploaded file."""
    return f"datasets/{dataset_id}/original/{Path(local_path).name}"


def upload_dataset_file(dataset_id: str, local_path: str | Path) -> str | None:
    """Persist an original dataset file under its stable dataset-scoped key."""
    return upload_file(local_path, dataset_object_key(dataset_id, local_path))


def restore_dataset_file(dataset_id: str, file_path: str | Path) -> Path | None:
    """Ensure a dataset's original file exists locally, restoring it if needed."""
    destination = resolve_runtime_path(file_path)
    if destination.exists():
        return destination
    restored = restore_file(
        destination,
        [dataset_object_key(dataset_id, destination)],
    )
    return destination if restored and destination.exists() else None


def restore_model_bundle(model_path: str | Path) -> Path | None:
    """Restore an ML model or a DL checkpoint with its optional sidecars."""
    destination = resolve_runtime_path(model_path)
    restore_file(destination, [f"models/{destination.name}"])

    if destination.suffix.lower() == ".pt":
        for suffix in (".scaler.joblib", ".preprocessor.joblib"):
            sidecar = Path(str(destination) + suffix)
            restore_file(sidecar, [f"models/{sidecar.name}"])

    return destination if destination.exists() else None


def restore_file(
    local_path: str | Path,
    object_keys: list[str] | tuple[str, ...],
) -> str | None:
    """Restore a missing local file from the first available MinIO object.

    The payload is written to a temporary file in the destination directory and
    atomically renamed so concurrent readers never observe a partial artifact.
    Returns the object key used, or ``None`` when no remote copy is available.
    """
    destination = Path(local_path)
    if destination.exists():
        return None

    settings = get_settings()
    if not settings.s3_enabled:
        return None

    try:
        client = _get_client()
    except Exception as exc:
        logger.warning("Object storage client unavailable while restoring %s: %s", destination, exc)
        return None

    destination.parent.mkdir(parents=True, exist_ok=True)
    for object_key in object_keys:
        temp_name: str | None = None
        try:
            response = client.get_object(Bucket=settings.s3_bucket, Key=object_key)
            body = response["Body"]
            payload = body.read()
            close = getattr(body, "close", None)
            if callable(close):
                close()

            with tempfile.NamedTemporaryFile(
                mode="wb",
                prefix=f".{destination.name}.",
                suffix=".tmp",
                dir=destination.parent,
                delete=False,
            ) as tmp:
                temp_name = tmp.name
                tmp.write(payload)
                tmp.flush()
                os.fsync(tmp.fileno())
            os.replace(temp_name, destination)
            logger.info("Restored s3://%s/%s -> %s", settings.s3_bucket, object_key, destination)
            return object_key
        except ClientError as exc:
            code = str(exc.response.get("Error", {}).get("Code", ""))
            if code not in {"404", "NoSuchKey", "NoSuchObject"}:
                logger.warning("Object restore failed (%s): %s", object_key, exc)
        except Exception as exc:
            logger.warning("Object restore failed (%s): %s", object_key, exc)
        finally:
            if temp_name:
                Path(temp_name).unlink(missing_ok=True)
    return None


def get_presigned_url(object_key: str, expires_in: int = 3600) -> str | None:
    """Return a pre-signed download URL for ``object_key``, valid for ``expires_in`` seconds."""
    settings = get_settings()
    if not settings.s3_enabled:
        return None
    try:
        client = _get_client()
        return client.generate_presigned_url(
            "get_object",
            Params={"Bucket": settings.s3_bucket, "Key": object_key},
            ExpiresIn=expires_in,
        )
    except Exception as exc:
        logger.warning("Failed to generate presigned URL for %s: %s", object_key, exc)
        return None


def list_objects(prefix: str = "") -> list[dict]:
    """List objects under ``prefix`` (e.g. ``models/`` or ``logs/``)."""
    settings = get_settings()
    if not settings.s3_enabled:
        return []
    try:
        client = _get_client()
        response = client.list_objects_v2(Bucket=settings.s3_bucket, Prefix=prefix)
        return [
            {
                "key": obj["Key"],
                "size": obj["Size"],
                "last_modified": obj["LastModified"].isoformat(),
            }
            for obj in response.get("Contents", [])
        ]
    except Exception as exc:
        logger.warning("list_objects(%s) failed: %s", prefix, exc)
        return []


def download_object_bytes(object_key: str) -> bytes | None:
    """Fetch an object's body as bytes. Returns None if S3 is disabled or the key is missing."""
    settings = get_settings()
    if not settings.s3_enabled:
        return None
    try:
        client = _get_client()
        obj = client.get_object(Bucket=settings.s3_bucket, Key=object_key)
        return obj["Body"].read()
    except Exception as exc:
        logger.warning("download_object_bytes(%s) failed: %s", object_key, exc)
        return None

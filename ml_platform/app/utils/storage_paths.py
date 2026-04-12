"""Helpers for storing and resolving runtime file paths across environments."""

from __future__ import annotations

from pathlib import Path, PurePosixPath, PureWindowsPath

from app.config import get_settings


def _raw_parts(raw_path: str | Path) -> tuple[str, ...]:
    text = str(raw_path).strip()
    if not text:
        return ()

    parser = PureWindowsPath if ("\\" in text or (len(text) >= 2 and text[1] == ":")) else PurePosixPath
    parts: list[str] = []
    for part in parser(text).parts:
        normalized = part.replace("\\", "/").strip("/")
        if not normalized or normalized.endswith(":"):
            continue
        parts.append(normalized)
    return tuple(parts)


def _map_legacy_storage_path(raw_path: str | Path) -> Path | None:
    settings = get_settings()
    parts = _raw_parts(raw_path)
    if not parts:
        return None

    lower_parts = [part.lower() for part in parts]
    storage_roots = {
        ("storage", "uploads"): settings.storage_uploads,
        ("storage", "models"): settings.storage_models,
        ("storage", "logs"): settings.storage_logs,
        ("mlruns",): settings.project_root / "mlruns",
    }

    for marker, root in storage_roots.items():
        marker_len = len(marker)
        for index in range(len(lower_parts) - marker_len + 1):
            if tuple(lower_parts[index:index + marker_len]) != marker:
                continue
            suffix = parts[index + marker_len:]
            return root.joinpath(*suffix).resolve()
    return None


def resolve_runtime_path(raw_path: str | Path) -> Path:
    """Resolve a stored path against the current runtime storage layout."""
    text = str(raw_path).strip()
    candidate = Path(text)

    if candidate.is_absolute() and candidate.exists():
        return candidate.resolve()

    mapped = _map_legacy_storage_path(text)
    if mapped is not None:
        return mapped

    settings = get_settings()
    if candidate.is_absolute():
        return candidate
    return (settings.project_root / candidate).resolve()


def to_portable_storage_path(raw_path: str | Path) -> str:
    """Store project-local paths as portable POSIX-style relative paths."""
    resolved = resolve_runtime_path(raw_path)
    project_root = get_settings().project_root.resolve()
    try:
        return resolved.relative_to(project_root).as_posix()
    except ValueError:
        return resolved.as_posix()

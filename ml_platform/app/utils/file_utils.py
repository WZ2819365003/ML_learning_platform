"""File utility functions for storage management."""

import os
import uuid
from pathlib import Path


def ensure_directories(paths: list[str]) -> None:
    """Create each directory in *paths* if it does not already exist.

    Intermediate parent directories are created as needed.
    """
    for raw_path in paths:
        Path(raw_path).mkdir(parents=True, exist_ok=True)


def generate_unique_filename(original_name: str) -> str:
    """Return a filename prefixed with a UUID to guarantee uniqueness.

    Example
    -------
    >>> generate_unique_filename("iris.csv")
    'a1b2c3d4-...-iris.csv'
    """
    unique_prefix = uuid.uuid4().hex[:12]
    return f"{unique_prefix}-{original_name}"


def get_file_size(path: str) -> int:
    """Return the size of the file at *path* in bytes.

    Raises
    ------
    FileNotFoundError
        If *path* does not point to an existing file.
    """
    return os.path.getsize(path)

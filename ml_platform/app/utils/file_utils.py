"""File utility functions for storage management."""

import os
import uuid
from pathlib import Path


def generate_unique_filename(original_name: str) -> str:
    """Return a filename prefixed with a UUID to guarantee uniqueness.

    Example
    -------
    >>> generate_unique_filename("iris.csv")
    'a1b2c3d4-...-iris.csv'
    """
    unique_prefix = uuid.uuid4().hex[:12]
    return f"{unique_prefix}-{original_name}"

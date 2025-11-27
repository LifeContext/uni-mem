"""Utility helpers for multimodal processing."""

from __future__ import annotations

import hashlib
import mimetypes
import os
from pathlib import Path
from typing import Iterator, Optional, Tuple


def guess_file_extension(name: str) -> str:
    """Return normalized file extension (without dot)."""

    return Path(name).suffix.lower().lstrip(".")


def guess_mime_type(path: str) -> Optional[str]:
    """Guess mime type using Python's mimetypes registry."""

    mime, _ = mimetypes.guess_type(path)
    return mime


def compute_file_hash(
    *,
    file_path: Optional[Path] = None,
    data: Optional[bytes] = None,
    algorithm: str = "sha256",
) -> Tuple[str, str]:
    """Return (algorithm, hex_digest) for the provided data or file."""

    hasher = hashlib.new(algorithm)
    if file_path:
        with open(file_path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1024 * 1024), b""):
                hasher.update(chunk)
    elif data is not None:
        hasher.update(data)
    else:
        raise ValueError("Either file_path or data must be provided")
    return algorithm, hasher.hexdigest()


def iter_file_chunks(path: Path, chunk_size: int = 1024 * 1024) -> Iterator[bytes]:
    """Yield file contents in chunks."""

    with open(path, "rb") as fh:
        while True:
            chunk = fh.read(chunk_size)
            if not chunk:
                break
            yield chunk


def ensure_directory(path: Path) -> None:
    """Ensure directory exists for the given path."""

    os.makedirs(path, exist_ok=True)


"""File hashing helpers used for deterministic OCR caching."""
from __future__ import annotations

import hashlib
from pathlib import Path


def file_sha1(path: str | Path) -> str:
    """Computes the SHA-1 hex digest of a file in streaming 1 MiB chunks."""
    digest = hashlib.sha1()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()

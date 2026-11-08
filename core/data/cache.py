"""Publish dataset caches atomically, so concurrent jobs never read a partial one."""

import hashlib
import os
from pathlib import Path
import shutil


def file_sha256(path):
    """Hex SHA-256 of a file, read in 1 MiB blocks."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def save_dataset(dataset, cache):
    """Write ``dataset`` beside ``cache`` and rename it into place.

    If another process published the same cache first, keep theirs: cache
    names encode every generation input, so both copies are identical.
    """
    cache = Path(cache)
    cache.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache.with_name(f".{cache.name}.tmp-{os.getpid()}")
    shutil.rmtree(temporary, ignore_errors=True)
    dataset.save_to_disk(str(temporary))
    try:
        os.replace(temporary, cache)
    except OSError:
        if not cache.exists():
            raise
        shutil.rmtree(temporary, ignore_errors=True)

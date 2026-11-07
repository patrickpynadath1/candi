"""Publish dataset caches atomically, so concurrent jobs never read a partial one."""

import os
from pathlib import Path
import shutil


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

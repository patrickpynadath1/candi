"""Restore checkpoint mtimes that nested tuning recorded, after a copy from S3 reset them.

Nested tuning fingerprints a trained checkpoint by (size, mtime_ns) and refuses to reuse
inference results when the fingerprint changes. S3 does not keep mtimes, so a machine that
restores a run tree would otherwise fail every case that had finished training. For each
checkpoint path recorded in a ``fingerprint``, set its mtime back when the size matches and
all records of that path agree; report anything else and leave it untouched.

    python3 restore_mtimes.py RUN_TREE
"""

import json
import os
import sys
from pathlib import Path


def fingerprints(value, found):
    if isinstance(value, dict):
        path, mark = value.get("checkpoint"), value.get("fingerprint")
        if isinstance(path, str) and isinstance(mark, dict) and {"size", "mtime_ns"} <= set(mark):
            found.setdefault(path, set()).add((mark["size"], mark["mtime_ns"]))
        for item in value.values():
            fingerprints(item, found)
    elif isinstance(value, list):
        for item in value:
            fingerprints(item, found)


def main(root):
    found = {}
    for path in Path(root).rglob("*.json"):
        try:
            fingerprints(json.loads(path.read_text()), found)
        except (OSError, ValueError):
            continue
    fixed = skipped = 0
    for checkpoint, marks in sorted(found.items()):
        target = Path(checkpoint)
        if len(marks) != 1 or not target.is_file():
            print(f"skip {checkpoint}: {'conflicting records' if len(marks) != 1 else 'missing'}")
            skipped += 1
            continue
        (size, mtime_ns), = marks
        stat = target.stat()
        if stat.st_size != size:
            print(f"skip {checkpoint}: size {stat.st_size} != recorded {size}")
            skipped += 1
        elif stat.st_mtime_ns != mtime_ns:
            os.utime(target, ns=(mtime_ns, mtime_ns))
            fixed += 1
    print(f"restored mtimes of {fixed} checkpoints, skipped {skipped}, {len(found)} recorded")


if __name__ == "__main__":
    main(sys.argv[1])

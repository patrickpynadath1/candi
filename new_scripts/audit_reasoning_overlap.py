#!/usr/bin/env python3
"""Exact puzzle overlap: all released Shah training examples versus Extreme test.

Does not claim equivalence under Sudoku symmetries. No test labels are inspected.
"""

import argparse
import json
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tasks.reasoning_data import file_hash


def audit(raw):
    shah = raw / "shah/Sudoku-train-data.npy"
    extreme = raw / "eqr/sudoku-extreme-1k-aug-1000/test/all__inputs.npy"
    source = np.load(shah, mmap_mode="r")
    target = np.load(extreme, mmap_mode="r")
    fingerprints = set()
    for offset in range(0, len(source), 16384):
        rows = source[offset : offset + 16384]
        cells = rows[:, 1:].reshape(-1, 81, 4)
        board = np.zeros((len(rows), 81), dtype=np.uint8)
        given = np.arange(81)[None, :] < rows[:, 0, None]
        board[np.arange(len(rows))[:, None], cells[:, :, 0] * 9 + cells[:, :, 1]] = (
            np.where(given, cells[:, :, 2], 0)
        )
        fingerprints.update(map(bytes, board))
    overlaps = [
        i
        for i, row in enumerate(target)
        if bytes((row - 1).astype(np.uint8)) in fingerprints
    ]
    return dict(
        protocol="exact puzzle bytes; no symmetry equivalence audit",
        shah_train_records=len(source),
        shah_unique_puzzles=len(fingerprints),
        extreme_test_records=len(target),
        overlap_count=len(overlaps),
        extreme_source_ids=[f"sudoku_extreme/test/{i}" for i in overlaps],
        sources={str(p.relative_to(raw)): file_hash(p) for p in (shah, extreme)},
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw", type=Path, default=Path("data_cache/reasoning_raw"))
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    result = audit(args.raw)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print({k: v for k, v in result.items() if k != "extreme_source_ids"})

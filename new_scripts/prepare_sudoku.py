#!/usr/bin/env python3
"""Build and validate the shared deterministic Sudoku dataset cache."""

import argparse
from pathlib import Path
import sys
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from tasks.sudoku import SudokuTokenizer, get_sudoku_dataset  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", type=Path, default=Path("data_cache"))
    parser.add_argument("--num-train", type=int, default=200_000)
    parser.add_argument("--num-valid", type=int, default=10_000)
    parser.add_argument("--difficulty", default="easy")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()

    config = SimpleNamespace(
        data=SimpleNamespace(
            cache_dir=str(args.cache_dir.resolve()),
            difficulty=args.difficulty,
            num_train=args.num_train,
            num_valid=args.num_valid,
            data_seed=args.seed,
            sudoku_num_workers=args.workers,
        )
    )
    dataset = get_sudoku_dataset(config, SudokuTokenizer())
    assert len(dataset["train"]) == args.num_train
    assert len(dataset["validation"]) == args.num_valid
    print(
        f"SUDOKU_CACHE_OK train={len(dataset['train'])} "
        f"validation={len(dataset['validation'])} cache={args.cache_dir.resolve()}"
    )


if __name__ == "__main__":
    main()

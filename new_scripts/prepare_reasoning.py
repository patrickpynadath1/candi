#!/usr/bin/env python3
"""Convert downloaded releases; --limit produces an explicitly labeled smoke subset."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tasks.reasoning_data import prepare

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--task",
        required=True,
        choices=["sudoku_extreme", "sudoku_shah", "zebra", "maze_unique"],
    )
    p.add_argument("--raw", type=Path, default=Path("data_cache/reasoning_raw"))
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--layout", nargs=2, type=int)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dev-fraction", type=float, default=0.05)
    p.add_argument("--limit", type=int)
    p.add_argument(
        "--full-training",
        action="store_true",
        help="all released training groups; no development split; fixed-budget refit only",
    )
    args = vars(p.parse_args())
    print(json.dumps(prepare(**args), indent=2))

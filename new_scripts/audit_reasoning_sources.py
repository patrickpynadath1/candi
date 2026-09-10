#!/usr/bin/env python3
"""Independent gold-label audit of the released arrays and symbolic clues."""

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tasks.reasoning_data import source_records
from tasks.maze import maze_metrics
from tasks.zebra import parse_clues, zebra_metrics


def valid_sudoku(puzzles, answers):
    answers = np.asarray(answers).reshape(-1, 9, 9)
    puzzles = np.asarray(puzzles).reshape(-1, 9, 9)
    expected = np.arange(1, 10)
    boxes = answers.reshape(-1, 3, 3, 3, 3).transpose(0, 1, 3, 2, 4).reshape(-1, 9, 9)
    return bool(
        np.all(np.sort(answers, axis=2) == expected)
        and np.all(np.sort(answers.transpose(0, 2, 1), axis=2) == expected)
        and np.all(np.sort(boxes, axis=2) == expected)
        and np.all((puzzles == 0) | (puzzles == answers))
    )


def audit(raw, include_zebra=False):
    report = {}
    for task in ("sudoku_extreme", "sudoku_shah"):
        for split in ("train", "test"):
            if task == "sudoku_extreme":
                directory = raw / "eqr/sudoku-extreme-1k-aug-1000" / split
                inputs = np.load(directory / "all__inputs.npy", mmap_mode="r")
                labels = np.load(directory / "all__labels.npy", mmap_mode="r")
            else:
                inputs = np.load(
                    raw / "shah" / f"Sudoku-{split}-data.npy", mmap_mode="r"
                )
            for offset in range(0, len(inputs), 8192):
                rows = inputs[offset : offset + 8192]
                if task == "sudoku_extreme":
                    puzzles = rows - 1
                    answers = labels[offset : offset + 8192] - 1
                else:
                    cells = rows[:, 1:].reshape(-1, 81, 4)
                    coords = cells[:, :, 0] * 9 + cells[:, :, 1]
                    if not np.all(np.sort(coords, axis=1) == np.arange(81)):
                        raise ValueError("invalid coordinates")
                    answers = np.zeros((len(rows), 81), dtype=np.uint8)
                    answers[np.arange(len(rows))[:, None], coords] = cells[:, :, 2]
                    puzzles = np.zeros_like(answers)
                    given = np.arange(81)[None, :] < rows[:, 0, None]
                    puzzles[np.arange(len(rows))[:, None], coords] = np.where(
                        given, cells[:, :, 2], 0
                    )
                    if np.any(cells[:, :, 3][given] != 0):
                        raise ValueError("invalid givens")
                if not valid_sudoku(puzzles, answers):
                    raise ValueError(f"invalid gold: {task}/{split}/{offset}")
            report[f"{task}/{split}"] = {
                "gold_labels_checked": len(inputs),
                "all_valid": True,
            }
    for split in ("train", "test"):
        count = 0
        for record in source_records(raw, "maze_unique", split):
            score = maze_metrics(record["answer"], record["answer"], record["puzzle"])
            if not score["optimal_path"] or not score["unique_shortest_path"]:
                raise ValueError(record["source_id"])
            count += 1
        report[f"maze_unique/{split}"] = {
            "gold_labels_checked": count,
            "all_valid": True,
        }
    if include_zebra:
        for split in ("train", "test"):
            count = 0
            layouts = Counter()
            max_prompt = 0
            for record in source_records(raw, "zebra", split):
                clues, tokens = parse_clues(record["puzzle"])
                if not zebra_metrics(record["answer"], record["answer"], clues)[
                    "constraints_satisfied"
                ]:
                    raise ValueError(record["source_id"])
                count += 1
                layouts[f"{record['attributes']}x{record['houses']}"] += 1
                max_prompt = max(max_prompt, len(tokens) + 6)
            report[f"zebra/{split}"] = {
                "gold_labels_checked": count,
                "all_valid": True,
                "layouts": dict(layouts),
                "max_prompt_tokens": max_prompt,
            }
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw", type=Path, default=Path("data_cache/reasoning_raw"))
    p.add_argument("--output", type=Path, required=True)
    p.add_argument(
        "--include-zebra",
        action="store_true",
        help="also parse/check every Zebra record (slower)",
    )
    args = p.parse_args()
    report = audit(args.raw, args.include_zebra)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report))

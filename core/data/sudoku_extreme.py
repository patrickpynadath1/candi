"""Pinned Sudoku-Extreme data and deterministic, lazy Sudoku symmetries."""

import csv
import hashlib
import json
from pathlib import Path
import tempfile

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from torch.utils.data import Dataset

from .sudoku import encode_example


def augment_sudoku(puzzle, solution, rng):
    """Same symmetry family as HRM/TRM: digits, transpose, bands/rows/stacks/cols."""
    digits = np.r_[0, rng.permutation(np.arange(1, 10))]
    transpose = rng.random() < 0.5
    rows = np.concatenate([3 * b + rng.permutation(3) for b in rng.permutation(3)])
    cols = np.concatenate([3 * b + rng.permutation(3) for b in rng.permutation(3)])

    def transform(grid):
        grid = grid.T if transpose else grid
        return digits[grid[np.ix_(rows, cols)]]

    return transform(puzzle), transform(solution)


def parse_pair(question, answer):
    if len(question) != 81 or len(answer) != 81:
        raise ValueError("Sudoku Extreme rows must contain 81 cells")
    if set(question) - set(".0123456789") or set(answer) - set("123456789"):
        raise ValueError("invalid Sudoku Extreme cell")
    puzzle = np.frombuffer(question.replace(".", "0").encode(), dtype=np.uint8) - 48
    solution = np.frombuffer(answer.encode(), dtype=np.uint8) - 48
    return np.stack((puzzle.reshape(9, 9), solution.reshape(9, 9)))


def validate_pairs(pairs):
    puzzle, solution = pairs[:, 0], pairs[:, 1]
    if np.any((puzzle != 0) & (puzzle != solution)):
        raise ValueError("Sudoku clues disagree with solution")
    expected = np.arange(1, 10)
    boxes = solution.reshape(-1, 3, 3, 3, 3).transpose(0, 1, 3, 2, 4).reshape(-1, 9, 9)
    if not all(
        np.all(np.sort(x, axis=-1) == expected)
        for x in (solution, solution.transpose(0, 2, 1), boxes)
    ):
        raise ValueError("invalid Sudoku solution")


def prepare_extreme(config):
    options = {
        key: config[key]
        for key in ("source_repo", "revision", "num_train", "num_valid", "data_seed")
    }
    if options["num_train"] < 1 or options["num_valid"] < 1:
        raise ValueError("training and validation sizes must be positive")
    identity = hashlib.sha256(json.dumps(options, sort_keys=True).encode()).hexdigest()[
        :16
    ]
    root = Path(config.cache_dir) / f"sudoku_extreme_v1_{identity}"
    if (root / "metadata.json").is_file():
        return root
    root.parent.mkdir(parents=True, exist_ok=True)
    paths = {
        split: Path(
            hf_hub_download(
                config.source_repo,
                f"{split}.csv",
                repo_type="dataset",
                revision=config.revision,
                cache_dir=str(Path(config.cache_dir) / "huggingface"),
            )
        )
        for split in ("train", "test")
    }
    with paths["train"].open() as f:
        count = sum(1 for _ in f) - 1
    size = config.num_train + config.num_valid
    if size > count:
        raise ValueError("not enough official training puzzles for disjoint splits")
    selected = np.random.default_rng(config.data_seed).choice(
        count, size, replace=False
    )
    positions = {int(index): pos for pos, index in enumerate(selected)}
    pool = np.empty((size, 2, 9, 9), dtype=np.uint8)
    with paths["train"].open(newline="") as f:
        for i, row in enumerate(csv.DictReader(f)):
            if i in positions:
                pool[positions[i]] = parse_pair(row["question"], row["answer"])
    with paths["test"].open(newline="") as f:
        test = np.stack(
            [parse_pair(row["question"], row["answer"]) for row in csv.DictReader(f)]
        )
    validate_pairs(pool)
    validate_pairs(test)
    # Reject exact solution leakage (including puzzles with different clue masks).
    train_keys = {row[1].tobytes() for row in pool[: config.num_train]}
    valid_keys = {row[1].tobytes() for row in pool[config.num_train :]}
    test_keys = {row[1].tobytes() for row in test}
    if train_keys & valid_keys or train_keys & test_keys or valid_keys & test_keys:
        raise ValueError("overlapping solutions across Sudoku Extreme partitions")
    test = test[np.random.default_rng(config.data_seed + 1).permutation(len(test))]
    with tempfile.TemporaryDirectory(dir=root.parent, prefix=".sudoku-extreme-") as tmp:
        tmp = Path(tmp)
        for split, values in [
            ("train", pool[: config.num_train]),
            ("validation", pool[config.num_train :]),
            ("test", test),
        ]:
            np.save(tmp / f"{split}.npy", values)
        metadata = {
            **options,
            "official_train_count": count,
            "official_test_count": len(test),
            "selected_train_rows": selected[: config.num_train].tolist(),
            "selected_validation_rows": selected[config.num_train :].tolist(),
            "test_permutation_seed": config.data_seed + 1,
            "source_sha256": {
                s: hashlib.sha256(p.read_bytes()).hexdigest() for s, p in paths.items()
            },
        }
        (tmp / "metadata.json").write_text(json.dumps(metadata, indent=2))
        try:
            tmp.rename(root)
        except FileExistsError:
            if not (root / "metadata.json").is_file():
                raise
    return root


class SudokuPairs(Dataset):
    """Original + a fixed set of indexed augmentations, without a million-row cache."""

    def __init__(self, pairs, tokenizer, augmentations=0, seed=42):
        if augmentations < 0:
            raise ValueError("augmentations must be nonnegative")
        self.pairs, self.tokenizer = pairs, tokenizer
        self.augmentations, self.seed = int(augmentations), seed

    def __len__(self):
        return len(self.pairs) * (1 + self.augmentations)

    def __getitem__(self, index):
        if index < 0 or index >= len(self):
            raise IndexError(index)
        base, augmentation = divmod(index, self.augmentations + 1)
        puzzle, solution = self.pairs[base]
        if augmentation:
            rng = np.random.default_rng(
                np.random.SeedSequence([self.seed, base, augmentation])
            )
            puzzle, solution = augment_sudoku(puzzle, solution, rng)
        tokens, mask = encode_example(
            puzzle.tolist(), solution.tolist(), self.tokenizer
        )
        return {"input_ids": torch.tensor(tokens), "attention_mask": torch.tensor(mask)}

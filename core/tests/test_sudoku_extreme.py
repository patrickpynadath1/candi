"""Sudoku Extreme data reuse, leakage checks, and loss/generation boundaries."""

import csv
import json

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from core.data.sudoku import SudokuTokenizer, encode_example, sudoku_metrics
from core.data.sudoku_extreme import (
    augment_sudoku,
    prepare_extreme,
    SudokuPairs,
    validate_pairs,
)
from core.tasks.sudoku_extreme import SudokuExtremeTask


def pair():
    solution = np.array(
        [[(3 * r + r // 3 + c) % 9 + 1 for c in range(9)] for r in range(9)]
    )
    puzzle = solution.copy()
    puzzle.flat[::2] = 0
    return puzzle, solution


def test_augmentations_preserve_sudoku_and_clues():
    tokenizer = SudokuTokenizer()
    for seed in range(30):
        puzzle, solution = augment_sudoku(*pair(), np.random.default_rng(seed))
        validate_pairs(np.array([[puzzle, solution]]))
        tokens, _ = encode_example(puzzle.tolist(), solution.tolist(), tokenizer)
        assert sudoku_metrics(
            torch.tensor(tokens[91:]), torch.tensor(tokens[91:]), tokens[1:90]
        )["valid_solution"]
    data = SudokuPairs(np.array([pair()]), tokenizer, augmentations=1000, seed=7)
    assert len(data) == 1001
    assert torch.equal(data[133]["input_ids"], data[133]["input_ids"])
    assert not torch.equal(data[0]["input_ids"], data[133]["input_ids"])


def test_cache_partitions_and_task_interface(tmp_path, monkeypatch):
    from core.data import sudoku_extreme

    for split, start, count in [("train", 0, 20), ("test", 20, 8)]:
        with (tmp_path / f"{split}.csv").open("w") as f:
            writer = csv.writer(f)
            writer.writerow(["source", "question", "answer", "rating"])
            for seed in range(start, start + count):
                puzzle, solution = augment_sudoku(*pair(), np.random.default_rng(seed))
                writer.writerow(
                    [
                        "fixture",
                        "".join(map(str, puzzle.flatten())).replace("0", "."),
                        "".join(map(str, solution.flatten())),
                        1,
                    ]
                )
    monkeypatch.setattr(
        sudoku_extreme,
        "hf_hub_download",
        lambda repo, filename, **kw: str(tmp_path / filename),
    )
    data = OmegaConf.create(
        dict(
            source_repo="fixture",
            revision="pinned",
            num_train=2,
            num_valid=3,
            data_seed=42,
            augmentations=5,
            cache_dir=str(tmp_path / "cache"),
        )
    )
    root = prepare_extreme(data)
    meta = json.loads((root / "metadata.json").read_text())
    assert not set(meta["selected_train_rows"]) & set(meta["selected_validation_rows"])
    monkeypatch.setattr(
        sudoku_extreme,
        "hf_hub_download",
        lambda *a, **kw: pytest.fail("cache must be reused"),
    )
    task = SudokuExtremeTask(data)
    assert len(task.training_dataset()) == 12
    assert len(task.validation_dataset()) == 3
    assert len(task.evaluation_dataset("test")) == 8
    examples = [task.validation_dataset()[0], task.validation_dataset()[1]]
    batch = task.collate_training(examples)
    assert not batch["loss_mask"][:, :91].any()
    tokens, fixed = task.prepare_generation(examples)
    assert fixed[:, :91].all() and not fixed[:, 91:].any()
    assert not tokens[:, 91:].any()
    metrics = task.create_metric_accumulator("test")
    metrics.update(torch.stack([e["input_ids"] for e in examples]), examples, [0, 1])
    assert metrics.compute().accuracy == 1

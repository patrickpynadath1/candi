"""Small deterministic N-Queens datasets with exact solver-generated labels."""

import itertools
import random
from pathlib import Path

import datasets


class NQueensTokenizer:
    pad_token_id = 0
    mask_token_id = None
    eos_token_id = None
    bos_token_id = None

    def __init__(self, size):
        self.size = size
        self.vocab_size = size + 1
        self.prompt_len = size
        self.total_seq_len = 2 * size

    def __len__(self):
        return self.vocab_size

    def decode(self, tokens, skip_special_tokens=False):
        return " ".join(str(int(x)) for x in tokens)

    def batch_decode(self, sequences, skip_special_tokens=False):
        return [self.decode(x, skip_special_tokens) for x in sequences]


def enumerate_solutions(size):
    """Return all labeled-board solutions; columns are one-based, row implicit."""
    if size not in (8, 10):
        raise ValueError("N-Queens currently supports board sizes 8 and 10")
    solutions = []

    def visit(columns, used, down, up):
        row = len(columns)
        if row == size:
            solutions.append(tuple(columns))
            return
        for col in range(1, size + 1):
            if col not in used and row - col not in down and row + col not in up:
                visit(
                    columns + [col], used | {col}, down | {row - col}, up | {row + col}
                )

    visit([], set(), set(), set())
    return solutions


def get_nqueens_dataset(config):
    size = int(config.size)
    minimum, maximum = int(config.min_clues), int(config.max_clues)
    counts = [int(config.num_train), int(config.num_valid)]
    if not 1 <= minimum <= maximum < size:
        raise ValueError("require 1 <= min_clues <= max_clues < size")
    if min(counts) <= 0:
        raise ValueError("training and validation counts must be positive")
    fraction = float(config.validation_solution_fraction)
    if not 0 < fraction < 1:
        raise ValueError("validation_solution_fraction must lie in (0,1)")
    seed = int(config.data_seed)
    cache = Path(config.cache_dir) / (
        f"nqueens_v1_n{size}_clues{minimum}-{maximum}_train{counts[0]}"
        f"_valid{counts[1]}_holdout{fraction:g}_seed{seed}"
    )
    if cache.exists():
        return datasets.load_from_disk(str(cache)).with_format("torch")
    solutions = enumerate_solutions(size)
    rng = random.Random(seed)
    rng.shuffle(solutions)
    split = max(1, min(len(solutions) - 1, round(len(solutions) * fraction)))
    pools = [solutions[split:], solutions[:split]]
    used_prompts = set()
    outputs = {}
    # Enumerating candidates avoids an unbounded rejection loop near capacity.
    clue_rows = [
        rows
        for k in range(minimum, maximum + 1)
        for rows in itertools.combinations(range(size), k)
    ]
    for name, pool, count in zip(("train", "validation"), pools, counts):
        candidates = [(solution, rows) for solution in pool for rows in clue_rows]
        rng.shuffle(candidates)
        records = []
        for solution, rows in candidates:
            selected = set(rows)
            puzzle = tuple(
                solution[row] if row in selected else 0 for row in range(size)
            )
            if puzzle in used_prompts:
                continue
            used_prompts.add(puzzle)
            records.append({"input_ids": list(puzzle + solution)})
            if len(records) == count:
                break
        if len(records) < count:
            raise ValueError(
                f"requested {count} unique {name} prompts but only {len(records)} "
                "available; reduce counts or increase clue range"
            )
        outputs[name] = datasets.Dataset.from_list(records)
    result = datasets.DatasetDict(outputs)
    cache.parent.mkdir(parents=True, exist_ok=True)
    result.save_to_disk(str(cache))
    return result.with_format("torch")


def is_valid_solution(columns, size):
    values = [int(x) for x in columns]
    return (
        len(values) == size
        and set(values) == set(range(1, size + 1))
        and len({row - col for row, col in enumerate(values)}) == size
        and len({row + col for row, col in enumerate(values)}) == size
    )

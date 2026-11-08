"""Sudoku with clues removed: does a sampler find many valid solutions or one?

A Sudoku puzzle with fewer clues admits more than one solution. This task takes
``num_puzzles`` validation puzzles, removes all but ``retained_fraction`` of each
one's clues (a nested, seeded subset, so fewer retained clues always means a subset
of the clues kept at a higher fraction), and asks for ``repetitions`` generations
per weakened puzzle. Its ``diversity`` evaluation counts, per puzzle, how many
generations are valid solutions of the weakened puzzle and how many distinct
boards they are. Training and the ``validation`` evaluation are plain Sudoku, so a
checkpoint trained with ``task=sudoku`` is evaluated with
``task=sudoku_diversity mode=evaluate``.
"""

from dataclasses import dataclass

import torch

from core.data.sudoku import SudokuTokenizer
from .base import PrimaryMetric
from .sudoku import SudokuTask
from .sudoku_metrics import SudokuMetrics, SudokuRecord


@dataclass(frozen=True)
class SudokuDiversityRecord:
    puzzle_index: int
    original_hints: int
    retained_hints: int
    generations: int
    valid: int
    unique: int
    valid_unique: int


@dataclass(frozen=True)
class SudokuDiversityAggregate:
    num_generations: int
    valid_count: int
    valid_rate: float
    unique_count_within_prompts: int
    valid_unique_count_within_prompts: int
    mean_unique_per_prompt: float
    mean_valid_unique_per_prompt: float
    pooled_unique_boards: int
    pooled_valid_unique_boards: int
    prompts: list[SudokuDiversityRecord]
    records: list[SudokuRecord]


def remove_hints(example, fraction, seed):
    """Keep a seeded ``fraction`` of the clues; separators and answer tokens are untouched.

    Returns (example, original clue count, retained clue count). The subsets are
    nested in ``fraction`` for a fixed seed.
    """
    if not 0 <= fraction <= 1:
        raise ValueError("retained hint fraction must lie in [0, 1]")
    result = {key: value.clone() for key, value in example.items()}
    puzzle = result["input_ids"][1 : SudokuTokenizer.prompt_len - 1]
    positions = ((puzzle >= 1) & (puzzle <= 9)).nonzero().flatten()
    order = torch.randperm(len(positions), generator=torch.Generator().manual_seed(seed))
    keep = int(len(positions) * fraction + 0.5)
    puzzle[positions[order[keep:]]] = 0
    return result, len(positions), keep


def canonical_board(text):
    """The 81 digits of a decoded answer grid, or None if it is not a full grid."""
    symbols = text.split()
    if len(symbols) != 89 or any(symbols[i] != "|" for i in range(9, 89, 10)):
        return None
    digits = [s for i, s in enumerate(symbols) if i % 10 != 9]
    if len(digits) != 81 or any(s not in "123456789" or len(s) != 1 for s in digits):
        return None
    return "".join(digits)


def diversity_summary(records, groups, repetitions):
    """Aggregate per-generation records into per-puzzle and pooled diversity counts.

    ``groups`` lists (puzzle index, original clues, retained clues) per weakened
    puzzle; generation ``i`` belongs to group ``i // repetitions``.
    """
    rows = {}
    for record in records:
        rows.setdefault(record.index // repetitions, []).append(record)
    if not rows or any(len(rows.get(g, [])) != repetitions for g in rows):
        raise ValueError(
            "diversity needs every generation of each scored puzzle; "
            "make evaluation.max_examples a multiple of task.repetitions"
        )
    summaries, pooled, pooled_valid = [], set(), set()
    for group in sorted(rows):
        index, original, retained = groups[group]
        unique, valid_unique = set(), set()
        for record in rows[group]:
            board = canonical_board(record.generated)
            if board is not None:
                unique.add(board)
                if record.valid_solution:
                    valid_unique.add(board)
        pooled |= unique
        pooled_valid |= valid_unique
        summaries.append(SudokuDiversityRecord(
            index, original, retained, len(rows[group]),
            sum(r.valid_solution for r in rows[group]), len(unique), len(valid_unique),
        ))
    total = len(records)
    return SudokuDiversityAggregate(
        num_generations=total,
        valid_count=sum(s.valid for s in summaries),
        valid_rate=sum(s.valid for s in summaries) / total,
        unique_count_within_prompts=sum(s.unique for s in summaries),
        valid_unique_count_within_prompts=sum(s.valid_unique for s in summaries),
        mean_unique_per_prompt=sum(s.unique for s in summaries) / len(summaries),
        mean_valid_unique_per_prompt=sum(s.valid_unique for s in summaries) / len(summaries),
        pooled_unique_boards=len(pooled),
        pooled_valid_unique_boards=len(pooled_valid),
        prompts=summaries,
        records=list(records),
    )


class SudokuDiversityMetrics:
    """Score each generation as Sudoku, then count valid and distinct boards per puzzle."""

    supports_variable_length = True

    def __init__(self, tokenizer, groups, repetitions):
        self.sudoku = SudokuMetrics(tokenizer)
        self.groups, self.repetitions = groups, repetitions

    def update(self, samples, examples, indices, *, lengths=None):
        self.sudoku.update(samples, examples, indices, lengths=lengths)

    def compute(self):
        if not self.sudoku.records:
            raise ValueError("cannot score an empty evaluation")
        return diversity_summary(self.sudoku.records, self.groups, self.repetitions)


class SudokuDiversityTask(SudokuTask):
    name = "sudoku_diversity"
    primary_metric = PrimaryMetric("mean_valid_unique_per_prompt", "max", "diversity")

    def __init__(
        self,
        data,
        vocab_size=12,
        length=180,
        variable_length=False,
        *,
        retained_fraction=0.5,
        repetitions=8,
        num_puzzles=32,
        seed=0,
    ):
        super().__init__(data, vocab_size, length, variable_length)
        if repetitions < 1 or num_puzzles < 1:
            raise ValueError("repetitions and num_puzzles must be positive")
        if not 0 <= retained_fraction <= 1:
            raise ValueError("retained_fraction must lie in [0, 1]")
        self.retained_fraction, self.repetitions = retained_fraction, repetitions
        self.num_puzzles, self.seed = num_puzzles, seed
        self._diversity = None

    def _weakened(self):
        """Each weakened puzzle ``repetitions`` times, and its (index, clues) group."""
        if self._diversity is None:
            validation = self.validation_dataset()
            if self.num_puzzles > len(validation):
                raise ValueError(
                    f"num_puzzles={self.num_puzzles} exceeds the {len(validation)} validation puzzles"
                )
            examples, groups = [], []
            for index in range(self.num_puzzles):
                example, original, retained = remove_hints(
                    validation[index], self.retained_fraction, self.seed + index
                )
                groups.append((index, original, retained))
                examples.extend([example] * self.repetitions)
            self._diversity = examples, groups
        return self._diversity

    def evaluation_dataset(self, name):
        if name == "diversity":
            return self._weakened()[0]
        return super().evaluation_dataset(name)

    def create_metric_accumulator(self, evaluation_name):
        if evaluation_name == "diversity":
            return SudokuDiversityMetrics(self.tokenizer, self._weakened()[1], self.repetitions)
        return super().create_metric_accumulator(evaluation_name)

"""Matched clue removal and repeated-generation Sudoku diversity metrics."""
from dataclasses import dataclass
import torch
from core.data.sudoku import SudokuTokenizer
from core.tasks.sudoku import SudokuTask


@dataclass(frozen=True)
class SudokuDiversityRecord:
    puzzle_index: int
    original_hints: int
    retained_hints: int
    generations: int
    valid: int
    unique: int
    valid_unique: int


def remove_hints(example, fraction, seed):
    """Nested deterministic clue subsets; preserve separators and answer tokens."""
    if not 0 <= fraction <= 1:
        raise ValueError('retained hint fraction must lie in [0, 1]')
    result = {key: value.clone() for key, value in example.items()}
    puzzle = result['input_ids'][1:SudokuTokenizer.prompt_len - 1]
    positions = ((puzzle >= 1) & (puzzle <= 9)).nonzero().flatten()
    order = torch.randperm(len(positions), generator=torch.Generator().manual_seed(seed))
    keep = int(len(positions) * fraction + .5)
    puzzle[positions[order[keep:]]] = 0
    return result, len(positions), keep


class HintRemovalTask(SudokuTask):
    def __init__(self, base, puzzle_indices, repetitions, fraction, seed):
        super().__init__(base.data, base.vocab_size, base.length)
        self.examples = []
        self.groups = []
        for index in puzzle_indices:
            example, original, retained = remove_hints(
                base.validation_dataset()[index], fraction, seed + index
            )
            self.groups.append((index, original, retained))
            self.examples.extend([example] * repetitions)
        self.repetitions = repetitions

    def validation_dataset(self):
        return self.examples


def canonical_board(text):
    symbols = text.split()
    if len(symbols) != 89 or any(symbols[i] != '|' for i in range(9, 89, 10)):
        return None
    digits = [s for i, s in enumerate(symbols) if i % 10 != 9]
    if len(digits) != 81 or any(s not in '123456789' or len(s) != 1 for s in digits):
        return None
    return ''.join(digits)


def diversity_summary(records, task):
    summaries = []
    pooled, pooled_valid = set(), set()
    for group, (index, original, retained) in enumerate(task.groups):
        rows = records[group * task.repetitions:(group + 1) * task.repetitions]
        if len(rows) != task.repetitions:
            raise ValueError('incomplete repeated-generation group')
        unique, valid_unique = set(), set()
        for row in rows:
            board = canonical_board(row['generated'])
            if board is not None:
                unique.add(board)
                if row['valid_solution']:
                    valid_unique.add(board)
        pooled.update(unique)
        pooled_valid.update(valid_unique)
        summaries.append(SudokuDiversityRecord(index, original, retained, len(rows),
                         sum(r['valid_solution'] for r in rows), len(unique), len(valid_unique)))
    return {
        'num_generations': len(records),
        'valid_count': sum(r.valid for r in summaries),
        'valid_rate': sum(r.valid for r in summaries) / len(records),
        'unique_count_within_prompts': sum(r.unique for r in summaries),
        'valid_unique_count_within_prompts': sum(r.valid_unique for r in summaries),
        'mean_unique_per_prompt': sum(r.unique for r in summaries) / len(summaries),
        'mean_valid_unique_per_prompt': sum(r.valid_unique for r in summaries) / len(summaries),
        'pooled_unique_boards': len(pooled),
        'pooled_valid_unique_boards': len(pooled_valid),
        'prompts': summaries,
    }

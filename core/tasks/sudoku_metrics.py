"""Typed Sudoku records and aggregate metrics."""

from dataclasses import dataclass

from core.data.sudoku import SudokuTokenizer, sudoku_metrics


@dataclass(frozen=True)
class SudokuRecord:
    index: int
    generated: str
    ground_truth: str
    exact: bool
    cell_accuracy: float
    digit_accuracy: float
    valid_grid: bool
    valid_rows: float
    valid_columns: float
    valid_boxes: float
    clue_consistency: bool
    valid_solution: bool


@dataclass(frozen=True)
class SudokuAggregate:
    accuracy: float
    cell_accuracy: float
    digit_accuracy: float
    valid_grid_rate: float
    valid_row_rate: float
    valid_column_rate: float
    valid_box_rate: float
    valid_solution_rate: float
    clue_consistency_rate: float
    num_correct: int
    num_total: int
    records: list[SudokuRecord]


class SudokuMetrics:
    supports_variable_length = True

    def __init__(self, tokenizer: SudokuTokenizer):
        self.tokenizer = tokenizer
        self.records: list[SudokuRecord] = []

    def update(self, samples, examples, indices, *, lengths=None) -> None:
        start = self.tokenizer.prompt_len
        for row, (sample, example, index) in enumerate(zip(samples, examples, indices)):
            if lengths is not None:
                sample = sample[: lengths[row]]
            prediction = sample[start:].cpu()
            target = example["input_ids"][start:].cpu()
            puzzle = example["input_ids"][1 : start - 1].cpu()
            self.records.append(
                SudokuRecord(
                    index=index,
                    generated=self.tokenizer.decode(prediction),
                    ground_truth=self.tokenizer.decode(target),
                    **sudoku_metrics(prediction, target, puzzle),
                )
            )

    def compute(self) -> SudokuAggregate:
        if not self.records:
            raise ValueError("cannot score an empty evaluation")
        records = self.records
        count = len(records)
        return SudokuAggregate(
            accuracy=sum(r.exact for r in records) / count,
            cell_accuracy=sum(r.cell_accuracy for r in records) / count,
            digit_accuracy=sum(r.digit_accuracy for r in records) / count,
            valid_grid_rate=sum(r.valid_grid for r in records) / count,
            valid_row_rate=sum(r.valid_rows for r in records) / count,
            valid_column_rate=sum(r.valid_columns for r in records) / count,
            valid_box_rate=sum(r.valid_boxes for r in records) / count,
            valid_solution_rate=sum(r.valid_solution for r in records) / count,
            clue_consistency_rate=sum(r.clue_consistency for r in records) / count,
            num_correct=sum(r.exact for r in records),
            num_total=count,
            records=list(records),
        )

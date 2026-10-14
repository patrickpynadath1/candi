"""Reference-independent correctness and diversity for N-Queens."""

from collections import defaultdict
from dataclasses import dataclass

from core.data.nqueens import is_valid_solution


@dataclass(frozen=True)
class NQueensRecord:
    index: int
    prompt: list[int]
    generated: list[int]
    ground_truth: list[int]
    exact: bool
    valid_board: bool
    clue_consistency: bool
    valid_solution: bool


@dataclass(frozen=True)
class NQueensAggregate:
    valid_solution_rate: float
    valid_board_rate: float
    clue_consistency_rate: float
    exact_match_rate: float
    unique_valid_solutions: int
    unique_valid_fraction: float
    mean_unique_valid_per_prompt: float
    solution_coverage: float
    num_total: int
    records: list[NQueensRecord]


class NQueensMetrics:
    supports_variable_length = True

    def __init__(self, size, solution_count):
        self.size = size
        self.solution_count = solution_count
        self.records = []

    def update(self, samples, examples, indices, *, lengths=None):
        for row, (sample, example, index) in enumerate(zip(samples, examples, indices)):
            if lengths is not None:
                sample = sample[: lengths[row]]
            values = sample.tolist()
            prompt = example["input_ids"][: self.size].tolist()
            target = example["input_ids"][self.size :].tolist()
            prediction = values[self.size :]
            valid = is_valid_solution(prediction, self.size)
            clues = len(prediction) == self.size and all(
                clue == 0 or prediction[row] == clue for row, clue in enumerate(prompt)
            )
            self.records.append(
                NQueensRecord(
                    index=int(index),
                    prompt=prompt,
                    generated=prediction,
                    ground_truth=target,
                    exact=prediction == target,
                    valid_board=valid,
                    clue_consistency=clues,
                    valid_solution=valid and clues,
                )
            )

    def compute(self):
        if not self.records:
            raise ValueError("cannot score an empty evaluation")
        records = self.records
        count = len(records)
        valid = [r for r in records if r.valid_solution]
        unique = {tuple(r.generated) for r in valid}
        groups = defaultdict(set)
        for record in records:
            group = groups[tuple(record.prompt)]
            if record.valid_solution:
                group.add(tuple(record.generated))
        return NQueensAggregate(
            valid_solution_rate=len(valid) / count,
            valid_board_rate=sum(r.valid_board for r in records) / count,
            clue_consistency_rate=sum(r.clue_consistency for r in records) / count,
            exact_match_rate=sum(r.exact for r in records) / count,
            unique_valid_solutions=len(unique),
            unique_valid_fraction=len(unique) / len(valid) if valid else 0.0,
            mean_unique_valid_per_prompt=sum(map(len, groups.values())) / len(groups),
            solution_coverage=len(unique) / self.solution_count,
            num_total=count,
            records=list(records),
        )

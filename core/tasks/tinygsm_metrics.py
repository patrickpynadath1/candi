"""Typed GSM8K generation records and original answer/accuracy mathematics."""

from dataclasses import dataclass, replace
import numpy as np
from .metrics.gsm8k import (
    extract_code,
    gold_answer,
    numeric_answer,
    numbers_equal,
    to_number,
)
from .sandbox import execute_python


@dataclass(frozen=True)
class TinyGSMRecord:
    index: int
    prompt: str
    response_ground_truth: str
    response: str
    prompt_tokens: int
    completion_tokens: int
    hit_eos: bool
    correct: bool = False
    parsed: bool = False
    executed: bool = False
    timed_out: bool = False
    sandbox_error: bool = False


@dataclass(frozen=True)
class TinyGSMAggregate:
    accuracy: float
    accuracy_se: float
    accuracy_ci_lo: float
    accuracy_ci_hi: float
    num_correct: int
    num_total: int
    scoring: str
    parsed_rate: float | None
    executed_rate: float | None
    timed_out_rate: float | None
    sandbox_error_rate: float | None
    eos_rate: float
    truncation_rate: float
    mean_completion_tokens: float
    records: list[TinyGSMRecord]


class TinyGSMMetrics:
    supports_variable_length = True

    def __init__(
        self, tokenizer, scoring="code", timeout=3.0, bootstrap_size=1, seed=1
    ):
        if scoring not in {"code", "numeric"}:
            raise ValueError("scoring must be code or numeric")
        if bootstrap_size < 1:
            raise ValueError("bootstrap_size must be positive")
        self.tokenizer, self.scoring, self.timeout = tokenizer, scoring, timeout
        self.bootstrap_size, self.seed = bootstrap_size, seed
        self.records = []

    def update(self, samples, examples, indices, *, lengths=None):
        for row, (sample, example, index) in enumerate(zip(samples, examples, indices)):
            if lengths is not None:
                sample = sample[: lengths[row]]
            prompt_length = min(len(example["input_ids"]), len(sample))
            completion = sample[prompt_length:].tolist()
            hit_eos = self.tokenizer.eos_token_id in completion
            if hit_eos:
                completion = completion[: completion.index(self.tokenizer.eos_token_id)]
            response = self.tokenizer.decode(completion, skip_special_tokens=True)
            record = TinyGSMRecord(
                index,
                example["prompt"],
                example["response_ground_truth"],
                response,
                prompt_length,
                len(completion),
                hit_eos,
            )
            if self.scoring == "code":
                execution = execute_python(
                    extract_code(response),
                    entrypoint="simple_math_problem",
                    timeout=self.timeout,
                )
                prediction = to_number(execution.result)
                if prediction is None and execution.executed:
                    prediction = numeric_answer(execution.stdout)
                record = replace(
                    record,
                    parsed=execution.parsed,
                    executed=execution.executed,
                    timed_out=execution.timed_out,
                    sandbox_error=execution.sandbox_error,
                )
            else:
                prediction = numeric_answer(response)
            self.records.append(
                replace(
                    record,
                    correct=numbers_equal(
                        prediction, gold_answer(record.response_ground_truth)
                    ),
                )
            )

    def compute(self):
        if not self.records:
            raise ValueError("cannot score an empty evaluation")
        correct = np.asarray([r.correct for r in self.records], dtype=np.float64)
        accuracy = float(correct.mean())
        low = high = accuracy
        standard_error = 0.0
        if self.bootstrap_size > 1:
            rng = np.random.default_rng(self.seed)
            sampled = rng.integers(
                0, len(correct), size=(self.bootstrap_size, len(correct))
            )
            estimates = correct[sampled].mean(1)
            standard_error = float(estimates.std(ddof=1))
            low, high = (float(v) for v in np.percentile(estimates, [2.5, 97.5]))

        def rate(name):
            return sum(getattr(r, name) for r in self.records) / len(self.records)

        return TinyGSMAggregate(
            accuracy,
            standard_error,
            low,
            high,
            int(correct.sum()),
            len(correct),
            self.scoring,
            *[
                rate(k) if self.scoring == "code" else None
                for k in ("parsed", "executed", "timed_out", "sandbox_error")
            ],
            rate("hit_eos"),
            1 - rate("hit_eos"),
            rate("completion_tokens"),
            list(self.records),
        )

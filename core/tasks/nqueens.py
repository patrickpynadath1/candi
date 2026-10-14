"""Conditional N-Queens completion and clue-free generation evaluation."""

import torch
from torch.utils.data import default_collate

from core.data.nqueens import NQueensTokenizer, get_nqueens_dataset, enumerate_solutions
from .base import PrimaryMetric, Task
from .nqueens_metrics import NQueensMetrics


class NQueensTask(Task):
    name = "nqueens"
    primary_metric = PrimaryMetric("valid_solution_rate", "max")

    def __init__(
        self,
        data,
        vocab_size,
        length,
        evaluation_mode="conditional",
        samples_per_prompt=1,
        variable_length=False,
    ):
        self.variable_length = variable_length
        self.data = data
        self.tokenizer = NQueensTokenizer(int(data.size))
        self.vocab_size, self.length = vocab_size, length
        if vocab_size != len(self.tokenizer) or length != self.tokenizer.total_seq_len:
            raise ValueError("NQueens requires vocab_size=size+1 and length=2*size")
        if evaluation_mode not in ("conditional", "unconditional"):
            raise ValueError("evaluation_mode must be conditional or unconditional")
        if samples_per_prompt < 1:
            raise ValueError("samples_per_prompt must be positive")
        self.evaluation_mode = evaluation_mode
        self.samples_per_prompt = samples_per_prompt
        self._dataset = None

    def _load(self):
        if self._dataset is None:
            self._dataset = get_nqueens_dataset(self.data)
        return self._dataset

    def prepare_data(self):
        self._load()

    def training_dataset(self):
        return self._load()["train"]

    def validation_dataset(self):
        return self._load()["validation"]

    def evaluation_dataset(self, name):
        if name != "validation":
            raise ValueError(f"unknown evaluation dataset: {name}")
        examples = []
        for example in self.validation_dataset():
            tokens = example["input_ids"].clone()
            if self.evaluation_mode == "unconditional":
                tokens[: self.tokenizer.size] = 0
            for _ in range(self.samples_per_prompt):
                examples.append({"input_ids": tokens.clone()})
        return examples

    def collate_training(self, examples):
        batch = default_collate(examples)
        loss = torch.zeros_like(batch["input_ids"], dtype=torch.bool)
        loss[:, self.tokenizer.size :] = True
        batch.update(
            loss_mask=loss,
            corruption_mask=loss.clone(),
            prompt_mask=~loss,
            padding_mask=torch.zeros_like(loss),
        )
        return batch

    def prepare_generation(self, examples):
        tokens = torch.stack([example["input_ids"] for example in examples])
        fixed = torch.zeros_like(tokens, dtype=torch.bool)
        fixed[:, : self.tokenizer.size] = True
        return torch.where(fixed, tokens, 0), fixed

    def create_metric_accumulator(self, evaluation_name):
        if evaluation_name != "validation":
            raise ValueError(f"unknown evaluation dataset: {evaluation_name}")
        return NQueensMetrics(
            self.tokenizer.size, len(enumerate_solutions(self.tokenizer.size))
        )

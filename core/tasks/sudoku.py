"""Sudoku data, representation, conditioning, and success metrics."""

import torch
from torch.utils.data import default_collate

from core.data.sudoku import SudokuTokenizer, get_sudoku_dataset
from .base import PrimaryMetric, Task
from .sudoku_metrics import SudokuMetrics


class SudokuTask(Task):
    name = "sudoku"
    primary_metric = PrimaryMetric("accuracy", "max")
    # Row separators are not end-of-sequence tokens.
    generation_eos_token_id = None

    def __init__(self, data, vocab_size=12, length=180, variable_length=False):
        self.variable_length = variable_length
        self.tokenizer = SudokuTokenizer()
        if vocab_size != len(self.tokenizer) or length != self.tokenizer.total_seq_len:
            raise ValueError("SudokuTask requires vocab_size=12 and length=180")
        self.data, self.vocab_size, self.length = data, vocab_size, length
        self._dataset = None

    def prepare_data(self):
        get_sudoku_dataset(self.data, self.tokenizer)

    def _load(self):
        if self._dataset is None:
            self._dataset = get_sudoku_dataset(self.data, self.tokenizer)
        return self._dataset

    def training_dataset(self):
        return self._load()["train"]

    def validation_dataset(self):
        return self._load()["validation"]

    def collate_training(self, examples):
        batch = default_collate(examples)
        # Keep the existing on-disk cache format; clarify its mask at the boundary.
        loss_mask = batch.pop("attention_mask").bool()
        batch.update(
            loss_mask=loss_mask,
            corruption_mask=loss_mask.clone(),
            prompt_mask=~loss_mask,
            padding_mask=torch.zeros_like(loss_mask),
        )
        return batch

    def prepare_generation(self, examples):
        tokens = torch.stack([example["input_ids"] for example in examples])
        fixed = torch.zeros_like(tokens, dtype=torch.bool)
        fixed[:, : self.tokenizer.prompt_len] = True
        return torch.where(fixed, tokens, 0), fixed

    def create_metric_accumulator(self, evaluation_name) -> SudokuMetrics:
        if evaluation_name != "validation":
            raise ValueError(f"unknown evaluation dataset: {evaluation_name}")
        return SudokuMetrics(self.tokenizer)

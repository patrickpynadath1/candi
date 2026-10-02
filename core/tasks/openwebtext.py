"""Unconditional OpenWebText task with external-reference generation metrics."""

import torch
from torch.utils.data import default_collate
from core.data.openwebtext import prepare_openwebtext
from .base import Task, PrimaryMetric
from .openwebtext_metrics import OpenWebTextMetrics


class OpenWebTextTask(Task):
    name = "openwebtext"

    def __init__(
        self,
        data,
        scorer,
        vocab_size=50257,
        length=1024,
        tokenizer_name="gpt2",
        local_files_only=False,
        num_eval_samples=128,
    ):
        if length <= 2 or num_eval_samples <= 0:
            raise ValueError(
                "length must exceed two and num_eval_samples must be positive"
            )
        self.data, self.scorer = data, scorer
        self.vocab_size, self.length = vocab_size, length
        self.tokenizer_name, self.local_files_only = tokenizer_name, local_files_only
        self.num_eval_samples = num_eval_samples
        self.primary_metric = PrimaryMetric("generative_perplexity", "min")
        self._tokenizer = None
        self._datasets = {}

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(
                self.tokenizer_name, local_files_only=self.local_files_only
            )
            if tokenizer.bos_token_id is None or tokenizer.eos_token_id is None:
                raise ValueError("OpenWebText tokenizer must define BOS and EOS")
            if tokenizer.pad_token is None:
                tokenizer.add_special_tokens({"pad_token": "[PAD]"})
            if tokenizer.vocab_size != self.vocab_size:
                raise ValueError("task.vocab_size must match base tokenizer vocabulary")
            self._tokenizer = tokenizer
        return self._tokenizer

    def _dataset(self, split):
        if split not in self._datasets:
            self._datasets[split] = prepare_openwebtext(
                self.data, self.tokenizer, self.length, split
            )
        return self._datasets[split]

    def prepare_data(self):
        for split in ("train", "validation"):
            self._dataset(split)

    def training_dataset(self):
        return self._dataset("train")

    def validation_dataset(self):
        return self._dataset("validation")

    def evaluation_dataset(self, name):
        if name != "validation":
            raise ValueError(f"unknown OpenWebText evaluation dataset: {name}")
        # Unconditional sampling does not require loading the held-out corpus.
        return range(self.num_eval_samples)

    def collate_training(self, examples):
        batch = default_collate(examples)
        active = torch.ones_like(batch["input_ids"], dtype=torch.bool)
        batch.update(
            loss_mask=active,
            corruption_mask=active.clone(),
            prompt_mask=~active,
            padding_mask=~active,
        )
        return batch

    def prepare_generation(self, examples):
        shape = (len(examples), self.length)
        return torch.zeros(shape, dtype=torch.long), torch.zeros(
            shape, dtype=torch.bool
        )

    def create_metric_accumulator(self, evaluation_name):
        self.evaluation_dataset(evaluation_name)
        return OpenWebTextMetrics(self.tokenizer, self.scorer, self.length)

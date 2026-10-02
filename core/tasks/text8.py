"""Character-level Text8 with generation-only word-count evaluation."""

from pathlib import Path
import numpy as np
import torch
from torch.utils.data import default_collate, Subset

from core.data.text8 import (
    Text8Tokenizer,
    Text8Dataset,
    raw_splits,
    prepare_split,
    training_vocabulary,
)
from .base import Task, PrimaryMetric
from .text8_metrics import Text8Metrics


class Text8Task(Task):
    name = "text8"

    def __init__(
        self,
        data,
        vocab_size=35,
        length=256,
        min_word_lengths=(3, 5, 7),
        evaluation_split="validation",
        num_eval_samples=128,
        evaluation_seed=23,
    ):
        if vocab_size != 35 or length <= 2:
            raise ValueError("Text8Task requires vocab_size=35 and length > 2")
        if evaluation_split not in {"validation", "test"}:
            raise ValueError("evaluation_split must be validation or test")
        if num_eval_samples <= 0:
            raise ValueError("num_eval_samples must be positive")
        self.num_eval_samples, self.evaluation_seed = num_eval_samples, evaluation_seed
        lengths = tuple(min_word_lengths)
        if any(
            not isinstance(n, int) or isinstance(n, bool) or n < 1 for n in lengths
        ) or len(set(lengths)) != len(lengths):
            raise ValueError("min_word_lengths must contain distinct positive integers")
        self.data, self.vocab_size, self.length = data, vocab_size, length
        self.min_word_lengths = tuple(sorted(lengths))
        self.primary_metric = PrimaryMetric("valid_word_rate", "max", evaluation_split)
        self.tokenizer = Text8Tokenizer()
        self._datasets = {}
        self._vocabulary = None
        self._paths = None

    def _sources(self):
        if self._paths is None:
            self._paths = raw_splits(self.data.raw_dir, self.data.download)
        return self._paths

    def _prepare(self, split):
        return prepare_split(
            self._sources()[split], Path(self.data.cache_dir) / "prepared", self.length
        )

    def prepare_data(self):
        for split in ("train", "validation", "test"):
            self._prepare(split)
        self._load_vocabulary()

    def _load_vocabulary(self):
        if self._vocabulary is None:
            _, raw, self._vocabulary_sha256 = self._prepare("train")
            self._vocabulary = training_vocabulary(
                raw, Path(self.data.cache_dir) / "prepared", self._vocabulary_sha256
            )
        return self._vocabulary

    def _dataset(self, split):
        if split not in self._datasets:
            path, _, _ = self._prepare(split)
            limit = self.data.get("limit_" + split)
            self._datasets[split] = Text8Dataset(path, limit)
        return self._datasets[split]

    def training_dataset(self):
        return self._dataset("train")

    def validation_dataset(self):
        return self._dataset("validation")

    def evaluation_dataset(self, name):
        if name not in {"validation", "test"}:
            raise ValueError(f"unknown Text8 evaluation dataset: {name}")
        dataset = self._dataset(name)
        indices = (
            np.random.default_rng(self.evaluation_seed)
            .choice(
                len(dataset), min(self.num_eval_samples, len(dataset)), replace=False
            )
            .tolist()
        )
        return Subset(dataset, sorted(indices))

    def collate_training(self, examples):
        batch = default_collate(examples)
        mask = torch.ones_like(batch["input_ids"], dtype=torch.bool)
        batch.update(
            loss_mask=mask,
            corruption_mask=mask.clone(),
            prompt_mask=~mask,
            padding_mask=~mask,
        )
        return batch

    def prepare_generation(self, examples):
        # Held-out examples supply comparison windows, never generation context.
        shape = (len(examples), self.length)
        return torch.zeros(shape, dtype=torch.long), torch.zeros(
            shape, dtype=torch.bool
        )

    def create_metric_accumulator(self, evaluation_name) -> Text8Metrics:
        if evaluation_name not in {"validation", "test"}:
            raise ValueError(f"unknown Text8 evaluation dataset: {evaluation_name}")
        vocabulary = self._load_vocabulary()
        return Text8Metrics(
            self.tokenizer,
            vocabulary,
            self.min_word_lengths,
            evaluation_name,
            self._vocabulary_sha256,
        )

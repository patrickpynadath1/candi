"""Prompt-conditioned short-story generation with no answer-length leakage."""

from collections import defaultdict
import numpy as np
import torch
from torch.utils.data import Subset

from core.data.tinystories import StoryByteTokenizer, prepare_stories
from .base import Task, PrimaryMetric
from .tinystories_metrics import TinyStoriesMetrics


class TinyStoriesTask(Task):
    name = "tinystories"
    # A diagnostic primary, NOT a proxy for coherent story quality.
    primary_metric = PrimaryMetric("termination_rate", "max")

    def __init__(
        self,
        data,
        vocab_size=260,
        length=1024,
        prompt_bytes=64,
        num_eval_samples=128,
        evaluation_seed=23,
        variable_length=False,
        calculate_bleu=False,
    ):
        if (
            vocab_size != 260
            or not 0 < prompt_bytes < length - 2
            or num_eval_samples < 1
        ):
            raise ValueError(
                "TinyStories needs vocab_size=260, 0 < prompt_bytes < length-2, positive eval size"
            )
        self.data, self.vocab_size, self.length = data, vocab_size, length
        self.prompt_bytes, self.num_eval_samples = prompt_bytes, num_eval_samples
        self.evaluation_seed, self.variable_length = evaluation_seed, variable_length
        self.calculate_bleu = calculate_bleu
        self.tokenizer = StoryByteTokenizer()
        self._datasets = {}

    def _dataset(self, split):
        if split not in self._datasets:
            self._datasets[split] = prepare_stories(
                self.data, split, self.length, self.prompt_bytes
            )
        return self._datasets[split]

    def prepare_data(self):
        train, validation = self._dataset("train"), self._dataset("validation")
        # Refuse source leakage rather than silently score memorized stories.
        train_ids = {r["id"].split(":", 1)[1] for r in train.records}
        if any(r["id"].split(":", 1)[1] in train_ids for r in validation.records):
            raise ValueError("TinyStories train/validation contain duplicate stories")

    def training_dataset(self):
        return self._dataset("train")

    def validation_dataset(self):
        return self._dataset("validation")

    def evaluation_dataset(self, name):
        if name != "validation":
            raise ValueError(f"unknown TinyStories evaluation set: {name}")
        data = self._dataset(name)
        ids = np.random.default_rng(self.evaluation_seed).choice(
            len(data), min(self.num_eval_samples, len(data)), replace=False
        )
        return Subset(data, sorted(ids.tolist()))

    def collate_training(self, examples):
        tokens = torch.zeros(len(examples), self.length, dtype=torch.long)
        attention = torch.zeros_like(tokens, dtype=torch.bool)
        prompt = torch.zeros_like(attention)
        for row, example in enumerate(examples):
            ids = example["input_ids"]
            tokens[row, : len(ids)] = torch.tensor(ids)
            attention[row, : len(ids)] = True
            prompt[row, : example["prompt_length"]] = True
        # Fixed-canvas PAD is learned, not exposed as clean target-length evidence.
        loss = (~prompt) & attention if self.variable_length else ~prompt
        return dict(
            input_ids=tokens,
            loss_mask=loss,
            corruption_mask=loss.clone(),
            prompt_mask=prompt,
            padding_mask=~attention,
            attention_mask=attention,
        )

    def prepare_generation(self, examples):
        tokens = torch.zeros(len(examples), self.length, dtype=torch.long)
        fixed = torch.zeros_like(tokens, dtype=torch.bool)
        for row, example in enumerate(examples):
            n = example["prompt_length"]
            tokens[row, :n] = torch.tensor(example["input_ids"][:n])
            fixed[row, :n] = True
        return tokens, fixed

    def evaluation_batches(self, dataset, indices, batch_size):
        buckets = defaultdict(list)
        for index in indices:
            buckets[dataset[index]["prompt_length"]].append(index)
        for bucket in buckets.values():
            for start in range(0, len(bucket), batch_size):
                yield bucket[start : start + batch_size]

    def create_metric_accumulator(self, evaluation_name):
        if evaluation_name != "validation":
            raise ValueError(f"unknown TinyStories evaluation set: {evaluation_name}")
        return TinyStoriesMetrics(
            self.tokenizer, self.calculate_bleu, self._dataset(evaluation_name).manifest
        )

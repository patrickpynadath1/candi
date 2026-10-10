"""TinyGSM training with independently selected GSM8K generation evaluation."""

from collections import defaultdict
import torch
from torch.utils.data import default_collate
from core.data.tinygsm import get_tinygsm_dataset
from core.data.gsm8k import get_gsm8k_dataset
from .base import Task, PrimaryMetric
from .tinygsm_metrics import TinyGSMMetrics


class TinyGSMTask(Task):
    name = "tinygsm"
    primary_metric = PrimaryMetric("accuracy", "max", "gsm8k")

    def __init__(self, data, evaluation, vocab_size=49153, length=512, tokenizer=None):
        if length < 2:
            raise ValueError("length must be at least two")
        if data.get("streaming", False):
            raise ValueError(
                "TinyGSM streaming is not yet supported by the core data module; use max_examples for a bounded subset"
            )
        self.data, self.evaluation = data, evaluation
        self.vocab_size, self.length = vocab_size, length
        self._tokenizer = tokenizer
        self._datasets = None
        self._evaluation_dataset = None

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(
                self.data.tokenizer_name_or_path,
                local_files_only=self.data.get("local_files_only", False),
            )
            if tokenizer.bos_token is None:
                if tokenizer.cls_token is None:
                    raise ValueError("tokenizer requires BOS or CLS")
                tokenizer.bos_token = tokenizer.cls_token
            if tokenizer.eos_token is None:
                if tokenizer.sep_token is None:
                    raise ValueError("tokenizer requires EOS or SEP")
                tokenizer.eos_token = tokenizer.sep_token
            if tokenizer.pad_token is None:
                tokenizer.add_special_tokens({"pad_token": "[PAD]"})
            if len(tokenizer) != self.vocab_size:
                raise ValueError(
                    f"configured vocab_size={self.vocab_size} does not match tokenizer size {len(tokenizer)}"
                )
            self._tokenizer = tokenizer
        return self._tokenizer

    def _load(self):
        if self._datasets is None:
            self._datasets = get_tinygsm_dataset(self.data, self.tokenizer, self.length)
        return self._datasets

    def prepare_data(self):
        self._load()

    def training_dataset(self):
        return self._load()["train"]

    def validation_dataset(self):
        return self._load()["validation"]

    def evaluation_dataset(self, name):
        if name != "gsm8k":
            raise ValueError(f"unknown TinyGSM evaluation dataset: {name}")
        if self._evaluation_dataset is None:
            self._evaluation_dataset = get_gsm8k_dataset(
                self.evaluation, self.tokenizer
            )
        return self._evaluation_dataset

    def collate_training(self, examples):
        batch = default_collate(examples)
        loss_mask = batch["attention_mask"].bool()
        batch.update(
            loss_mask=loss_mask,
            corruption_mask=loss_mask.clone(),
            prompt_mask=~loss_mask,
            padding_mask=batch["input_ids"] == self.tokenizer.pad_token_id,
        )
        return batch

    def prepare_generation(self, examples):
        tokens = torch.zeros(len(examples), self.length, dtype=torch.long)
        fixed = torch.zeros_like(tokens, dtype=torch.bool)
        for row, example in enumerate(examples):
            ids = torch.as_tensor(example["input_ids"][: self.length])
            tokens[row, : len(ids)] = ids
            fixed[row, : len(ids)] = True
        return tokens, fixed

    def evaluation_batches(self, dataset, indices, batch_size):
        buckets = defaultdict(list)
        for index in indices:
            buckets[len(dataset[index]["input_ids"])].append(index)
        for length in sorted(buckets):
            bucket = buckets[length]
            for start in range(0, len(bucket), batch_size):
                yield bucket[start : start + batch_size]

    def create_metric_accumulator(self, evaluation_name):
        if evaluation_name != "gsm8k":
            raise ValueError(f"unknown TinyGSM evaluation dataset: {evaluation_name}")
        return TinyGSMMetrics(
            self.tokenizer,
            self.evaluation.scoring,
            self.evaluation.timeout,
            self.evaluation.bootstrap_size,
            self.evaluation.seed,
        )

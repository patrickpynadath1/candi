"""Fixed gene-order count generation, optionally conditioned on a cell-type prefix."""

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, Subset

from .base import PrimaryMetric, Task
from .single_cell_metrics import SingleCellMetrics


class CellDataset(Dataset):
    def __init__(self, directory, split, conditional):
        self.directory, self.split, self.conditional = (
            Path(directory),
            split,
            conditional,
        )
        self._tokens = self._labels = None
        self.size = np.load(self.directory / f"{split}.npy", mmap_mode="r").shape[0]

    def __len__(self):
        return self.size

    def __getstate__(self):
        return {**self.__dict__, "_tokens": None, "_labels": None}

    def __getitem__(self, index):
        if self._tokens is None:
            self._tokens = np.load(self.directory / f"{self.split}.npy", mmap_mode="r")
            self._labels = np.load(
                self.directory / f"{self.split}_labels.npy", mmap_mode="r"
            )
        tokens = torch.tensor(self._tokens[index].copy(), dtype=torch.long)
        label = int(self._labels[index])
        if self.conditional:
            tokens = torch.cat((torch.tensor([label]), tokens))
        return {"input_ids": tokens, "cell_type": label}


class SingleCellTask(Task):
    primary_metric = PrimaryMetric("mmd", "min", dataset="test")

    def __init__(self, data, vocab_size, length, conditional=False):
        self.data, self.vocab_size, self.length = data, int(vocab_size), int(length)
        self.conditional = bool(conditional)
        self.name = (
            f"{data['dataset']}-{'conditional' if conditional else 'unconditional'}"
        )
        self.directory = Path(data["cache_dir"]) / data["dataset"]
        self._metadata = None

    def _load_metadata(self):
        if self._metadata is None:
            path = self.directory / "metadata.json"
            if not path.exists():
                raise FileNotFoundError(
                    f"Prepare single-cell data first; missing {path}"
                )
            meta = json.loads(path.read_text())
            if meta["format_version"] != 1 or meta["dataset"] != self.data["dataset"]:
                raise ValueError("Prepared dataset identity/format mismatch")
            if (
                meta["vocab_size"] != self.vocab_size
                or meta["num_genes"] + int(self.conditional) != self.length
            ):
                raise ValueError(
                    "Task shape does not match prepared metadata; rebuild the sweep plan"
                )
            if self.conditional and len(meta["labels"]) > self.vocab_size:
                raise ValueError("Cell-type labels exceed vocabulary")
            self._metadata = meta
        return self._metadata

    def prepare_data(self):
        self._load_metadata()
        for split in ("train", "validation", "test"):
            self.evaluation_dataset(split)

    def prepare_evaluation(self, name):
        self.evaluation_dataset(name)

    def training_dataset(self):
        dataset = self.evaluation_dataset("train")
        limit = self.data.get("max_train")
        return (
            Subset(dataset, range(min(len(dataset), int(limit)))) if limit else dataset
        )

    def validation_dataset(self):
        return self.evaluation_dataset("validation")

    def evaluation_dataset(self, name):
        if name not in ("train", "validation", "test"):
            raise ValueError(f"unknown evaluation dataset: {name}")
        meta = self._load_metadata()
        tokens = np.load(self.directory / f"{name}.npy", mmap_mode="r")
        labels = np.load(self.directory / f"{name}_labels.npy", mmap_mode="r")
        if (
            tokens.shape != (meta["splits"][name]["cells"], meta["num_genes"])
            or labels.shape != tokens.shape[:1]
        ):
            raise ValueError(f"Prepared {name} split shape mismatch")
        return CellDataset(self.directory, name, self.conditional)

    def collate_training(self, examples):
        tokens = torch.stack([e["input_ids"] for e in examples])
        loss = torch.ones_like(tokens, dtype=torch.bool)
        if self.conditional:
            loss[:, 0] = False
        return {
            "input_ids": tokens,
            "loss_mask": loss,
            "corruption_mask": loss.clone(),
            "prompt_mask": ~loss,
            "padding_mask": torch.zeros_like(loss),
        }

    def prepare_generation(self, examples):
        # Only the label is passed to generation; gene counts never leave the task.
        prompt = torch.zeros((len(examples), self.length), dtype=torch.long)
        fixed = torch.zeros_like(prompt, dtype=torch.bool)
        if self.conditional:
            prompt[:, 0] = torch.tensor([e["cell_type"] for e in examples])
            fixed[:, 0] = True
        return prompt, fixed

    def create_metric_accumulator(self, evaluation_name):
        if evaluation_name not in ("validation", "test"):
            raise ValueError(f"unknown evaluation dataset: {evaluation_name}")
        return SingleCellMetrics(self.vocab_size, self.conditional)

"""Supplied-map path generation; the target length is never an input."""

import torch
from core.data.maze import load_maze_dataset
from .base import Task, PrimaryMetric
from .maze_metrics import MazeMetrics


class MazeTokenizer:
    pad_token_id = 0
    eos_token_id = 1
    bos_token_id = 2
    map_token_id = 3
    anchors_token_id = 4
    answer_token_id = 5
    cell_offset = 16
    node_offset = 32


class MazeTask(Task):
    name = "maze"
    primary_metric = PrimaryMetric("success_rate", "max")

    def __init__(
        self, data, vocab_size=132, length=256, variable_length=False, eval_repeats=1
    ):
        self.data, self.length, self.vocab_size = data, length, vocab_size
        self.variable_length, self.eval_repeats = variable_length, eval_repeats
        self.tokenizer = MazeTokenizer()
        self._data = None
        if vocab_size != data["grid_size"] ** 2 + self.tokenizer.node_offset:
            raise ValueError("maze vocabulary must contain every grid node")
        prefix_length = data["grid_size"] ** 2 + data["num_waypoints"] + 6
        if length < prefix_length + data["max_path_edges"] + 2 or eval_repeats < 1:
            raise ValueError("sequence capacity too small or invalid eval_repeats")

    def prepare_data(self):
        self._load()

    def _load(self):
        if self._data is None:
            self._data = load_maze_dataset(self.data)
        return self._data

    def training_dataset(self):
        return self._load()["train"]

    def validation_dataset(self):
        return self._load()["validation"]

    def evaluation_dataset(self, name):
        if name not in {"validation", "test"}:
            raise ValueError(f"unknown maze split {name}")
        return [
            example for example in self._load()[name] for _ in range(self.eval_repeats)
        ]

    def prompt(self, example):
        t = self.tokenizer
        return (
            [t.bos_token_id, t.map_token_id]
            + [t.cell_offset + v for v in example["cells"]]
            + [t.anchors_token_id]
            + [t.node_offset + v for v in example["anchors"]]
            + [t.answer_token_id]
        )

    def collate_training(self, examples):
        tokens = torch.zeros((len(examples), self.length), dtype=torch.long)
        valid = torch.zeros_like(tokens, dtype=torch.bool)
        prompt = torch.zeros_like(valid)
        for row, example in enumerate(examples):
            prefix = self.prompt(example)
            sequence = (
                prefix
                + [self.tokenizer.node_offset + v for v in example["path"]]
                + [self.tokenizer.eos_token_id]
            )
            if len(sequence) > self.length:
                raise ValueError(
                    "maze sequence exceeds capacity; truncating would invalidate the path"
                )
            tokens[row, : len(sequence)] = torch.tensor(sequence)
            valid[row, : len(sequence)] = True
            prompt[row, : len(prefix)] = True
        # Fixed canvases must learn EOS and padding, not observe gold padding boundaries.
        loss = ~prompt & (valid if self.variable_length else torch.ones_like(valid))
        return dict(
            input_ids=tokens,
            attention_mask=valid,
            loss_mask=loss,
            corruption_mask=loss.clone(),
            prompt_mask=prompt,
            padding_mask=~valid,
        )

    def prepare_generation(self, examples):
        tokens = torch.zeros((len(examples), self.length), dtype=torch.long)
        fixed = torch.zeros_like(tokens, dtype=torch.bool)
        for row, example in enumerate(examples):
            prefix = self.prompt(example)
            tokens[row, : len(prefix)] = torch.tensor(prefix)
            fixed[row, : len(prefix)] = True
        return tokens, fixed

    def create_metric_accumulator(self, evaluation_name):
        if evaluation_name not in {"validation", "test"}:
            raise ValueError(f"unknown maze split {evaluation_name}")
        return MazeMetrics(self)

"""Published Sudoku Extreme corpus using the shared Sudoku task interface."""

import numpy as np

from core.data.sudoku_extreme import SudokuPairs, prepare_extreme
from .sudoku import SudokuTask
from .sudoku_metrics import SudokuMetrics


class SudokuExtremeTask(SudokuTask):
    name = "sudoku_extreme"

    def prepare_data(self):
        prepare_extreme(self.data)

    def _load(self):
        if self._dataset is None:
            root = prepare_extreme(self.data)
            self._dataset = {
                split: SudokuPairs(
                    np.load(root / f"{split}.npy", mmap_mode="r"),
                    self.tokenizer,
                    self.data.augmentations if split == "train" else 0,
                    self.data.data_seed,
                )
                for split in ("train", "validation", "test")
            }
        return self._dataset

    def evaluation_dataset(self, name):
        if name not in ("validation", "test"):
            raise ValueError(f"unknown Sudoku Extreme split: {name}")
        return self._load()[name]

    def create_metric_accumulator(self, evaluation_name):
        if evaluation_name not in ("validation", "test"):
            raise ValueError(f"unknown Sudoku Extreme split: {evaluation_name}")
        return SudokuMetrics(self.tokenizer)

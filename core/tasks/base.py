"""Task semantics, independent of models, samplers, and Lightning."""

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class PrimaryMetric:
    name: str
    mode: str
    dataset: str = "validation"

    def __post_init__(self):
        if self.mode not in {"min", "max"}:
            raise ValueError("metric mode must be min or max")


class Task(ABC):
    @abstractmethod
    def prepare_data(self): ...

    @abstractmethod
    def training_dataset(self): ...

    @abstractmethod
    def validation_dataset(self): ...

    def evaluation_dataset(self, name):
        if name != "validation":
            raise ValueError(f"unknown evaluation dataset: {name}")
        return self.validation_dataset()

    def prepare_evaluation(self, name):
        """Prepare only resources needed for evaluation, never training by default.

        Tasks with lazy evaluation datasets/scorers need no additional setup.
        """
        pass

    @abstractmethod
    def collate_training(self, examples): ...

    @abstractmethod
    def prepare_generation(self, examples):
        """Return prompt tokens and fixed-position mask, without answer payloads."""
        ...

    def evaluation_batches(self, dataset, indices, batch_size):
        for offset in range(0, len(indices), batch_size):
            yield indices[offset : offset + batch_size]

    @abstractmethod
    def create_metric_accumulator(self, evaluation_name):
        """Return an accumulator whose compute() produces a task aggregate dataclass."""
        ...

"""Typed job definitions; Ray Tune owns their execution and scheduling."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from omegaconf import DictConfig


@dataclass(frozen=True)
class InferenceJob:
    """Sampler/evaluation settings applied to a parent's saved model and weights."""

    sampler: str | None = None
    overrides: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        for key in self.overrides:
            if not (
                key.startswith("sampler.")
                or key in {"seed", "loader.eval_batch_size", "trainer.precision"}
                or key
                in {
                    "evaluation.max_examples",
                    "evaluation.use_ema",
                    "evaluation.policy",
                    "evaluator.evaluation_name",
                    "task.evaluation_mode",
                    "task.samples_per_prompt",
                    "task.eval_repeats",
                    "task.calculate_bleu",
                }
                or key in {"wandb.name", "wandb.job_type"}
            ):
                raise ValueError(
                    f"Inference cannot override training/model field {key!r}"
                )


@dataclass(frozen=True)
class TrainingJob:
    """A concrete training configuration and the inference jobs it will produce."""

    name: str
    experiment: str | None = None
    overrides: Mapping[str, Any] | Sequence[str] = field(default_factory=dict)
    config: DictConfig | Mapping[str, Any] | None = None
    inference: Sequence[InferenceJob] = ()
    checkpoint: str = "last.ckpt"

    def __post_init__(self):
        if not self.name:
            raise ValueError("TrainingJob requires a nonempty name")
        if self.config is not None and (self.experiment is not None or self.overrides):
            raise ValueError("Use either config or experiment/overrides, not both")
        is_filename = Path(self.checkpoint).name == self.checkpoint
        if not is_filename or not self.checkpoint.endswith(".ckpt"):
            raise ValueError("checkpoint must be a filename such as last.ckpt")

    def to_trial_spec(self) -> dict[str, Any]:
        """Convert to the existing Hydra/Tune adapter's input at the boundary."""
        if self.config is not None:
            return {"config": self.config}
        return {"experiment": self.experiment, "overrides": self.overrides}

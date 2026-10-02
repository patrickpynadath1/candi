import json
from pathlib import Path
import numpy as np
import torch
import lightning as L

from core.evaluation.generation import evaluate
from core.evaluation.reporting import scalar_metrics


class TaskEvaluation(L.Callback):
    def __init__(
        self,
        task,
        every_n_steps,
        subset_size,
        seed,
        sampler_config,
        batch_size,
        precision,
        output_dir,
    ):
        self.task = task
        self.every_n_steps, self.subset_size, self.seed = (
            every_n_steps,
            subset_size,
            seed,
        )
        self.sampler_config, self.batch_size = sampler_config, batch_size
        self.precision, self.output_dir = precision, Path(output_dir)
        self.last_step = -1

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        step = trainer.global_step
        every = self.every_n_steps
        if not every or step == 0 or step == self.last_step or step % every:
            return
        self.last_step = step
        if trainer.is_global_zero:
            self.task.prepare_evaluation(self.task.primary_metric.dataset)
            dataset = self.task.evaluation_dataset(self.task.primary_metric.dataset)
            rng = np.random.default_rng(self.seed)
            indices = sorted(
                rng.choice(
                    len(dataset),
                    min(len(dataset), self.subset_size),
                    replace=False,
                ).tolist()
            )
            # Diagnostics must not change subsequent training corruption/dropout.
            devices = (
                [pl_module.device.index] if pl_module.device.type == "cuda" else []
            )
            with torch.random.fork_rng(devices=devices):
                torch.manual_seed(self.seed)
                result = evaluate(
                    pl_module,
                    self.task,
                    self.sampler_config,
                    batch_size=self.batch_size,
                    indices=indices,
                    precision=self.precision,
                )
            self.output_dir.mkdir(parents=True, exist_ok=True)
            (self.output_dir / f"step_{step:08d}.json").write_text(
                json.dumps(result, indent=2)
            )
            if trainer.logger:
                trainer.logger.log_metrics(
                    {
                        f"{self.task.name}/{key}": value
                        for key, value in scalar_metrics(result)
                    },
                    step=step,
                )
        trainer.strategy.barrier("task-evaluation")

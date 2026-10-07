import json
from pathlib import Path
import numpy as np
import torch
import lightning as L
from lightning.pytorch.loggers import WandbLogger

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
        every_n_epochs=0,
        evaluation_name=None,
        save_records=True,
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
        self.every_n_epochs = every_n_epochs
        self.evaluation_name = evaluation_name
        self.save_records = save_records

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        step = trainer.global_step
        every = self.every_n_steps
        if not every or step == 0 or step == self.last_step or step % every:
            return
        self._evaluate(trainer, pl_module)

    def on_train_epoch_end(self, trainer, pl_module):
        if (
            self.every_n_epochs
            and (trainer.current_epoch + 1) % self.every_n_epochs == 0
        ):
            self._evaluate(trainer, pl_module)

    def _evaluate(self, trainer, pl_module):
        step = trainer.global_step
        self.last_step = step
        if trainer.is_global_zero:
            evaluation_name = self.evaluation_name or self.task.primary_metric.dataset
            self.task.prepare_evaluation(evaluation_name)
            dataset = self.task.evaluation_dataset(evaluation_name)
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
                    evaluation_name=evaluation_name,
                )
            result["evaluation_split"] = evaluation_name
            result["epoch"] = trainer.current_epoch + 1
            if not self.save_records:
                result.pop("records", None)
            self.output_dir.mkdir(parents=True, exist_ok=True)
            result_path = self.output_dir / f"step_{step:08d}.json"
            result_path.write_text(json.dumps(result, indent=2))
            for logger in trainer.loggers:
                logger.log_metrics(
                    {
                        f"{self.task.name}/{key}": value
                        for key, value in scalar_metrics(result)
                    },
                    step=step,
                )
                if isinstance(logger, WandbLogger):
                    import wandb

                    records = result.get("records", [])
                    if records:
                        columns = list(
                            dict.fromkeys(key for row in records for key in row)
                        )
                        data = [
                            [
                                json.dumps(row.get(key))
                                if isinstance(row.get(key), (dict, list))
                                else row.get(key)
                                for key in columns
                            ]
                            for row in records
                        ]
                        logger.log_table(
                            key=f"{self.task.name}/samples",
                            columns=columns,
                            data=data,
                            step=step,
                        )
                    artifact = wandb.Artifact(
                        f"{self.task.name}-generation",
                        type="evaluation",
                        metadata={"step": step, "task": self.task.name},
                    )
                    artifact.add_file(str(result_path))
                    logger.experiment.log_artifact(artifact)
        trainer.strategy.barrier("task-evaluation")

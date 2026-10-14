import json
from pathlib import Path
import numpy as np
import torch
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from omegaconf import OmegaConf

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
        sampler_configs=None,
    ):
        self.task = task
        self.every_n_steps, self.subset_size, self.seed = (
            every_n_steps,
            subset_size,
            seed,
        )
        self.sampler_config, self.batch_size = sampler_config, batch_size
        self.precision, self.output_dir = precision, Path(output_dir)
        self.sampler_configs = sampler_configs
        self.last_step = -1

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        step = trainer.global_step
        every = self.every_n_steps
        if not every or step == 0 or step == self.last_step or step % every:
            return
        self.last_step = step
        samplers = self.sampler_configs or {None: self.sampler_config}
        for policy, sampler_config in samplers.items():
            self._evaluate(trainer, pl_module, step, policy, sampler_config)
        trainer.strategy.barrier("task-evaluation")

    def _evaluate(self, trainer, pl_module, step, policy, sampler_config):
        namespace = f"{self.task.name}/{policy}" if policy else self.task.name
        output_dir = self.output_dir / policy if policy else self.output_dir
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
                    sampler_config,
                    batch_size=self.batch_size,
                    indices=indices,
                    precision=self.precision,
                )
            config = getattr(pl_module, "config", {})
            if config.get("benchmark") is not None:
                result["benchmark"] = OmegaConf.to_container(config.benchmark, resolve=True)
            output_dir.mkdir(parents=True, exist_ok=True)
            result_path = output_dir / f"step_{step:08d}.json"
            result_path.write_text(json.dumps(result, indent=2))
            for logger in trainer.loggers:
                logger.log_metrics(
                    {
                        f"{namespace}/{key}": value
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
                            key=f"{namespace}/samples",
                            columns=columns,
                            data=data,
                            step=step,
                        )
                    artifact = wandb.Artifact(
                        f"{self.task.name}-{policy}-generation"
                        if policy
                        else f"{self.task.name}-generation",
                        type="evaluation",
                        metadata={"step": step, "task": self.task.name},
                    )
                    artifact.add_file(str(result_path))
                    logger.experiment.log_artifact(artifact)

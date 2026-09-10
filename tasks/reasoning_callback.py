"""Periodic generation monitoring, independent of validation/model selection."""

import json
import random
from pathlib import Path

import lightning
import numpy as np
import torch

from .reasoning import load_prepared, generate_reasoning_metrics
from .sudoku import get_sudoku_dataset
from .sudoku_callback import generate_sudoku_records, summarize_sudoku_records


class ReasoningEvaluationCallback(lightning.Callback):
    def __init__(
        self,
        every_n_train_steps=1000,
        subset_size=128,
        batch_size=4,
        num_steps=45,
        seed=42,
        output_dir="reasoning_callback",
    ):
        if min(every_n_train_steps, subset_size, batch_size, num_steps) <= 0:
            raise ValueError(
                "interval, subset size, batch size, and sampling steps must be positive"
            )
        self.every_n_train_steps = every_n_train_steps
        self.subset_size = subset_size
        self.batch_size = batch_size
        self.num_steps = num_steps
        self.seed = seed
        self.output_dir = Path(output_dir)
        self._dataset = None
        self._last_step = -1

    @property
    def state_key(self):
        return self._generate_state_key(
            every_n_train_steps=self.every_n_train_steps,
            subset_size=self.subset_size,
            seed=self.seed,
            num_steps=self.num_steps,
        )

    def state_dict(self):
        return {"last_step": self._last_step}

    def load_state_dict(self, state_dict):
        self._last_step = state_dict["last_step"]

    def _setup_dataset(self, model):
        if self._dataset is not None:
            return
        self.task = model.config.data.train
        if self.task == "sudoku":
            self.split = "validation"
            dataset = get_sudoku_dataset(model.config, model.tokenizer)[self.split]
            source = {
                k: model.config.data.get(k)
                for k in ("difficulty", "data_seed", "num_train", "num_valid")
            }
        else:
            manifest = json.loads(
                (Path(model.config.data.prepared_dir) / "manifest.json").read_text()
            )
            # Keep the full released small-data training set. Test monitoring is never model selection.
            self.split = "test" if manifest["full_training"] else "validation"
            dataset = load_prepared(model.config, model.tokenizer, self.split)
            source = {"dataset_fingerprint": manifest["fingerprint"]}
        if not len(dataset):
            raise ValueError("empty monitoring dataset")
        indices = sorted(
            np.random.default_rng(self.seed)
            .choice(len(dataset), min(self.subset_size, len(dataset)), replace=False)
            .tolist()
        )
        self._dataset = dataset.select(indices)
        self._provenance = dict(
            task=self.task,
            split=self.split,
            monitoring_only=True,
            checkpoint_selection=False,
            seed=self.seed,
            indices=indices,
            source_ids=list(self._dataset["source_id"])
            if "source_id" in dataset.column_names
            else indices,
            requested_nfe=self.num_steps,
            batch_size=self.batch_size,
            **source,
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)
        destination = self.output_dir / "subset.json"
        if (
            destination.exists()
            and json.loads(destination.read_text()) != self._provenance
        ):
            raise ValueError(f"monitoring subset/settings changed: {destination}")
        destination.write_text(json.dumps(self._provenance, indent=2))

    @torch.no_grad()
    def _run(self, trainer, model):
        self._setup_dataset(model)
        device = model.device
        devices = (
            [device.index if device.index is not None else torch.cuda.current_device()]
            if device.type == "cuda"
            else []
        )
        python_state, numpy_state = random.getstate(), np.random.get_state()
        try:
            # Monitoring must not change subsequent training randomness or adaptive updates.
            with torch.random.fork_rng(devices=devices):
                torch.random.default_generator.manual_seed(self.seed)
                for index in devices:
                    torch.cuda.default_generators[index].manual_seed(self.seed)
                random.seed(self.seed)
                np.random.seed(self.seed)
                model._eval_mode()
                try:
                    if self.task == "sudoku":
                        rows = generate_sudoku_records(
                            model,
                            self._dataset,
                            model.tokenizer,
                            self.batch_size,
                            self.num_steps,
                            range(len(self._dataset)),
                        )
                        results = summarize_sudoku_records(rows)
                    else:
                        results = generate_reasoning_metrics(
                            model,
                            self._dataset,
                            model.tokenizer,
                            self.batch_size,
                            self.num_steps,
                            self.task,
                            self.split,
                        )
                        results["solve_rate"] = results["exact"]
                finally:
                    model._train_mode()
        finally:
            random.setstate(python_state)
            np.random.set_state(numpy_state)
        step = int(trainer.global_step)
        results.update(
            step=step,
            monitoring_only=True,
            checkpoint_selection=False,
            task=self.task,
            split=self.split,
            requested_nfe=self.num_steps,
        )
        destination = self.output_dir / f"step_{step:08d}.json"
        destination.write_text(json.dumps(results, indent=2))
        scalars = {
            f"monitor/{self.task}/{self.split}/{k}": v
            for k, v in results.items()
            if isinstance(v, (float, int))
            and k not in ("step", "monitoring_only", "checkpoint_selection")
        }
        if trainer.logger is not None:
            trainer.logger.log_metrics(scalars, step=step)
        print(
            f"Generation monitor step={step} task={self.task} split={self.split} "
            f"solve_rate={results['solve_rate']:.4f} results={destination}",
            flush=True,
        )

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        step = int(trainer.global_step)
        if step <= 0 or step <= self._last_step or step % self.every_n_train_steps:
            return
        if trainer.is_global_zero:
            self._run(trainer, pl_module)
        trainer.strategy.barrier("reasoning-generation-monitor")
        self._last_step = step

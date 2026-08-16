"""Periodic conditional GSM8K generation and sandboxed code evaluation."""

import json
from pathlib import Path

import lightning
import numpy as np

from .evaluation import generate_gsm8k_records, score_gsm8k_records
from .gsm8k import get_gsm8k_dataset


class GSM8KEvaluationCallback(lightning.Callback):
    def __init__(
        self,
        every_n_train_steps,
        subset_size=128,
        batch_size=16,
        num_steps=32,
        timeout=3.0,
        seed=42,
        log_examples=16,
        output_dir="gsm8k_callback",
        full_eval_at_end=True,
    ):
        if every_n_train_steps <= 0:
            raise ValueError("every_n_train_steps must be positive")
        if subset_size <= 0 or batch_size <= 0:
            raise ValueError("subset_size and batch_size must be positive")
        self.every_n_train_steps = every_n_train_steps
        self.subset_size = subset_size
        self.batch_size = batch_size
        self.num_steps = num_steps
        self.timeout = timeout
        self.seed = seed
        self.log_examples = log_examples
        self.output_dir = output_dir
        self.full_eval_at_end = full_eval_at_end
        self._dataset = None
        self._subset_indices = None
        self._last_step = -1

    def _setup_dataset(self, pl_module):
        if self._dataset is None:
            self._dataset = get_gsm8k_dataset(pl_module.config, pl_module.tokenizer)
            rng = np.random.default_rng(self.seed)
            count = min(self.subset_size, len(self._dataset))
            self._subset_indices = sorted(
                rng.choice(len(self._dataset), size=count, replace=False).tolist()
            )

    def _save(self, step, results, full):
        directory = Path(self.output_dir)
        directory.mkdir(parents=True, exist_ok=True)
        suffix = "full" if full else "subset"
        destination = directory / f"step_{step:08d}_{suffix}.json"
        destination.write_text(json.dumps(results, indent=2))
        return destination

    def _log(self, trainer, step, results, full):
        prefix = "gsm8k/full" if full else "gsm8k/subset"
        scalar_keys = (
            "accuracy",
            "num_correct",
            "num_total",
            "parsed_rate",
            "executed_rate",
            "timed_out_rate",
            "sandbox_error_rate",
            "eos_rate",
            "mean_completion_tokens",
            "truncation_rate",
        )
        metrics = {
            f"{prefix}/{key}": results[key] for key in scalar_keys if key in results
        }
        if trainer.logger is not None:
            trainer.logger.log_metrics(metrics, step=step)
            if hasattr(trainer.logger, "log_table") and self.log_examples > 0:
                rows = results["records"][: self.log_examples]
                trainer.logger.log_table(
                    key=f"{prefix}/examples_step_{step}",
                    columns=["prompt", "response", "gold", "correct"],
                    data=[
                        [
                            row["prompt"],
                            row["response"],
                            row["response_ground_truth"],
                            row["correct"],
                        ]
                        for row in rows
                    ],
                )

    def _run(self, trainer, pl_module, full=False):
        self._setup_dataset(pl_module)
        indices = range(len(self._dataset)) if full else self._subset_indices
        pl_module._eval_mode()
        try:
            records = generate_gsm8k_records(
                pl_module,
                self._dataset,
                pl_module.tokenizer,
                self.batch_size,
                self.num_steps,
                indices,
            )
            results = score_gsm8k_records(
                records,
                method="code",
                timeout=self.timeout,
                seed=self.seed,
            )
        finally:
            pl_module._train_mode()
        step = trainer.global_step
        self._save(step, results, full)
        self._log(trainer, step, results, full)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        del outputs, batch, batch_idx
        step = trainer.global_step
        should_run = (
            step > 0
            and step != self._last_step
            and step % self.every_n_train_steps == 0
        )
        if should_run:
            self._last_step = step
            if trainer.is_global_zero:
                self._run(trainer, pl_module, full=False)
            trainer.strategy.barrier("gsm8k-subset-evaluation")

    def on_train_end(self, trainer, pl_module):
        if self.full_eval_at_end:
            if trainer.is_global_zero:
                self._run(trainer, pl_module, full=True)
            trainer.strategy.barrier("gsm8k-full-evaluation")

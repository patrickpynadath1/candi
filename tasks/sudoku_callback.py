"""Periodic conditional Sudoku evaluation during training."""

import json
from pathlib import Path

import lightning
import numpy as np
import torch

from .sudoku import get_sudoku_dataset, sudoku_metrics


@torch.no_grad()
def generate_sudoku_records(
    model, dataset, tokenizer, batch_size, num_steps, indices,
    sampler_name=None, sampler_overrides=None,
):
    records = []
    indices = list(indices)
    prompt_length = tokenizer.prompt_len
    for start in range(0, len(indices), batch_size):
        batch_indices = indices[start : start + batch_size]
        batch = torch.stack([dataset[index]["input_ids"] for index in batch_indices])
        count = batch.shape[0]
        prompt_tokens = torch.zeros(
            count, model.num_tokens, dtype=torch.long, device=model.device
        )
        prompt_mask = torch.zeros_like(prompt_tokens, dtype=torch.bool)
        prompt_tokens[:, :prompt_length] = batch[:, :prompt_length].to(model.device)
        prompt_mask[:, :prompt_length] = True
        sampling_result = model.generate_samples(
            count,
            num_steps=num_steps,
            prompt_tokens=prompt_tokens,
            prompt_mask=prompt_mask,
            sampler_name=sampler_name,
            sampler_overrides=sampler_overrides,
            return_result=True,
        )
        generated = sampling_result.samples.cpu()
        per_sample_nfe = sampling_result.metadata.get("nfe_per_sample")
        shared_nfe = sampling_result.metadata.get("nfe")
        truth = batch[:, prompt_length:]
        predictions = generated[:, prompt_length : prompt_length + truth.shape[1]]
        for row, (index, prediction, target) in enumerate(
            zip(batch_indices, predictions, truth)
        ):
            record = {
                    "index": index,
                    "generated": tokenizer.decode(prediction),
                    "ground_truth": tokenizer.decode(target),
                    **sudoku_metrics(prediction, target),
            }
            if per_sample_nfe is not None:
                record["nfe"] = float(per_sample_nfe[row])
            elif shared_nfe is not None:
                record["nfe"] = float(shared_nfe)
            records.append(record)
    return records


def summarize_sudoku_records(records):
    total = len(records)
    denominator = max(total, 1)
    nfes = [record["nfe"] for record in records if "nfe" in record]
    return {
        "solve_rate": sum(record["exact"] for record in records) / denominator,
        "cell_accuracy": sum(record["cell_accuracy"] for record in records)
        / denominator,
        "valid_grid_rate": sum(record["valid_grid"] for record in records)
        / denominator,
        "valid_row_rate": sum(record["valid_rows"] for record in records)
        / denominator,
        "valid_column_rate": sum(record["valid_columns"] for record in records)
        / denominator,
        "valid_box_rate": sum(record["valid_boxes"] for record in records)
        / denominator,
        "num_correct": sum(record["exact"] for record in records),
        "num_total": total,
        "mean_nfe": sum(nfes) / len(nfes) if nfes else None,
        "min_nfe": min(nfes) if nfes else None,
        "max_nfe": max(nfes) if nfes else None,
        "records": records,
    }


class SudokuEvaluationCallback(lightning.Callback):
    def __init__(
        self,
        every_n_train_steps,
        subset_size=128,
        batch_size=64,
        num_steps=45,
        seed=42,
        log_examples=8,
        output_dir="sudoku_callback",
        full_eval_at_end=False,
        samplers=None,
    ):
        if every_n_train_steps <= 0:
            raise ValueError("every_n_train_steps must be positive")
        if subset_size <= 0 or batch_size <= 0 or num_steps <= 0:
            raise ValueError("subset_size, batch_size, and num_steps must be positive")
        self.every_n_train_steps = every_n_train_steps
        self.subset_size = subset_size
        self.batch_size = batch_size
        self.num_steps = num_steps
        self.seed = seed
        self.log_examples = log_examples
        self.output_dir = output_dir
        self.full_eval_at_end = full_eval_at_end
        self.samplers = self._normalize_samplers(samplers)
        self._dataset = None
        self._subset_indices = None
        self._last_step = -1

    def _normalize_samplers(self, samplers):
        if not samplers:
            return [{"name": "default", "predictor": None,
                     "num_steps": self.num_steps, "overrides": {}}]
        normalized = []
        seen = set()
        for entry in samplers:
            entry = dict(entry)
            name = str(entry.pop("name"))
            if name in seen:
                raise ValueError(f"duplicate Sudoku callback sampler name: {name}")
            seen.add(name)
            predictor = entry.pop("predictor", None)
            num_steps = int(entry.pop("num_steps", self.num_steps))
            overrides = dict(entry.pop("overrides", {}))
            overrides.update(entry)
            if num_steps <= 0:
                raise ValueError("sampler num_steps must be positive")
            normalized.append({
                "name": name,
                "predictor": predictor,
                "num_steps": num_steps,
                "overrides": overrides,
            })
        return normalized

    def _setup_dataset(self, pl_module):
        if self._dataset is None:
            self._dataset = get_sudoku_dataset(
                pl_module.config, pl_module.tokenizer
            )["validation"]
            rng = np.random.default_rng(self.seed)
            count = min(self.subset_size, len(self._dataset))
            self._subset_indices = sorted(
                rng.choice(len(self._dataset), size=count, replace=False).tolist()
            )

    def _save(self, step, results, full, sampler_name):
        directory = Path(self.output_dir)
        directory.mkdir(parents=True, exist_ok=True)
        suffix = "full" if full else "subset"
        sampler_suffix = "" if sampler_name == "default" else f"_{sampler_name}"
        destination = directory / f"step_{step:08d}_{suffix}{sampler_suffix}.json"
        destination.write_text(json.dumps(results, indent=2))
        return destination

    def _log(self, trainer, step, results, full, sampler_name):
        prefix = "sudoku/full" if full else "sudoku/subset"
        if sampler_name != "default":
            prefix = f"{prefix}/{sampler_name}"
        scalar_keys = (
            "solve_rate",
            "cell_accuracy",
            "valid_grid_rate",
            "valid_row_rate",
            "valid_column_rate",
            "valid_box_rate",
            "num_correct",
            "num_total",
            "mean_nfe",
            "min_nfe",
            "max_nfe",
        )
        metrics = {
            f"{prefix}/{key}": results[key]
            for key in scalar_keys
            if results[key] is not None
        }
        if trainer.logger is not None:
            trainer.logger.log_metrics(metrics, step=step)
            if hasattr(trainer.logger, "log_table") and self.log_examples > 0:
                rows = results["records"][: self.log_examples]
                trainer.logger.log_table(
                    key=f"{prefix}/examples_step_{step}",
                    columns=["index", "generated", "ground_truth", "exact"],
                    data=[
                        [
                            row["index"],
                            row["generated"],
                            row["ground_truth"],
                            row["exact"],
                        ]
                        for row in rows
                    ],
                )

    def _run(self, trainer, pl_module, full=False):
        self._setup_dataset(pl_module)
        indices = range(len(self._dataset)) if full else self._subset_indices
        pl_module._eval_mode()
        try:
            all_results = []
            for sampler in self.samplers:
                records = generate_sudoku_records(
                    pl_module,
                    self._dataset,
                    pl_module.tokenizer,
                    self.batch_size,
                    sampler["num_steps"],
                    indices,
                    sampler_name=sampler["predictor"],
                    sampler_overrides=sampler["overrides"],
                )
                all_results.append(
                    (sampler["name"], summarize_sudoku_records(records))
                )
        finally:
            pl_module._train_mode()
        step = trainer.global_step
        for sampler_name, results in all_results:
            self._save(step, results, full, sampler_name)
            self._log(trainer, step, results, full, sampler_name)

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
            trainer.strategy.barrier("sudoku-subset-evaluation")

    def on_train_end(self, trainer, pl_module):
        if self.full_eval_at_end:
            if trainer.is_global_zero:
                self._run(trainer, pl_module, full=True)
            trainer.strategy.barrier("sudoku-full-evaluation")

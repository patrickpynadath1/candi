"""Periodic multi-sampler generation for Text8 training runs."""

import json
from pathlib import Path

import lightning
import torch


def _sample_entropy(samples):
    values = []
    for sample in samples:
        counts = torch.unique(sample, return_counts=True)[1].float()
        probabilities = counts / counts.sum()
        values.append(torch.special.entr(probabilities).sum().item())
    return sum(values) / max(len(values), 1)


class Text8SamplingCallback(lightning.Callback):
    def __init__(
        self,
        every_n_train_steps,
        num_samples=128,
        batch_size=128,
        num_steps=128,
        log_examples=8,
        output_dir="text8_callback",
        samplers=None,
    ):
        if every_n_train_steps <= 0:
            raise ValueError("every_n_train_steps must be positive")
        if num_samples <= 0 or batch_size <= 0 or num_steps <= 0:
            raise ValueError("sample counts and num_steps must be positive")
        self.every_n_train_steps = int(every_n_train_steps)
        self.num_samples = int(num_samples)
        self.batch_size = int(batch_size)
        self.num_steps = int(num_steps)
        self.log_examples = int(log_examples)
        self.output_dir = output_dir
        self.samplers = self._normalize_samplers(samplers)
        self._last_step = -1

    def _normalize_samplers(self, samplers):
        if not samplers:
            return [{"name": "default", "predictor": None,
                     "temperature": None, "overrides": {}}]
        normalized = []
        seen = set()
        for raw in samplers:
            entry = dict(raw)
            name = str(entry.pop("name"))
            if name in seen:
                raise ValueError(f"duplicate Text8 sampler name: {name}")
            seen.add(name)
            normalized.append({
                "name": name,
                "predictor": entry.pop("predictor", None),
                "temperature": entry.pop("temperature", None),
                "overrides": entry,
            })
        return normalized

    @torch.no_grad()
    def _generate(self, model, sampler):
        batches = []
        nfes = []
        original_temperature = getattr(model, "temp", None)
        if sampler["temperature"] is not None:
            model.temp = float(sampler["temperature"])
        try:
            for start in range(0, self.num_samples, self.batch_size):
                count = min(self.batch_size, self.num_samples - start)
                result = model.generate_samples(
                    count,
                    num_steps=self.num_steps,
                    sampler_name=sampler["predictor"],
                    sampler_overrides=sampler["overrides"],
                    return_result=True,
                )
                batches.append(result.samples.cpu())
                per_sample = result.metadata.get("nfe_per_sample")
                if per_sample is not None:
                    nfes.extend(float(value) for value in per_sample)
                elif result.metadata.get("nfe") is not None:
                    nfes.extend([float(result.metadata["nfe"])] * count)
        finally:
            if original_temperature is not None:
                model.temp = original_temperature
        samples = torch.cat(batches)
        decoded = model.tokenizer.batch_decode(samples)
        return {
            "num_samples": len(samples),
            "temperature": sampler["temperature"],
            "sample_entropy": _sample_entropy(samples),
            "mean_nfe": sum(nfes) / len(nfes) if nfes else None,
            "samples": decoded,
        }

    def _run(self, trainer, model):
        model._eval_mode()
        try:
            results = [(sampler["name"], self._generate(model, sampler))
                       for sampler in self.samplers]
        finally:
            model._train_mode()
        step = trainer.global_step
        directory = Path(self.output_dir)
        directory.mkdir(parents=True, exist_ok=True)
        for name, result in results:
            (directory / f"step_{step:08d}_{name}.json").write_text(
                json.dumps(result, indent=2)
            )
            metrics = {
                f"text8/sampling/{name}/sample_entropy": result["sample_entropy"]
            }
            if result["mean_nfe"] is not None:
                metrics[f"text8/sampling/{name}/mean_nfe"] = result["mean_nfe"]
            if trainer.logger is not None:
                trainer.logger.log_metrics(metrics, step=step)
                if hasattr(trainer.logger, "log_table") and self.log_examples > 0:
                    trainer.logger.log_table(
                        key=f"text8/sampling/{name}/examples_step_{step}",
                        columns=["Generated sample"],
                        data=[[value] for value in result["samples"][:self.log_examples]],
                    )

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        del outputs, batch, batch_idx
        step = trainer.global_step
        if (step > 0 and step != self._last_step
                and step % self.every_n_train_steps == 0):
            self._last_step = step
            if trainer.is_global_zero:
                self._run(trainer, pl_module)
            trainer.strategy.barrier("text8-sampling-evaluation")

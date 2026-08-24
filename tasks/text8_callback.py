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
        at_validation_end=False,
        seed=23,
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
        self.at_validation_end = bool(at_validation_end)
        self.seed = int(seed)
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
        accepted_total = 0
        rejected_total = 0
        accepted_prefix_total = 0
        verified_advance_total = 0
        verification_sequences = 0
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
                accepted_total += int(result.metadata.get("accepted", 0))
                rejected_total += int(result.metadata.get("rejected", 0))
                accepted_prefix_total += int(
                    result.metadata.get("accepted_prefix_total", 0)
                )
                verified_advance_total += int(
                    result.metadata.get("verified_advance_total", 0)
                )
                verification_sequences += int(
                    result.metadata.get("verification_sequences", 0)
                )
        finally:
            if original_temperature is not None:
                model.temp = original_temperature
        samples = torch.cat(batches)
        decoded = model.tokenizer.batch_decode(samples)
        proposal_total = accepted_total + rejected_total
        return {
            "num_samples": len(samples),
            "temperature": sampler["temperature"],
            "sample_entropy": _sample_entropy(samples),
            "mean_nfe": sum(nfes) / len(nfes) if nfes else None,
            "accepted": accepted_total,
            "rejected": rejected_total,
            "acceptance_rate": (
                accepted_total / proposal_total if proposal_total else None
            ),
            "mean_accepted_length": (
                accepted_prefix_total / verification_sequences
                if verification_sequences
                else None
            ),
            "mean_verified_advance": (
                verified_advance_total / verification_sequences
                if verification_sequences
                else None
            ),
            "samples": decoded,
        }

    def _run(self, trainer, model):
        step = trainer.global_step
        cuda_devices = []
        if model.device.type == "cuda":
            cuda_devices = [model.device.index or 0]
        # Evaluation must not perturb training corruption/dropout RNG. Using a
        # step-derived seed also gives matched proposals across ablation runs.
        with torch.random.fork_rng(devices=cuda_devices):
            torch.manual_seed(self.seed + step)
            model._eval_mode()
            try:
                results = [(sampler["name"], self._generate(model, sampler))
                           for sampler in self.samplers]
            finally:
                model._train_mode()
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
            for metric in (
                "acceptance_rate",
                "mean_accepted_length",
                "mean_verified_advance",
            ):
                if result[metric] is not None:
                    metrics[f"text8/sampling/{name}/{metric}"] = result[metric]
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
        if self.at_validation_end:
            return
        step = trainer.global_step
        if (step > 0 and step != self._last_step
                and step % self.every_n_train_steps == 0):
            self._last_step = step
            if trainer.is_global_zero:
                self._run(trainer, pl_module)
            trainer.strategy.barrier("text8-sampling-evaluation")

    def on_validation_end(self, trainer, pl_module):
        if not self.at_validation_end or trainer.sanity_checking:
            return
        step = trainer.global_step
        if step <= 0 or step == self._last_step:
            return
        self._last_step = step
        if trainer.is_global_zero:
            self._run(trainer, pl_module)
        trainer.strategy.barrier("text8-validation-sampling-evaluation")

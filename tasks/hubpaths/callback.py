"""Lightning evaluation hook using the repository's conditional sampler API."""

import json
from pathlib import Path
import lightning
import numpy as np
import torch
from .data import HubPathsDataset, hints_from_row
from .evaluation import evaluate_samples
from .graph import Graph


class HubPathsEvaluationCallback(lightning.Callback):
    def __init__(
        self,
        every_n_train_steps=1000,
        subset_size=16,
        samples_per_prompt=16,
        num_steps=32,
        output_dir="hubpaths_eval",
        full_eval_at_end=True,
    ):
        if min(every_n_train_steps, subset_size, samples_per_prompt, num_steps) <= 0:
            raise ValueError("evaluation counts must be positive")
        self.every_n_train_steps = every_n_train_steps
        self.subset_size, self.samples_per_prompt = subset_size, samples_per_prompt
        self.num_steps, self.output_dir = num_steps, Path(output_dir)
        self.full_eval_at_end = full_eval_at_end
        self.last_step = -1

    @torch.no_grad()
    def _run(self, trainer, model):
        root = Path(model.config.data.prepared_dir)
        metadata = json.loads((root / "metadata.json").read_text())
        if metadata.get("task_mode") in {"latent_core", "grammar_core"}:
            raise ValueError(
                "conditional waypoint evaluation cannot use latent-core sidecars as prompts; use hubpaths_latent_ar"
            )
        graph = Graph.load(root / "graph.npz")
        view = "ar" if model.config.algo.name == "ar" else "infilling"
        model._eval_mode()
        try:
            for split in ("test", "test_ood"):
                dataset = HubPathsDataset(root, split, view)
                predictions, prompts, hints, nlls = [], [], [], []
                for index in range(min(self.subset_size, len(dataset))):
                    item = dataset[index]
                    truth = item["input_ids"].to(model.device)[None]
                    loss_mask = item["attention_mask"].to(model.device)[None]
                    prompt_tokens = truth.expand(self.samples_per_prompt, -1)
                    prompt_mask = (~loss_mask.bool()).expand_as(prompt_tokens)
                    generated = model.generate_samples(
                        self.samples_per_prompt,
                        num_steps=self.num_steps,
                        prompt_tokens=prompt_tokens,
                        prompt_mask=prompt_mask,
                        return_result=True,
                    ).samples
                    predictions.append(generated.cpu().numpy())
                    prompts.append(truth[0, : graph.config.n_waypoints].cpu().numpy())
                    hints.append(
                        hints_from_row(
                            dataset.tokens[index],
                            dataset.masks[index],
                            graph.config.n_waypoints,
                        )
                    )
                    if model.config.algo.name in {"ar", "mdlm"}:
                        losses = model._loss(truth, loss_mask)
                        nlls.append(float(losses.nlls.sum().cpu()) / np.log(2))
                if not predictions:
                    continue
                result = evaluate_samples(
                    graph,
                    np.asarray(prompts),
                    np.asarray(predictions),
                    hints,
                    nlls or None,
                    cache_dir=root / "goal_cache",
                )
                result["likelihood_kind"] = (
                    "exact AR NLL"
                    if view == "ar"
                    else "single Monte Carlo MDLM bound estimate"
                    if nlls
                    else "unavailable"
                )
                self.output_dir.mkdir(parents=True, exist_ok=True)
                (
                    self.output_dir / f"{split}_step_{trainer.global_step:08d}.json"
                ).write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
                if trainer.logger:
                    trainer.logger.log_metrics(
                        {
                            f"hubpaths/{split}/{k}": v
                            for k, v in result.items()
                            if isinstance(v, (int, float))
                        },
                        step=trainer.global_step,
                    )
        finally:
            model._train_mode()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        step = trainer.global_step
        if step > 0 and step != self.last_step and step % self.every_n_train_steps == 0:
            self.last_step = step
            if trainer.is_global_zero:
                self._run(trainer, pl_module)
            trainer.strategy.barrier("hubpaths-evaluation")

    def on_train_end(self, trainer, pl_module):
        if self.full_eval_at_end and trainer.global_step != self.last_step:
            if trainer.is_global_zero:
                self._run(trainer, pl_module)
            trainer.strategy.barrier("hubpaths-final-evaluation")

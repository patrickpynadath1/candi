"""Prepare inherited-path map/endpoint comparisons from explicit base configs.

Input JSON maps task names to {config, checkpoint} paths. No path or checkpoint
is guessed: an embedding base must actually declare an embedding representation.
"""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from omegaconf import OmegaConf
from core.sweeps import TrainingJob, InferenceJob
from core.sweeps.submissions import SubmissionQueue


def build_jobs(bases, updates=3000, examples=256, batch_size=16):
    jobs = []
    for task, paths in bases.items():
        for method in ("flow_map", "beckmann"):
            for baseline in (True, False):
                config = OmegaConf.load(paths["config"])
                config.mode = "train"
                config.seed = 1
                base = OmegaConf.to_container(config.algorithm, resolve=True)
                config.algorithm = dict(
                    _target_="core.training.procedures.affine_maps.AffineMap",
                    base=base,
                    endpoint=method == "beckmann",
                )
                config.procedure = OmegaConf.load(
                    Path(__file__).parents[1]
                    / "configs/procedure"
                    / f"{method}_inherited.yaml"
                )
                if baseline:
                    config.procedure = None
                config.training.initialize_from = str(paths["checkpoint"])
                config.training.warmup_steps = 100
                config.checkpoint.resume = None
                config.checkpoint.every_n_steps = max(1, updates)
                config.checkpoint.save_top_k = 0
                config.optimizer.lr = 1e-4
                config.trainer.max_steps = 0 if baseline else updates
                config.trainer.precision = "bf16-mixed"
                config.trainer.accumulate_grad_batches = 1
                config.trainer.gradient_clip_val = 1.0
                config.trainer.val_check_interval = min(500, updates)
                config.trainer.limit_val_batches = 2
                config.trainer.num_sanity_val_steps = 0
                config.trainer.enable_progress_bar = False
                config.trainer.strategy = "auto"
                config.loader.global_batch_size = batch_size
                config.loader.eval_batch_size = 4
                config.loader.num_workers = 2
                config.datamodule.loader_config = "${loader}"
                config.datamodule.accumulation_steps = (
                    "${trainer.accumulate_grad_batches}"
                )
                config.compile = method != "beckmann"
                if "num_eval_samples" in config.task:
                    config.task.num_eval_samples = examples
                if task == "tinystories":
                    config.task.calculate_bleu = True
                config.sampler = dict(
                    _target_="core.training.procedures.affine_maps.AffineMapSampler",
                    steps=4,
                )
                if baseline:
                    config.callback = None
                elif config.get("callback"):
                    config.callback.every_n_steps = 500
                    config.callback.subset_size = 32
                    config.callback.sampler_configs = None
                    config.callback.sampler_config = "${sampler}"
                    config.callback.output_dir = "${output_dir}/task_callback"
                    config.callback.batch_size = "${loader.eval_batch_size}"
                config.evaluation.max_examples = examples
                config.evaluation.use_ema = True
                config.evaluation.legacy = False
                config.evaluation.device = "cuda:0"
                label = "base" if baseline else "distilled"
                name = f"fewstep-{updates}-{method}-inherited-{task}-{label}-seed-1"
                config.wandb = dict(
                    project="distillation-smoke-20261026",
                    entity="rz-text-dlp",
                    group=f"fewstep-{updates}",
                    name=name + "/train",
                    job_type="train",
                )
                choices = tuple(
                    InferenceJob(
                        overrides={
                            "sampler.steps": nfe,
                            "evaluation.max_examples": examples,
                            "evaluation.policy": f"nfe-{nfe}",
                            "wandb.name": name + f"/nfe-{nfe}",
                        }
                    )
                    for nfe in (4, 16)
                )
                jobs.append(TrainingJob(name=name, config=config, inference=choices))
    return sorted(jobs, key=lambda j: "-base-" not in j.name)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bases", type=Path, required=True)
    p.add_argument("--queue", type=Path, required=True)
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--submit", action="store_true")
    args = p.parse_args()
    jobs = build_jobs(json.loads(args.bases.read_text()))
    rows = []
    for job in jobs:
        row = asdict(job)
        row["config"] = OmegaConf.to_container(job.config, resolve=True)
        rows.append(row)
    args.plan.write_text(json.dumps(rows, indent=2) + "\n")
    print(f"{len(jobs)} jobs, {sum(len(j.inference) for j in jobs)} evaluations")
    if args.submit:
        print(f"Submitted {SubmissionQueue(args.queue).submit(jobs)} jobs")


if __name__ == "__main__":
    main()

"""Small task-matched procedure sweep using the existing Ray training queue.

Run --plan first. Checkpoint discovery is restricted to explicit completed base
runs; the resulting plan records every source path. No data are downloaded.
"""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from omegaconf import OmegaConf
from core.sweeps import TrainingJob, InferenceJob
from core.sweeps.configuration import ROOT

TASKS = ("sudoku_hard", "text8", "tinystories")
RECIPES = ("alpha", "sdtt", "dmmd", "idlm", "dmd", "flow_map", "beckmann")


def discover(root):
    found = {}
    for p in sorted(root.glob("*/checkpoint_*/config.yaml")):
        checkpoint = p.parent / "checkpoints/last.ckpt"
        if not checkpoint.is_file():
            continue
        cfg = OmegaConf.load(p)
        name = OmegaConf.select(cfg, "wandb.name", default="")
        for family, label in [
            ("masked", "masked-sc-off"),
            ("gaussian", "gaussian-rank-linear-cdcd-sc-off"),
        ]:
            for task in TASKS:
                if name == f"{label}-20m-{task}-seed-1/train":
                    key = (family, task)
                    if key in found:
                        raise ValueError(f"ambiguous base checkpoint: {key}")
                    found[key] = (cfg, str(checkpoint))
    missing = {(f, t) for f in ("masked", "gaussian") for t in TASKS} - found.keys()
    if missing:
        raise ValueError(f"missing base checkpoints: {missing}")
    return found


def build_jobs(
    bases,
    *,
    updates=200,
    evaluation_examples=16,
    project="distillation-smoke-20261026",
    recipes=RECIPES,
):
    jobs = []
    for recipe in recipes:
        for task in TASKS:
            base, checkpoint = bases[
                ("gaussian" if recipe == "dmd" else "masked", task)
            ]
            cfg = OmegaConf.create(OmegaConf.to_container(base, resolve=True))
            cfg.mode = "train"
            cfg.seed = 1
            cfg.training.initialize_from = checkpoint
            cfg.training.warmup_steps = 10
            cfg.training.phase_gradient_clip = 1.0
            cfg.optimizer.lr = 1e-4
            cfg.checkpoint.resume = None
            cfg.procedure = OmegaConf.load(
                ROOT / "configs/procedure" / f"{recipe}.yaml"
            )
            alternating = recipe in ("dmd", "dmmd", "idlm")
            cfg.trainer.max_steps = updates * (2 if alternating else 1)
            cfg.trainer.accumulate_grad_batches = 1
            cfg.trainer.gradient_clip_val = 0.0 if alternating else 1.0
            # Lightning counts batches here, but manual global_step counts both
            # auxiliary and student optimizer updates.
            cfg.trainer.val_check_interval = updates
            cfg.trainer.limit_val_batches = 2
            cfg.trainer.num_sanity_val_steps = 0
            cfg.trainer.enable_progress_bar = False
            cfg.trainer.precision = "bf16-mixed"
            cfg.trainer.strategy = "auto"
            cfg.loader.global_batch_size = 4
            cfg.loader.eval_batch_size = 4
            cfg.loader.num_workers = 2
            cfg.datamodule.loader_config = "${loader}"
            cfg.datamodule.accumulation_steps = "${trainer.accumulate_grad_batches}"
            cfg.checkpoint.every_n_steps = cfg.trainer.max_steps
            cfg.checkpoint.save_top_k = 0
            cfg.evaluation.max_examples = evaluation_examples
            cfg.evaluation.device = "cuda:0"
            cfg.evaluation.use_ema = True
            cfg.evaluation.legacy = False
            cfg.callback.every_n_steps = cfg.trainer.max_steps
            cfg.callback.subset_size = min(8, evaluation_examples)
            cfg.callback.sampler_configs = None
            cfg.callback.sampler_config = "${sampler}"
            cfg.callback.output_dir = "${output_dir}/task_callback"
            cfg.callback.batch_size = "${loader.eval_batch_size}"
            cfg.compile = recipe != "beckmann"
            if task == "tinystories":
                cfg.task.calculate_bleu = True
            if recipe in ("flow_map", "beckmann"):
                endpoint = recipe == "beckmann"
                cfg.training.initialize_from = None
                backbone = dict(
                    _target_="core.models.dit.DiT",
                    vocab_size=cfg.task.vocab_size,
                    hidden_size=512,
                    cond_dim=128,
                    n_blocks=6,
                    n_heads=8,
                    dropout=0.0,
                    adaLN=True,
                )
                cfg.model = dict(
                    _target_="core.training.procedures.maps."
                    + ("EndpointMap" if endpoint else "MeanDenoiserMap"),
                    backbone=backbone,
                    vocab_size=cfg.task.vocab_size,
                )
                cfg.algorithm = dict(
                    _target_="core.training.procedures.map_training.MapAlgorithm",
                    model="${model}",
                    length=cfg.task.length,
                    endpoint=endpoint,
                )
                cfg.sampler = dict(
                    _target_="core.training.procedures.map_training.MapSampler",
                    steps=4,
                    step=dict(
                        _target_="core.training.procedures.maps."
                        + ("EndpointRefinementStep" if endpoint else "LearnedMapStep")
                    ),
                )
            else:
                cfg.sampler = dict(
                    _target_="core.samplers.composed.ComposedSampler",
                    steps=3,
                    tokens=dict(
                        _target_="core.samplers.policies.TokenSelection", mode="sample"
                    ),
                    transition=dict(
                        _target_="core.samplers.policies.ForwardCorruption"
                    ),
                    final_denoise="greedy",
                )
                if recipe in ("alpha", "sdtt", "dmmd", "idlm"):
                    cfg.sampler.tokens.mode = "posterior"
                    cfg.sampler.transition = dict(
                        _target_="core.samplers.policies.ReverseTransition"
                    )
            name = f"{recipe}-{task}-seed-1"
            cfg.wandb = dict(
                project=project,
                entity="rz-text-dlp",
                name=name + "/train",
                group="smoke",
                job_type="train",
            )
            choices = []
            for nfe in (4, 16):
                choices.append(
                    InferenceJob(
                        overrides={
                            "sampler.steps": nfe
                            if recipe in ("flow_map", "beckmann")
                            else nfe - 1,
                            "evaluation.max_examples": evaluation_examples,
                            "evaluation.policy": f"nfe-{nfe}",
                            "wandb.name": name + f"/nfe-{nfe}",
                        }
                    )
                )
            jobs.append(TrainingJob(name=name, config=cfg, inference=tuple(choices)))
    return jobs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-root",
        type=Path,
        default=Path.home() / ".cache/duo/batteries/three-tasks-20m-bf16-20261012",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path.home() / ".cache/duo/batteries/distillation-smoke-20261026",
    )
    parser.add_argument("--updates", type=int, default=200)
    parser.add_argument("--recipes", nargs="+", choices=RECIPES, default=list(RECIPES))
    parser.add_argument("--plan", action="store_true")
    parser.add_argument("--run-name", default="ray")
    args = parser.parse_args()
    jobs = build_jobs(
        discover(args.base_root), updates=args.updates, recipes=args.recipes
    )
    args.output.mkdir(parents=True, exist_ok=True)
    plan = dict(
        aliases={"discrete_flow_map_psd": "flow_map"},
        updates_per_student=args.updates,
        jobs=[
            dict(name=j.name, config=OmegaConf.to_container(j.config, resolve=True))
            for j in jobs
        ],
    )
    (args.output / "plan.json").write_text(json.dumps(plan, indent=2))
    print(
        f"{len(jobs)} training jobs; {2 * len(jobs)} inference jobs; plan={args.output}/plan.json",
        flush=True,
    )
    if args.plan:
        return
    import ray
    from ray import tune
    from core.sweeps.queue import TrainingInferenceSearch, run_job
    from core.sweeps.submissions import SubmissionQueue

    queue_path = args.output / "submissions.sqlite"
    queue = (
        SubmissionQueue(queue_path)
        if queue_path.exists()
        else SubmissionQueue.create(queue_path)
    )
    queue.submit(jobs)
    # Other experiments may have Ray heads on this machine. This queue owns
    # its four visible GPUs and must not attach to an unrelated head by default.
    ray.init(
        address="local",
        num_cpus=16,
        include_dashboard=False,
        _temp_dir=str(Path.home() / ".cache/duo/ray/dsmoke"),
    )
    trainable = tune.with_resources(run_job, {"cpu": 4, "gpu": 1})
    experiment = args.output / args.run_name
    if tune.Tuner.can_restore(str(experiment)):
        tuner = tune.Tuner.restore(
            str(experiment), trainable=trainable, resume_errored=True
        )
    else:
        tuner = tune.Tuner(
            trainable,
            tune_config=tune.TuneConfig(
                search_alg=TrainingInferenceSearch(submission_path=queue_path),
                num_samples=-1,
                max_concurrent_trials=4,
            ),
            run_config=tune.RunConfig(
                storage_path=str(args.output),
                name=args.run_name,
                # Ray's Tee lacks isatty(), which torch.compile queries.
                # Worker output still reaches the cache-resident driver log.
                log_to_file=False,
                failure_config=tune.FailureConfig(max_failures=0),
            ),
        )
    results = tuner.fit()
    results.get_dataframe().to_csv(args.output / "summary.csv", index=False)
    ray.shutdown()


if __name__ == "__main__":
    main()

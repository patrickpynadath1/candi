"""Controlled Sudoku corruption x hidden self-conditioning sweep (preview by default).

Train every cell for 10k steps; evaluate a native 128-NFE reverse sampler,
its top-1-clean-estimate variant, and greedy frozen EB (at most 128 NFE).
Spherical uses its manifold flow update,
not an ancestral kernel. Select one policy per configuration across seeds.
"""

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import random
import sys

os.environ.setdefault("RAY_ENABLE_UV_RUN_RUNTIME_ENV", "0")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from omegaconf import OmegaConf

from core.sweeps import InferenceJob, TrainingJob, compose


FAMILIES = ("blurring", "gaussian", "spherical", "simplex", "masked", "uniform")
POLICIES = ("native_128", "native_128_greedy", "eb_greedy")


@dataclass(frozen=True)
class Variant:
    family: str
    name: str
    schedule: str | None = None


def variants(args):
    for family in args.families:
        if family == "gaussian":
            yield Variant(family, "gaussian-rank-linear-cdcd", "rank-linear-cdcd")
        else:
            yield Variant(family, family)


def base_config(variant, args):
    cfg = compose(
        "corruption_factorial",
        {
            "hybrid_view@model.views.content": variant.family,
            "data.num_train": args.train_examples,
            "data.num_valid": args.eval_examples,
            "data.difficulty": "hard",
            "trainer.max_steps": args.train_steps,
            "trainer.val_check_interval": args.eval_every,
            "checkpoint.every_n_steps": args.eval_every,
            "loader.global_batch_size": args.batch_size,
            "loader.eval_batch_size": args.eval_batch_size,
            "loader.num_workers": args.data_workers,
        },
    )
    # Lightning counts validation intervals in microbatches, callbacks in updates.
    cfg.trainer.val_check_interval = (
        args.eval_every * cfg.trainer.accumulate_grad_batches
    )
    if args.data_cache_dir is not None:
        cfg.data.cache_dir = str(args.data_cache_dir.resolve())
    view = cfg.model.views.content
    if variant.family == "gaussian":
        schedule_cfg = compose(overrides={"noise_schedule": "cdcd_gaussian_adaptive"})
        base = compose(overrides={"noise_schedule": "one_hot_rank"}).noise_schedule
        base.inversion = "analytic"
        base.coupling = "linear"
        schedule_cfg.noise_schedule.base = base
        view.schedule = OmegaConf.to_container(
            schedule_cfg.noise_schedule, resolve=False
        )
        view.representation = {
            "_target_": "core.representations.noisy_inputs.ProjectedRepresentation",
            "state_dim": "${task.vocab_size}",
            "hidden_size": "${model.backbone.hidden_size}",
            "state_representation": {
                "_target_": "core.representations.one_hot.OneHotRepresentation",
                "vocab_size": "${task.vocab_size}",
            },
        }
    elif variant.family == "spherical":
        # Native default, not the adaptive/truncated historical Sudoku preset.
        source = compose("spherical_flow_matching")
        view.schedule = OmegaConf.to_container(source.noise_schedule, resolve=False)
        view.schedule.dimension = source.representation.dimension
    elif variant.family == "blurring":
        view.schedule.feature_rho = 0.5
    return cfg


def build_jobs(args):
    jobs = []
    for variant in variants(args):
        base = base_config(variant, args)
        for seed in args.seeds:
            for enabled in (False, True):
                cfg = OmegaConf.create(OmegaConf.to_container(base, resolve=False))
                name = f"{variant.name}-sc-{'on' if enabled else 'off'}-seed-{seed}"
                cfg.seed = seed
                cfg.self_conditioning.enabled = enabled
                cfg.wandb = {
                    "project": args.wandb_project,
                    "entity": args.wandb_entity,
                    "name": name,
                    "group": args.name,
                    "job_type": "train",
                    "mode": "online",
                    "tags": [variant.family, "sc-on" if enabled else "sc-off"],
                }
                OmegaConf.update(
                    cfg,
                    "factorial",
                    {
                        "study": "corruption_self_conditioning",
                        "run_name": name,
                        "family": variant.family,
                        "variant": variant.name,
                        "gaussian_schedule": variant.schedule,
                        "schedule_adaptation": "cdcd"
                        if variant.family == "gaussian"
                        else "none",
                        "self_conditioning": enabled,
                        "training_seed": seed,
                        "native_update": (
                            "manifold_flow"
                            if variant.family == "spherical"
                            else "stochastic_reverse"
                        ),
                    },
                    force_add=True,
                )
                common = {
                    "evaluation.max_examples": args.eval_examples,
                    "evaluation.use_ema": True,
                    "seed": args.evaluation_seed,
                }
                inference = (
                    InferenceJob(
                        "hybrid",
                        {
                            **common,
                            "sampler.steps": 127,
                            "sampler.final_denoise": "greedy",
                            "sampler.final_transition_noise": False,
                        },
                    ),
                    InferenceJob(
                        "hybrid",
                        {
                            **common,
                            "sampler.steps": 127,
                            "sampler.final_denoise": "greedy",
                            "sampler.final_transition_noise": False,
                            "sampler.greedy_clean_estimate": True,
                        },
                    ),
                    InferenceJob(
                        "parallel_entropy_bounded",
                        {
                            **common,
                            "sampler.steps": 127,
                            "sampler.entropy_bound": args.entropy_bound,
                            "sampler.freeze": True,
                            "sampler.greedy_clean_estimate": True,
                        },
                    ),
                )
                cfg.callback.every_n_steps = args.eval_every
                cfg.callback.subset_size = args.periodic_eval_examples
                cfg.callback.seed = args.evaluation_seed
                samplers = {}
                labelled = []
                for policy, choice in zip(POLICIES, inference):
                    sampler = compose(overrides={"sampler": choice.sampler}).sampler
                    for key, value in choice.overrides.items():
                        if key.startswith("sampler."):
                            OmegaConf.update(
                                sampler, key.removeprefix("sampler."), value
                            )
                    samplers[policy] = OmegaConf.to_container(sampler, resolve=False)
                    labelled.append(
                        InferenceJob(
                            choice.sampler,
                            {
                                **choice.overrides,
                                "wandb.name": f"{name}/{policy}",
                                "wandb.job_type": "evaluate",
                                "evaluation.policy": policy,
                            },
                        )
                    )
                cfg.callback.sampler_configs = samplers
                inference = tuple(labelled)
                OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
                jobs.append(
                    TrainingJob(
                        name,
                        config=OmegaConf.to_container(cfg, resolve=False),
                        inference=inference,
                    )
                )
    return jobs


def make_plan(args, jobs):
    indices = list(range(args.eval_examples))
    random.Random(args.split_seed).shuffle(indices)
    middle = len(indices) // 2
    contents = {
        "version": 1,
        "jobs": [asdict(job) for job in jobs],
        "protocol": {
            "train_steps": args.train_steps,
            "seeds": args.seeds,
            "evaluation_seed": args.evaluation_seed,
            "split_seed": args.split_seed,
            "selection_indices": sorted(indices[:middle]),
            "report_indices": sorted(indices[middle:]),
            "policies": list(POLICIES),
            "selection": "highest mean selection success across seeds per variant x SC",
            "tie_break": "lowest mean selection NFE, then policy name",
            "metric": "valid_solution",
            "family_weighting": "equal families; one recipe per family",
        },
    }
    encoded = json.dumps(contents, sort_keys=True, allow_nan=False)
    return {"fingerprint": hashlib.sha256(encoded.encode()).hexdigest(), **contents}


def parser():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--run", action="store_true")
    cli.add_argument("--resume", action="store_true")
    cli.add_argument(
        "--report", action="store_true", help="Summarize existing evaluations only"
    )
    cli.add_argument(
        "--write-plan", type=Path, help="Save the preview manifest without launching"
    )
    cli.add_argument("--families", nargs="+", choices=FAMILIES, default=list(FAMILIES))
    cli.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    cli.add_argument("--train-steps", type=int, default=10_000)
    cli.add_argument("--eval-every", type=int, default=1000)
    cli.add_argument("--periodic-eval-examples", type=int, default=128)
    cli.add_argument("--train-examples", type=int, default=48_000)
    cli.add_argument("--eval-examples", type=int, default=2_000)
    cli.add_argument("--evaluation-seed", type=int, default=1729)
    cli.add_argument("--split-seed", type=int, default=271828)
    cli.add_argument("--entropy-bound", type=float, default=0.1)
    cli.add_argument("--batch-size", type=int, default=256)
    cli.add_argument("--eval-batch-size", type=int, default=64)
    cli.add_argument("--data-workers", type=int, default=2)
    cli.add_argument(
        "--data-cache-dir", type=Path, default=Path.home() / ".cache/duo/data_cache"
    )
    cli.add_argument("--max-concurrent", type=int, default=2)
    cli.add_argument("--cpus-per-trial", type=int, default=4)
    cli.add_argument(
        "--storage", type=Path, default=Path.home() / ".cache/duo/outputs/ray"
    )
    cli.add_argument(
        "--ray-temp-dir", type=Path, default=Path.home() / ".cache/duo/ray"
    )
    cli.add_argument("--name", default="corruption-sc-10k")
    cli.add_argument("--wandb-project", default="self-cond-sweep")
    cli.add_argument("--wandb-entity", default="rz-text-dlp")
    cli.add_argument("--verbose", action="store_true")
    return cli


def main(argv=None):
    cli = parser()
    args = cli.parse_args(argv)
    if Path(args.name).name != args.name or args.name in {".", ".."}:
        cli.error("--name must be a single directory name")
    experiment = args.storage.resolve() / args.name
    if args.report:
        if args.run or args.resume or args.write_plan:
            cli.error("--report cannot be combined with launch or plan writing")
        from examples.report_corruption_self_conditioning import report

        report(experiment)
        return
    if args.resume and not args.run:
        cli.error("--resume requires --run")
    for key in (
        "train_steps",
        "eval_every",
        "periodic_eval_examples",
        "train_examples",
        "batch_size",
        "eval_batch_size",
        "max_concurrent",
        "cpus_per_trial",
    ):
        if getattr(args, key) <= 0:
            cli.error(f"--{key.replace('_', '-')} must be positive")
    if args.eval_examples < 2 or args.data_workers < 0:
        cli.error("Need at least two evaluation examples and nonnegative data workers")
    if not 0 <= args.entropy_bound < float("inf"):
        cli.error("Entropy bound must be finite and nonnegative")
    if any(seed < 0 for seed in [*args.seeds, args.evaluation_seed, args.split_seed]):
        cli.error("Seeds must be nonnegative")
    for key in ("families", "seeds"):
        values = getattr(args, key)
        if len(values) != len(set(values)):
            cli.error(f"Duplicate values in --{key.replace('_', '-')} are not allowed")
    jobs = build_jobs(args)
    plan = make_plan(args, jobs)
    print(f"Training: {len(jobs)} runs x {args.train_steps:,} steps")
    print(
        f"Inference: {len(POLICIES) * len(jobs)} jobs; native soft/top-1 128 NFE and greedy frozen EB <=128 NFE"
    )
    print(
        "Spherical uses manifold flow; continuous EB is an experimental commitment policy."
    )
    print(
        f"Selection/report: {args.eval_examples // 2}/{args.eval_examples - args.eval_examples // 2} puzzles"
    )
    print(
        "Best sampler is selected separately per corruption configuration x SC, across seeds."
    )
    if args.verbose:
        for job in jobs:
            print(job.name)
    if args.write_plan:
        if args.write_plan.exists():
            cli.error("--write-plan path already exists; choose another path")
        args.write_plan.parent.mkdir(parents=True, exist_ok=True)
        args.write_plan.write_text(json.dumps(plan, indent=2) + "\n")
    if not args.run:
        print("Preview only: no data preparation, training, or Ray initialization.")
        return
    launch(args, plan, jobs, cli)


def launch(args, plan, jobs, cli):
    import ray
    from ray import tune
    from core.sweeps.queue import TrainingInferenceSearch, run_job

    experiment = args.storage.resolve() / args.name
    manifest = experiment / "job-plan.json"
    data = OmegaConf.create(jobs[0].config).data
    cache = Path(data.cache_dir) / (
        f"sudoku_{data.difficulty}_train{data.num_train}_valid{data.num_valid}"
        f"_seed{data.data_seed}"
    )
    if not (cache / "dataset_dict.json").is_file():
        cli.error(
            f"Existing Sudoku cache required at {cache}; set --data-cache-dir. No data generated."
        )
    if args.resume:
        if not manifest.is_file() or not tune.Tuner.can_restore(str(experiment)):
            cli.error("No restorable sweep with a manifest at this location")
        if json.loads(manifest.read_text())["fingerprint"] != plan["fingerprint"]:
            cli.error(
                "Plan changed; restore with the original arguments or use a new name"
            )
    elif experiment.exists():
        cli.error("Experiment already exists; use --run --resume or a new name")
    else:
        experiment.mkdir(parents=True)
        manifest.write_text(json.dumps(plan, indent=2) + "\n")
    args.ray_temp_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("WANDB_DATA_DIR", str(Path.home() / ".cache/duo/wandb-data"))
    Path(os.environ["WANDB_DATA_DIR"]).mkdir(parents=True, exist_ok=True)
    ray.init(_temp_dir=str(args.ray_temp_dir.absolute()))
    try:
        trainable = tune.with_resources(run_job, {"cpu": args.cpus_per_trial, "gpu": 1})
        tuner = (
            tune.Tuner.restore(str(experiment), trainable=trainable)
            if args.resume
            else tune.Tuner(
                trainable,
                tune_config=tune.TuneConfig(
                    search_alg=TrainingInferenceSearch(jobs),
                    num_samples=-1,
                    max_concurrent_trials=args.max_concurrent,
                ),
                run_config=tune.RunConfig(
                    storage_path=str(args.storage.resolve()),
                    name=args.name,
                    log_to_file=True,
                    failure_config=tune.FailureConfig(max_failures=1),
                ),
            )
        )
        results = tuner.fit()
        from examples.report_corruption_self_conditioning import report

        report(experiment)
        if results.num_errors:
            raise RuntimeError(f"{results.num_errors} trials failed; inspect Ray logs")
    finally:
        ray.shutdown()


if __name__ == "__main__":
    main()

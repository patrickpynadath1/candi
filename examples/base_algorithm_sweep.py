"""Plan or launch a finite catalogue of base diffusion experiments on Sudoku hard.

Preview: uv run python examples/base_algorithm_sweep.py
Launch:  CUDA_VISIBLE_DEVICES=2,3 uv run --extra tracking python examples/base_algorithm_sweep.py --run --skip-unsupported
"""

import argparse
from dataclasses import asdict, dataclass, field
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any

# Local Ray workers use this checkout and uv environment, including cache links.
os.environ.setdefault("RAY_ENABLE_UV_RUN_RUNTIME_ENV", "0")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from omegaconf import OmegaConf

from core.sweeps import InferenceJob, TrainingJob, compose


@dataclass(frozen=True)
class ScheduleChoice:
    name: str
    group: str
    options: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MethodVariant:
    family: str
    name: str
    experiment: str
    overrides: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BlockedJob:
    name: str
    reason: str


FAMILIES = (
    "gaussian",
    "simplex",
    "spherical",
    "uniform_plugin",
    "uniform_ce",
    "mdlm",
    "replaid",
    "blurring",
)
CLOCKS = ("fixed", "cdcd")
# Native coefficient paths only: no rescaled_gaussian variants.
COMMON_GAUSSIAN = (
    ScheduleChoice("exponential-ve", "exponential_ve"),
    ScheduleChoice("linear-interpolation", "linear_interpolation"),
)


def gaussian_variants():
    for representation in ("one_hot", "embedding"):
        rank = (
            [
                ScheduleChoice(
                    f"rank-analytic-{coupling}",
                    "one_hot_rank",
                    {"inversion": "analytic", "coupling": coupling},
                )
                for coupling in ("linear", "geometric")
            ]
            if representation == "one_hot"
            else [
                ScheduleChoice(
                    f"rank-{coupling}", "embedding_sphere_rank", {"coupling": coupling}
                )
                for coupling in ("linear", "geometric")
            ]
        )
        errors = (
            [
                ScheduleChoice(
                    f"argmax-error-{coupling}", "argmax_error", {"coupling": coupling}
                )
                for coupling in ("linear", "geometric")
            ]
            if representation == "one_hot"
            else []
        )
        for schedule in [*rank, *errors, *COMMON_GAUSSIAN]:
            for clock in CLOCKS:
                overrides = {"representation": representation}
                if clock == "fixed":
                    overrides["noise_schedule"] = schedule.group
                    prefix = "noise_schedule"
                else:
                    overrides["noise_schedule"] = f"{clock}_gaussian_adaptive"
                    overrides["noise_schedule@noise_schedule.base"] = schedule.group
                    prefix = "noise_schedule.base"
                overrides.update(
                    {
                        f"{prefix}.{key}": value
                        for key, value in schedule.options.items()
                    }
                )
                yield MethodVariant(
                    "gaussian",
                    f"gaussian-{representation}-{schedule.name}-{clock}",
                    "gaussian_base",
                    overrides,
                )


def method_variants():
    yield from gaussian_variants()
    # SDM reference Sudoku setting: constant-then-linear normalized variance.
    for adaptive in (False, True):
        yield MethodVariant(
            "simplex",
            f"simplex-constant_linear-adaptive-{int(adaptive)}",
            "simplex",
            {
                "noise_schedule.kind": "constant_linear",
                "noise_schedule.nu0": 0.4,
                "noise_schedule.nu1": 0.75,
                "noise_schedule.pivot": 0.8,
                "time_sampler.adaptive": adaptive,
            },
        )
    for adaptive in (False, True):
        for truncated in (False, True):
            prefix = "noise_schedule.base" if adaptive else "noise_schedule"
            overrides = {
                "noise_schedule": "cdcd_adaptive" if adaptive else "spherical",
                f"{prefix}.auto_truncation": truncated,
            }
            if adaptive:
                overrides["noise_schedule@noise_schedule.base"] = "spherical"
            yield MethodVariant(
                "spherical",
                f"spherical-{'cdcd' if adaptive else 'fixed'}-truncated-{int(truncated)}",
                "spherical_flow_matching",
                overrides,
            )
    for family, experiment in (
        ("uniform_plugin", "uniform_diffusion"),
        ("uniform_ce", "uniform_posterior"),
        ("mdlm", "sudoku_hard"),
    ):
        for clock in CLOCKS:
            yield MethodVariant(
                family,
                f"{family}-{clock}",
                experiment,
                {
                    "noise_schedule": "log_linear"
                    if clock == "fixed"
                    else f"{clock}_adaptive",
                },
            )
    yield MethodVariant("replaid", "replaid", "replaid")
    # Reference blurring preset: joint position/feature decay, rho=.5.
    yield MethodVariant(
        "blurring",
        "blurring-position-feature",
        "sudoku_blurring_hard",
        {"noise_schedule.feature_rho": 0.5},
    )


def inference_jobs(family, steps, entropy_bounds, max_examples):
    def job(sampler=None, **overrides):
        return InferenceJob(
            sampler=sampler,
            overrides={"evaluation.max_examples": max_examples, **overrides},
        )

    if family == "gaussian":
        return tuple(
            job(sampler, **{"sampler.steps": n})
            for sampler in ("euler", "heun")
            for n in steps
        )
    if family in {"mdlm", "uniform_plugin", "uniform_ce"}:
        ancestral, eb = (
            ("ancestral", "entropy_bounded")
            if family == "mdlm"
            else ("uniform_ancestral", "uniform_entropy_bounded")
        )
        return tuple(job(ancestral, **{"sampler.steps": n}) for n in steps) + tuple(
            job(eb, **{"sampler.entropy_bound": bound}) for bound in entropy_bounds
        )
    if family == "simplex":
        return tuple(
            job(None, **{"sampler.steps": n, "sampler.churn": churn})
            for n in steps
            for churn in (0.0, 1.0)
        )
    # Keeping the preset sampler also preserves RePlaid's bootstrap feedback and
    # final-transition-noise settings. Its reference runs require final/raw weights.
    extra = {"evaluation.use_ema": False} if family == "replaid" else {}
    return tuple(job(None, **{"sampler.steps": n, **extra}) for n in steps)


def build_jobs(args):
    jobs, blocked = [], []
    for variant in method_variants():
        if variant.family not in args.families:
            continue
        for seed in args.seeds:
            for conditioning in args.self_conditioning:
                name = f"{variant.name}-sc-{conditioning}-seed-{seed}"
                overrides = {
                    **variant.overrides,
                    "seed": seed,
                    "trainer.max_steps": args.train_steps,
                    "trainer.val_check_interval": args.train_steps,
                    "checkpoint.every_n_steps": args.train_steps,
                    "loader.global_batch_size": args.batch_size,
                    "loader.eval_batch_size": args.eval_batch_size,
                    "loader.num_workers": args.data_workers,
                    "self_conditioning.enabled": conditioning == "on",
                }
                for key, value in (
                    ("data.cache_dir", args.data_cache_dir),
                    ("data.num_train", args.train_examples),
                    ("data.num_valid", args.validation_examples),
                ):
                    if value is not None:
                        overrides[key] = (
                            str(value) if isinstance(value, Path) else value
                        )
                if variant.family != "replaid":
                    overrides.update(
                        {
                            "self_conditioning.source": "hidden",
                            "self_conditioning.probability": 0.9,
                        }
                    )
                # RePlaid retains posterior-mean feedback, p=.25, fixed-count
                # selection, learned schedule and specialized optimizer groups.
                cfg = compose(variant.experiment, overrides)
                cfg.wandb = {
                    "project": args.wandb_project,
                    "name": name,
                    "group": args.name,
                    "job_type": "train",
                    "mode": args.wandb_mode,
                    **({"entity": args.wandb_entity} if args.wandb_entity else {}),
                }
                OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
                jobs.append(
                    TrainingJob(
                        name=name,
                        config=OmegaConf.to_container(cfg, resolve=False),
                        inference=inference_jobs(
                            variant.family,
                            args.sampling_steps,
                            args.entropy_bounds,
                            args.max_examples,
                        ),
                    )
                )
    return jobs, blocked


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument(
        "--run", action="store_true", help="Launch Ray; otherwise only print the plan."
    )
    result.add_argument(
        "--resume", action="store_true", help="Restore this exact saved plan."
    )
    result.add_argument(
        "--skip-unsupported",
        action="store_true",
        help="Explicitly omit any combinations listed as blocked.",
    )
    result.add_argument(
        "--families", nargs="+", choices=FAMILIES, default=list(FAMILIES)
    )
    result.add_argument("--seeds", nargs="+", type=int, default=[1])
    result.add_argument(
        "--self-conditioning", nargs="+", choices=("off", "on"), default=["off", "on"]
    )
    result.add_argument("--train-steps", type=int, default=50_000)
    result.add_argument("--sampling-steps", nargs="+", type=int, default=[32, 64, 128])
    result.add_argument(
        "--entropy-bounds", nargs="+", type=float, default=[0.05, 0.1, 0.2]
    )
    result.add_argument("--max-examples", type=int, default=2_000)
    result.add_argument("--batch-size", type=int, default=256)
    result.add_argument("--eval-batch-size", type=int, default=64)
    result.add_argument("--data-workers", type=int, default=2)
    result.add_argument("--data-cache-dir", type=Path, default=None)
    result.add_argument("--train-examples", type=int, default=None)
    result.add_argument("--validation-examples", type=int, default=None)
    result.add_argument("--max-concurrent", type=int, default=2)
    result.add_argument("--train-cpus", type=int, default=4)
    result.add_argument("--inference-cpus", type=int, default=2)
    result.add_argument("--storage", type=Path, default=Path("core/outputs/ray"))
    result.add_argument("--name", default="base-algorithms")
    result.add_argument("--wandb-project", default="candi")
    result.add_argument("--wandb-entity", default=None)
    result.add_argument(
        "--wandb-mode", choices=("online", "offline", "disabled"), default="online"
    )
    result.add_argument(
        "--verbose",
        action="store_true",
        help="List individual jobs and their inference counts.",
    )
    return result


def main(argv=None):
    cli = parser()
    args = cli.parse_args(argv)
    if any(
        value is not None and value <= 0
        for value in (args.train_examples, args.validation_examples)
    ):
        cli.error("Dataset example counts must be positive")
    if any(
        value <= 0
        for value in [
            args.train_steps,
            args.max_examples,
            args.batch_size,
            args.eval_batch_size,
            args.max_concurrent,
            args.train_cpus,
            args.inference_cpus,
            *args.sampling_steps,
        ]
    ):
        cli.error(
            "Steps, batch sizes, resource counts and example counts must be positive"
        )
    if args.data_workers < 0 or any(
        not 0 < value < float("inf") for value in args.entropy_bounds
    ):
        cli.error(
            "Data workers must be nonnegative and entropy bounds finite and positive"
        )
    for label in (
        "families",
        "seeds",
        "self_conditioning",
        "sampling_steps",
        "entropy_bounds",
    ):
        values = getattr(args, label)
        if len(values) != len(set(values)):
            cli.error(f"Duplicate values in --{label.replace('_', '-')}")
    if Path(args.name).name != args.name or args.name in {".", ".."}:
        cli.error("--name must be a single directory name")
    jobs, blocked = build_jobs(args)
    print(f"Training jobs: {len(jobs)} supported; {len(blocked)} blocked")
    print(
        f"Inference jobs: {sum(len(job.inference) for job in jobs)} (created after successful training)"
    )
    print(
        f"Budget: {args.train_steps} optimizer steps; {args.max_examples} evaluation examples"
    )
    for family in args.families:
        count = sum(job.name.startswith(family + "-") for job in jobs)
        print(f"  {family}: {count} training jobs")
    if args.verbose:
        for job in jobs:
            print(f"  {job.name}: {len(job.inference)} inference jobs")
    for item in blocked:
        print(f"  BLOCKED {item.name}: {item.reason}")
    if not args.run:
        print(
            "Plan only. Use --run to launch; no training or data preparation performed."
        )
        return
    if blocked and not args.skip_unsupported:
        cli.error(
            "Unsupported combinations remain; use --skip-unsupported to run the explicitly supported subset"
        )
    if not jobs:
        cli.error("No supported jobs selected")
    from ray import tune
    import ray
    from core.sweeps.queue import TrainingInferenceSearch, run_job

    experiment = args.storage.resolve() / args.name
    plan = json.dumps([asdict(job) for job in jobs], sort_keys=True)
    fingerprint = hashlib.sha256(plan.encode()).hexdigest()
    manifest = experiment / "job-plan.json"
    if args.resume:
        if not tune.Tuner.can_restore(str(experiment)) or not manifest.is_file():
            cli.error("No restorable sweep with a recorded job plan at this location")
        if json.loads(manifest.read_text())["fingerprint"] != fingerprint:
            cli.error("The job plan changed; use a new --name instead of --resume")
    elif experiment.exists():
        cli.error("Experiment directory already exists; use --resume or a new --name")
    else:
        experiment.mkdir(parents=True)
        manifest.write_text(
            json.dumps(
                {
                    "fingerprint": fingerprint,
                    "jobs": json.loads(plan),
                    "blocked": [asdict(item) for item in blocked],
                },
                indent=2,
            )
        )

    def resources(job):
        return {
            "cpu": args.train_cpus if job["stage"] == "train" else args.inference_cpus,
            "gpu": 1,
        }

    ray.init()
    try:
        trainable = tune.with_resources(run_job, resources)
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
        print(results.get_dataframe())
        if results.num_errors:
            raise RuntimeError(
                f"{results.num_errors} jobs failed; inspect results.errors"
            )
    finally:
        ray.shutdown()


if __name__ == "__main__":
    main()

"""36 matched Sudoku hard runs; preview by default, launch with --run.

Continuous baselines use one view of the same modifier/backbone used by their
hybrids, so adding a second view doesn't also replace the backbone/readout.
"""

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("RAY_ENABLE_UV_RUN_RUNTIME_ENV", "0")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from omegaconf import OmegaConf, open_dict
from core.sweeps import TrainingJob, InferenceJob, compose

FEEDBACK = ("none", "hidden", "token")
CONTINUOUS = ("gaussian", "spherical", "simplex")
DISCRETE = {
    "masked": "sudoku_hard",
    "uniform-ce": "uniform_posterior",
    "uniform-plugin": "uniform_diffusion",
}


def continuous_config(family, mixture):
    cfg = compose(
        "hybrid",
        {
            "hybrid_view@model.views.first": family,
            "hybrid_view@model.views.second": "masked"
            if mixture == "none"
            else mixture,
        },
    )
    with open_dict(cfg):
        if mixture == "none":
            del cfg.model.views.second
        if family == "gaussian":
            cfg.model.time_conditioning = "log_one_minus_time"
            cfg.model.views.first.update = {
                "_target_": "core.samplers.hybrid_updates.GaussianVelocityEulerUpdate"
            }
        elif family == "spherical":
            paper = compose("sudoku_spherical_truncated_adaptive")
            cfg.model.views.first.schedule = OmegaConf.to_container(
                paper.noise_schedule, resolve=True
            )
            cfg.model.time_conditioning = "negative_log_keep"
        else:
            paper = compose("simplex")
            cfg.time_sampler = OmegaConf.to_container(paper.time_sampler, resolve=True)
            cfg.trainer.precision = "32-true"
    return cfg


def build_job(group, family, feedback, mixture=None):
    """Compose one feedback recipe without constructing unrelated variants."""
    cfg = (
        continuous_config(family, mixture)
        if family in CONTINUOUS
        else compose(DISCRETE[family])
    )
    label = (
        f"{family}{'-' + mixture if mixture not in (None, 'none') else ''}-{feedback}"
    )
    settings = {
        "seed": 1,
        "data.difficulty": "hard",
        "data.num_train": 48000,
        "data.num_valid": 2000,
        "data.cache_dir": "/home/patrick/.cache/duo/data_cache",
        "trainer.max_steps": 20000,
        "trainer.val_check_interval": 20000,
        "trainer.accumulate_grad_batches": 4,
        "trainer.enable_progress_bar": False,
        "checkpoint.every_n_steps": 1000,
        "checkpoint.save_top_k": 2,
        "loader.global_batch_size": 256,
        "loader.eval_batch_size": 64,
        "loader.num_workers": 2,
        "callback.every_n_steps": 1000,
        "callback.subset_size": 128,
        "self_conditioning.enabled": feedback == "hidden",
        "self_conditioning.source": "hidden",
        "self_conditioning.probability": 0.9,
        "token_feedback.enabled": feedback == "token",
        "token_feedback.depth": 3,
        "token_feedback.loss_weight": 1.0,
        "token_feedback.gate": "scalar",
        "token_feedback.detach": False,
    }
    for key, value in settings.items():
        OmegaConf.update(cfg, key, value, merge=False, force_add=True)
    if family == "masked":
        cfg.sampler = compose(overrides={"sampler": "ancestral"}).sampler
    cfg.sampler.steps = 180
    cfg.wandb = {
        "project": "candi",
        "entity": "rz-text-dlp",
        "group": group,
        "name": label,
        "job_type": "train",
        "mode": "online",
    }
    inference = [
        InferenceJob(overrides={"sampler.steps": n, "evaluation.max_examples": 2000})
        for n in (32, 180)
    ]
    if family in DISCRETE:
        inference.append(
            InferenceJob(
                "entropy_bounded" if family == "masked" else "uniform_entropy_bounded",
                {"sampler.entropy_bound": 0.1, "evaluation.max_examples": 2000},
            )
        )
    return TrainingJob(
        label,
        config=OmegaConf.to_container(cfg, resolve=False),
        inference=tuple(inference),
    )


def build_jobs(group):
    # Interleave families so results don't require waiting through all Gaussian jobs.
    variants = [(f, "none") for f in CONTINUOUS] + [(f, None) for f in DISCRETE]
    variants += [(f, mixture) for mixture in ("masked", "uniform") for f in CONTINUOUS]
    return [
        build_job(group, family, feedback, mixture)
        for feedback in FEEDBACK
        for family, mixture in variants
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true")
    parser.add_argument(
        "--name", default="sudoku-continuous-hybrid-feedback-48k-20k-20261003"
    )
    parser.add_argument("--gpus", type=int, default=4)
    parser.add_argument(
        "--recovery-plan",
        help="JSON list of TrainingJob records with checkpoint.resume set",
    )
    args = parser.parse_args()
    if args.recovery_plan:
        records = json.loads(Path(args.recovery_plan).read_text())
        jobs = [
            TrainingJob(
                **{
                    **record,
                    "inference": tuple(InferenceJob(**i) for i in record["inference"]),
                }
            )
            for record in records
        ]
    else:
        jobs = build_jobs(args.name)
    for job in jobs:
        print(job.name, flush=True)
    print(
        f"{len(jobs)} training jobs; {sum(len(j.inference) for j in jobs)} dependent inference jobs",
        flush=True,
    )
    if not args.run:
        return
    import ray
    from ray import tune
    from core.sweeps.queue import TrainingInferenceSearch, run_job

    storage = Path("core/outputs/ray").resolve()
    output = storage / args.name
    output.mkdir(parents=True, exist_ok=False)
    (output / "job-plan.json").write_text(
        json.dumps([asdict(j) for j in jobs], indent=2)
    )
    ray.init(
        address="local",
        num_cpus=args.gpus * 4,
        num_gpus=args.gpus,
        object_store_memory=2 * 1024**3,
        include_dashboard=False,
    )
    try:
        result = tune.Tuner(
            tune.with_resources(run_job, {"cpu": 4, "gpu": 1}),
            tune_config=tune.TuneConfig(
                search_alg=TrainingInferenceSearch(jobs),
                num_samples=-1,
                max_concurrent_trials=args.gpus,
            ),
            run_config=tune.RunConfig(
                storage_path=str(storage),
                name=args.name,
                log_to_file=True,
                failure_config=tune.FailureConfig(max_failures=1),
            ),
        ).fit()
        print(result.get_dataframe())
        if result.num_errors:
            raise RuntimeError(f"{result.num_errors} jobs failed")
    finally:
        ray.shutdown()


if __name__ == "__main__":
    main()

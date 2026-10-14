"""Nine matched discrete feedback runs, 5k Sudoku Hard steps each."""

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("RAY_ENABLE_UV_RUN_RUNTIME_ENV", "0")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from omegaconf import OmegaConf
from core.sweeps import TrainingJob, InferenceJob
from examples.continuous_hybrid_feedback_sweep import (
    DISCRETE,
    FEEDBACK,
    build_job as feedback_job,
)


def build_job(group, family, feedback):
    """One discrete recipe with the 5k budget and expanded inference policies."""
    if family not in DISCRETE:
        raise ValueError(f"Unknown discrete family: {family}")
    base = feedback_job(group, family, feedback)
    config = OmegaConf.create(base.config)
    config.trainer.max_steps = 5000
    config.trainer.val_check_interval = 5000
    inference = [
        InferenceJob(overrides={"sampler.steps": n, "evaluation.max_examples": 2000})
        for n in (32, 180)
    ]
    for bound in (0.1, 0.01):
        if family == "masked":
            inference.append(
                InferenceJob(
                    "entropy_bounded",
                    {
                        "sampler.entropy_bound": bound,
                        "sampler.max_steps": 180,
                        "evaluation.max_examples": 2000,
                    },
                )
            )
        else:
            for freeze in (True, False):
                inference.append(
                    InferenceJob(
                        "uniform_entropy_bounded",
                        {
                            "sampler.entropy_bound": bound,
                            "sampler.steps": 180,
                            "sampler.freeze": freeze,
                            "evaluation.max_examples": 2000,
                        },
                    )
                )
    return TrainingJob(
        base.name,
        config=OmegaConf.to_container(config, resolve=False),
        inference=tuple(inference),
    )


def build_jobs(group):
    return [
        build_job(group, family, feedback)
        for feedback in FEEDBACK
        for family in DISCRETE
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--name", default="sudoku-discrete-feedback-48k-5k-20261005")
    parser.add_argument("--gpus", type=int, default=4)
    args = parser.parse_args()
    jobs = build_jobs(args.name)
    for job in jobs:
        print(job.name, flush=True)
    print(
        f"{len(jobs)} training jobs; {sum(len(j.inference) for j in jobs)} inference jobs",
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

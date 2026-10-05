"""One dynamic Tune queue: each completed training job adds inference jobs.

Run from the repository root:
RAY_ENABLE_UV_RUN_RUNTIME_ENV=0 CUDA_VISIBLE_DEVICES=2,3 uv run python examples/train_then_evaluate.py
"""

from pathlib import Path
import sys

# This repository is intentionally not installed as a Python package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import ray
from ray import tune

from core.sweeps import InferenceJob, TrainingJob
from core.sweeps.queue import TrainingInferenceSearch, run_job


# Sampler choices are conditional on the trained method, not a Cartesian product
# of every method with every sampler. Add compatible presets/settings here.
SAMPLERS = {
    "sudoku_hard": [
        InferenceJob(sampler="ancestral", overrides={"sampler.steps": 64}),
        InferenceJob(
            sampler="entropy_bounded", overrides={"sampler.entropy_bound": 0.1}
        ),
    ],
    "gaussian_diffusion_sudoku": [
        InferenceJob(sampler="euler", overrides={"sampler.steps": 64}),
        InferenceJob(sampler="heun", overrides={"sampler.steps": 64}),
    ],
}


def resources(job):
    # Ray schedules both stages together, using each job's own requirements.
    return {"cpu": 4 if job["stage"] == "train" else 2, "gpu": 1}


def main():
    storage = Path("core/outputs/ray").resolve()
    experiment = storage / "training-and-inference"
    jobs = [
        TrainingJob(
            name=f"{method}-seed-{seed}",
            experiment=method,
            overrides={
                "trainer.max_steps": 2_000,
                "trainer.val_check_interval": 2_000,
                "seed": seed,
            },
            inference=tuple(
                InferenceJob(
                    sampler=choice.sampler,
                    overrides={**choice.overrides, "evaluation.max_examples": 200},
                )
                for choice in choices
            ),
        )
        for method, choices in SAMPLERS.items()
        for seed in [1, 2]
    ]
    ray.init()
    try:
        trainable = tune.with_resources(run_job, resources)
        if tune.Tuner.can_restore(str(experiment)):
            # Restore the saved queue and trials, rather than enqueue duplicates.
            tuner = tune.Tuner.restore(str(experiment), trainable=trainable)
        else:
            tuner = tune.Tuner(
                trainable,
                tune_config=tune.TuneConfig(
                    search_alg=TrainingInferenceSearch(jobs),
                    num_samples=-1,  # The job source signals completion itself.
                    max_concurrent_trials=2,
                ),
                run_config=tune.RunConfig(
                    storage_path=str(storage),
                    name=experiment.name,
                    log_to_file=True,
                    failure_config=tune.FailureConfig(max_failures=1),
                ),
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

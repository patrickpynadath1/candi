"""Four matched Sudoku MDLM runs: baseline, bootstrap, and token feedback."""

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("RAY_ENABLE_UV_RUN_RUNTIME_ENV", "0")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from omegaconf import OmegaConf
from core.sweeps import TrainingJob, InferenceJob, compose


def build_jobs(name):
    variants = [
        ("baseline-8layers", 8, False, False, 3),
        ("hidden-selfcond-8layers", 8, True, False, 3),
        ("token-feedback-8layers-depth4", 8, False, True, 3),
        ("token-feedback-16layers-depth8", 16, False, True, 7),
    ]
    jobs = []
    for label, layers, bootstrap, feedback, depth in variants:
        cfg = compose(
            "sudoku_hard",
            {
                "seed": 1,
                "data.num_train": 48000,
                "data.num_valid": 2000,
                "data.cache_dir": "/home/patrick/.cache/duo/data_cache",
                "trainer.max_steps": 20000,
                "trainer.val_check_interval": 20000,
                "checkpoint.every_n_steps": 5000,
                "loader.global_batch_size": 256,
                "loader.eval_batch_size": 64,
                "loader.num_workers": 2,
                "model.n_blocks": layers,
                "self_conditioning.enabled": bootstrap,
                "self_conditioning.source": "hidden",
                "self_conditioning.probability": 0.9,
                "token_feedback.enabled": feedback,
                "token_feedback.depth": depth,
                "token_feedback.loss_weight": 1.0,
                "token_feedback.gate": "scalar",
                "token_feedback.detach": False,
            },
        )
        cfg.wandb = {
            "project": "candi",
            "entity": "rz-text-dlp",
            "group": name,
            "name": label,
            "job_type": "train",
            "mode": "online",
        }
        jobs.append(
            TrainingJob(
                name=label,
                config=OmegaConf.to_container(cfg, resolve=False),
                inference=(
                    InferenceJob(
                        "ancestral",
                        {"sampler.steps": 180, "evaluation.max_examples": 2000},
                    ),
                    InferenceJob(
                        "entropy_bounded",
                        {"sampler.entropy_bound": 0.1, "evaluation.max_examples": 2000},
                    ),
                ),
            )
        )
    return jobs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--name", default="mdlm-feedback-ablation-20k-20261003")
    args = parser.parse_args()
    jobs = build_jobs(args.name)
    for job in jobs:
        print(job.name, json.dumps(job.config["token_feedback"]))
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
    ray.init(address="local", num_cpus=12, num_gpus=2, object_store_memory=2 * 1024**3)
    try:
        results = tune.Tuner(
            tune.with_resources(run_job, {"cpu": 4, "gpu": 1}),
            tune_config=tune.TuneConfig(
                search_alg=TrainingInferenceSearch(jobs),
                num_samples=-1,
                max_concurrent_trials=2,
            ),
            run_config=tune.RunConfig(
                storage_path=str(storage),
                name=args.name,
                log_to_file=True,
                failure_config=tune.FailureConfig(max_failures=1),
            ),
        ).fit()
        print(results.get_dataframe())
        if results.num_errors:
            raise RuntimeError(f"{results.num_errors} ablation jobs failed")
    finally:
        ray.shutdown()


if __name__ == "__main__":
    main()

"""Submit an ordered TinyGSM batch to an existing Ray head via Tune.

Use a new --root for each batch. Reusing a root restores its Tune experiment.
The head remains available for further Ray Jobs submissions after a batch ends.
"""

import argparse
import json
import os
from pathlib import Path

import ray
from ray import tune
from omegaconf import OmegaConf

from core.sweeps.jobs import TrainingJob
from core.sweeps.queue import TrainingInferenceSearch
from core.sweeps.configuration import compose
from core.sweeps.tune import run_trial


def run_job(job):
    # Apply to restored Tune snapshots as well as newly composed jobs.
    config = OmegaConf.create(job["trial"]["config"])
    config.callback.every_n_steps = 10000
    config.callback.subset_size = 128
    config.callback.seed = 42
    config.callback.batch_size = 4
    overrides = json.loads(os.environ.get("CANDI_TINY_GSM_OVERRIDES", "{}"))
    config = OmegaConf.merge(config, overrides)
    resume = os.environ.get("CANDI_MASKED_RESUME")
    if resume and config.wandb.name == "tinygsm_masked_sc":
        config.checkpoint.resume = resume
    run_trial({"config": config})


DEFAULT_EXPERIMENTS = [
    "tinygsm_masked_sc",
    "tinygsm_uniform_sc",
    "tinygsm_masked_gaussian_sc_adaptive",
    "tinygsm_uniform_gaussian_sc_adaptive",
    "tinygsm_simplex_sc",
    "tinygsm_simplex_masked_sc",
    "tinygsm_simplex_uniform_sc",
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--gpus", type=int, default=8)
    parser.add_argument("--cpus", type=int, default=112)
    parser.add_argument("--experiments", nargs="+", default=DEFAULT_EXPERIMENTS)
    parser.add_argument("--resume-masked-from", type=Path)
    parser.add_argument(
        "--config-overrides",
        type=Path,
        help="YAML overrides applied to new and restored trials",
    )
    parser.add_argument("--max-concurrent-trials", type=int, default=1)
    parser.add_argument(
        "--wandb-group", help="Keep run IDs/group when migrating to a new queue root"
    )
    parser.add_argument(
        "--resume-checkpoints",
        type=Path,
        help="JSON mapping experiment names to checkpoints",
    )
    args = parser.parse_args()
    if args.max_concurrent_trials < 1:
        parser.error("--max-concurrent-trials must be positive")
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    cache = str(Path(os.environ["RCAC_SCRATCH"]) / "candi-1-data/tinygsm")
    resumes = (
        json.loads(args.resume_checkpoints.read_text())
        if args.resume_checkpoints
        else {}
    )
    if set(resumes) - set(args.experiments):
        parser.error("resume checkpoint names must be selected experiments")
    for checkpoint in resumes.values():
        if not Path(checkpoint).is_file():
            raise FileNotFoundError(checkpoint)
    group = args.wandb_group or root.name
    jobs = []
    for name in args.experiments:
        cfg = compose(
            name,
            {"strategy": "ddp" if args.gpus > 1 else None, "data.cache_dir": cache},
        )
        cfg.wandb = dict(
            project="candi-tinygsm",
            entity="rz-text-dlp",
            group=group,
            name=name,
            id=f"{group}-{name}",
            resume="allow",
        )
        if name in resumes:
            cfg.checkpoint.resume = resumes[name]
        jobs.append(
            TrainingJob(name=name, config=OmegaConf.to_container(cfg, resolve=False))
        )
    runtime_env = {"env_vars": {}}
    if args.config_overrides:
        runtime_env["env_vars"]["CANDI_TINY_GSM_OVERRIDES"] = json.dumps(
            OmegaConf.to_container(OmegaConf.load(args.config_overrides), resolve=True)
        )
    if args.resume_masked_from:
        if not args.resume_masked_from.is_file():
            raise FileNotFoundError(args.resume_masked_from)
        runtime_env["env_vars"]["CANDI_MASKED_RESUME"] = str(args.resume_masked_from)
    ray.init(address="auto", runtime_env=runtime_env)
    try:
        trainable = tune.with_resources(run_job, {"cpu": args.cpus, "gpu": args.gpus})
        path = root / "training"
        if tune.Tuner.can_restore(str(path)):
            if args.max_concurrent_trials != 1:
                raise ValueError(
                    "Changing concurrency requires a new queue root; restored Tune queues retain their scheduler configuration"
                )
            tuner = tune.Tuner.restore(
                str(path),
                trainable=trainable,
                resume_unfinished=True,
                resume_errored=True,
            )
        else:
            tuner = tune.Tuner(
                trainable,
                tune_config=tune.TuneConfig(
                    search_alg=TrainingInferenceSearch(jobs),
                    num_samples=-1,
                    max_concurrent_trials=args.max_concurrent_trials,
                ),
                run_config=tune.RunConfig(
                    storage_path=str(root), name="training", log_to_file=True
                ),
            )
        results = tuner.fit()
        if results.num_errors:
            raise RuntimeError(
                f"{results.num_errors} training trials failed; inspect {path}"
            )
    finally:
        ray.shutdown()


if __name__ == "__main__":
    main()

"""Compose one algorithm with tasks/seeds and hand the jobs to Ray Tune."""

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import sys

from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.sweeps import InferenceJob, TrainingJob, compose
from core.sweeps.configuration import ROOT

METHOD_FIELDS = {
    "algorithm",
    "model",
    "parameterization",
    "prediction",
    "representation",
    "process",
    "objective",
    "time_sampler",
    "noise_schedule",
    "sampler",
    "self_conditioning",
    "block_diffusion",
    "token_feedback",
    "variable_length",
    "optimizer",
    "lr_scheduler",
}
TASK_FIELDS = {
    "task",
    "data",
    "loader",
    "trainer",
    "training",
    "checkpoint",
    "callback",
    "evaluation",
}


def _apply(config, settings, allowed):
    for key, value in settings.items():
        if key.split(".")[0] not in allowed:
            raise ValueError(f"{key} belongs outside this configuration")
        if allowed == TASK_FIELDS and key.startswith("callback.") and "sampler" in key:
            raise ValueError("Task settings cannot change sampler definitions")
        if key == "callback" and value is not None:
            raise ValueError("Set callback budget fields individually")
        OmegaConf.update(config, key, value, merge=False, force_add=True)


def build_jobs(algorithm, battery):
    """Return ordinary TrainingJobs; no scheduler, filesystem writes or model loading."""
    components = algorithm["components"]
    if any(
        k.split("@")[0] not in METHOD_FIELDS | {"hybrid_view"}
        or ("@" in k and k.split("@")[1].split(".")[0] not in METHOD_FIELDS)
        for k in components
    ):
        raise ValueError("Algorithm components must not select tasks or data")
    definition = {key: algorithm[key] for key in ("components", "settings", "policies")}
    algorithm_id = hashlib.sha256(
        json.dumps(definition, sort_keys=True).encode()
    ).hexdigest()
    seeds = battery["seeds"]
    if (
        not seeds
        or len(set(seeds)) != len(seeds)
        or any(type(s) is not int or s < 0 for s in seeds)
    ):
        raise ValueError("Seeds must be distinct nonnegative integers")
    jobs = []
    for task in battery["tasks"]:
        config = compose(overrides={**components, "task": task["task"]})
        _apply(config, algorithm["settings"], METHOD_FIELDS)
        _apply(config, battery.get("settings", {}), TASK_FIELDS)
        _apply(config, task.get("settings", {}), TASK_FIELDS)
        if config.get("variable_length") is not None:
            if "variable_length" not in config.task:
                raise ValueError(
                    f"{task['name']} does not support variable-length generation"
                )
            config.task.variable_length = True
        config.algorithm_definition = definition
        evaluations = task.get("evaluations", [{"name": "default", "overrides": {}}])
        choices = []
        for policy in algorithm["policies"]:
            if any(
                not key.startswith("sampler.") for key in policy.get("overrides", {})
            ):
                raise ValueError(
                    "Algorithm policies may only override sampler settings"
                )
            for evaluation in evaluations:
                if any(
                    key.startswith("sampler.")
                    for key in evaluation.get("overrides", {})
                ):
                    raise ValueError(
                        "Task evaluations cannot change sampler definitions"
                    )
                choices.append(
                    InferenceJob(
                        policy.get("sampler"),
                        {
                            **policy.get("overrides", {}),
                            **evaluation.get("overrides", {}),
                            "evaluation.policy": f"{policy['name']}--{evaluation['name']}",
                        },
                    )
                )
        labels = [choice.overrides["evaluation.policy"] for choice in choices]
        if not labels or len(set(labels)) != len(labels):
            raise ValueError("Evaluation policies must have distinct names")
        for seed in seeds:
            cfg = copy.deepcopy(config)
            cfg.seed = seed
            cfg.benchmark = {
                "algorithm_id": algorithm_id,
                "algorithm_name": algorithm["name"],
                "battery_name": battery["name"],
                "task_id": task["name"],
                "training_seed": seed,
            }
            jobs.append(
                TrainingJob(
                    name=f"{task['name']}-seed-{seed}",
                    config=OmegaConf.to_container(cfg, resolve=False),
                    inference=choices,
                )
            )
    if not jobs or len({job.name for job in jobs}) != len(jobs):
        raise ValueError("Battery must contain distinct tasks")
    return jobs


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--algorithm",
        type=Path,
        default=ROOT / "configs/algorithm_definition/masked.yaml",
    )
    parser.add_argument(
        "--battery", type=Path, default=ROOT / "configs/battery/small_tasks.yaml"
    )
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--name", default="task-battery")
    parser.add_argument(
        "--storage", type=Path, default=Path.home() / ".cache/duo/batteries"
    )
    parser.add_argument("--max-concurrent", type=int, default=1)
    parser.add_argument("--wandb-project")
    parser.add_argument("--wandb-entity")
    args = parser.parse_args(argv)
    if args.max_concurrent < 1:
        parser.error("--max-concurrent must be positive")
    algorithm = OmegaConf.to_container(OmegaConf.load(args.algorithm), resolve=False)
    battery = OmegaConf.to_container(OmegaConf.load(args.battery), resolve=False)
    jobs = build_jobs(algorithm, battery)
    for job in jobs:
        if args.wandb_project:
            job.config["wandb"] = {
                "project": args.wandb_project,
                "group": args.name,
                "name": job.name,
                **({"entity": args.wandb_entity} if args.wandb_entity else {}),
            }
    print(
        f"{len(jobs)} training jobs; {sum(len(j.inference) for j in jobs)} evaluations"
    )
    if not args.run:
        print("Preview only. Pass --run to submit jobs to Ray.")
        return
    from ray import tune
    import ray
    from core.sweeps.queue import TrainingInferenceSearch, run_job

    storage = args.storage.expanduser().resolve()
    if (storage / args.name).exists():
        parser.error("Experiment exists; restore it with Ray or choose a new --name")
    os.environ.setdefault("RAY_ENABLE_UV_RUN_RUNTIME_ENV", "0")
    os.environ.setdefault("WANDB_DATA_DIR", str(Path.home() / ".cache/duo/wandb-data"))
    Path(os.environ["WANDB_DATA_DIR"]).mkdir(parents=True, exist_ok=True)
    temporary = Path.home() / ".cache/duo/ray"
    temporary.mkdir(parents=True, exist_ok=True)
    ray.init(_temp_dir=str(temporary))
    try:
        results = tune.Tuner(
            tune.with_resources(run_job, {"cpu": 4, "gpu": 1}),
            tune_config=tune.TuneConfig(
                search_alg=TrainingInferenceSearch(jobs),
                num_samples=-1,
                max_concurrent_trials=args.max_concurrent,
            ),
            run_config=tune.RunConfig(
                storage_path=str(storage), name=args.name, log_to_file=True
            ),
        ).fit()
        if results.num_errors:
            raise RuntimeError(
                f"{results.num_errors} jobs failed; inspect the Ray trial logs"
            )
    finally:
        ray.shutdown()


if __name__ == "__main__":
    main()

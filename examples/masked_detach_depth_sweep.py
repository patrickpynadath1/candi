"""Detached intermediate-token feedback: current depth versus first block."""
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
from examples.continuous_hybrid_feedback_sweep import build_jobs as old_jobs


def build_jobs(group):
    from examples.discrete_feedback_sweep import build_jobs as discrete_jobs
    base = next(job for job in discrete_jobs(group) if job.name == "masked-token")
    jobs = []
    for depth in (3, 0):
        config = OmegaConf.create(base.config)
        name = f"masked-token-detached-depth{depth}"
        config.token_feedback.detach = True
        config.token_feedback.depth = depth
        config.wandb.name = name
        jobs.append(TrainingJob(name, config=OmegaConf.to_container(config, resolve=False),
                                inference=base.inference))
    return jobs


def hint_evaluations(results, group):
    import subprocess
    source = Path("core/outputs") / (group + "-evaluation")
    source.mkdir(parents=True, exist_ok=True)
    rows = []
    for result in results:
        metrics = result.metrics
        paths = [Path(metrics["output_dir"]),
                 Path(metrics["artifact_root"]) / metrics.get("checkpoint_dir_name", "run")]
        saved = next((p for p in paths if (p / "results.json").exists()), None)
        if saved is None:
            continue  # Training trials contain checkpoints, not evaluation results.
        config = OmegaConf.load(saved / "config.yaml")
        sampler = config.sampler
        label = (f"eb_frozen_{sampler.entropy_bound:g}" if "entropy" in sampler._target_.lower()
                 else f"regular_{sampler.steps}")
        rows.append(dict(run=config.wandb.name, sampler=label, result_file=str(saved / "results.json")))
    assert len(rows) == 8, f"Expected eight evaluations, found {len(rows)}"
    (source / "summary.json").write_text(json.dumps(rows, indent=2))
    output = (Path("core/outputs") / (group + "-hints")).resolve()
    subprocess.run([sys.executable, "examples/hint_removal_sweep.py", "--prepare", "--run",
                    "--source", str(source.resolve()), "--output", str(output),
                    "--gpus", os.environ["CUDA_VISIBLE_DEVICES"]], check=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run",action="store_true")
    parser.add_argument("--name",default="sudoku-masked-detach-depth-48k-5k-20261005")
    parser.add_argument("--gpus",type=int,default=2)
    args=parser.parse_args()
    jobs=build_jobs(args.name)
    for job in jobs:print(job.name,flush=True)
    print(f"{len(jobs)} training jobs; {sum(len(j.inference) for j in jobs)} inference jobs",flush=True)
    if not args.run:return
    import ray
    from ray import tune
    from core.sweeps.queue import TrainingInferenceSearch, run_job
    storage = Path("core/outputs/ray").resolve()
    output = storage / args.name
    output.mkdir(parents=True, exist_ok=False)
    (output / "job-plan.json").write_text(json.dumps([asdict(j) for j in jobs], indent=2))
    ray.init(address="local", num_cpus=args.gpus * 4, num_gpus=args.gpus,
             object_store_memory=2 * 1024**3, include_dashboard=False)
    try:
        result = tune.Tuner(tune.with_resources(run_job, {"cpu": 4, "gpu": 1}),
            tune_config=tune.TuneConfig(search_alg=TrainingInferenceSearch(jobs), num_samples=-1,
                                       max_concurrent_trials=args.gpus),
            run_config=tune.RunConfig(storage_path=str(storage), name=args.name, log_to_file=True,
                                     failure_config=tune.FailureConfig(max_failures=1))).fit()
        print(result.get_dataframe())
        if result.num_errors:
            raise RuntimeError(f"{result.num_errors} jobs failed")
    finally:
        ray.shutdown()
    hint_evaluations(result, args.name)


if __name__ == "__main__":
    main()

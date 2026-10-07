"""Adapt the recorded 5k Sudoku full-hint sweep to scCANDI's three datasets.

Preview by default. Prepared metadata supplies exact gene/vocabulary dimensions.
Use --run only after preparing data; --write-plan persists a reviewable job list.
"""

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("RAY_ENABLE_UV_RUN_RUNTIME_ENV", "0")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from omegaconf import OmegaConf, open_dict

from core.data.single_cell import DATASETS
from core.sweeps import InferenceJob, TrainingJob, compose
from examples.adaptive_hybrid_feedback_sweep import continuous_config
from examples.continuous_hybrid_feedback_sweep import CONTINUOUS, DISCRETE


def variants():
    # Exactly the non-token rows in results/sudoku-hard-5k/full-hints/report.md,
    # plus plain SEDD. Continuous-only no-SC was not in that results table.
    for family in DISCRETE:
        for feedback in ("none", "hidden"):
            yield family, None, feedback
    for family in CONTINUOUS:
        yield family, "none", "hidden"
        for mixture in ("masked", "uniform"):
            for feedback in ("none", "hidden"):
                yield family, mixture, feedback
    yield "sedd", None, "none"


def inference_jobs(family, mixture, max_examples):
    def job(sampler=None, **options):
        return InferenceJob(
            sampler,
            {
                "evaluation.max_examples": max_examples,
                "evaluation.use_ema": True,
                "seed": 1,
                **options,
            },
        )

    jobs = [job(**{"sampler.steps": n}) for n in (32, 180)]
    if family == "sedd":
        return tuple(jobs)
    if family in CONTINUOUS and mixture == "none":
        jobs += [job(**{"sampler.steps": nfe - 1}) for nfe in (4, 8)]
    for bound in (0.1, 0.01):
        if family == "masked":
            jobs.append(
                job(
                    "entropy_bounded",
                    **{
                        "sampler.entropy_bound": bound,
                        "sampler.max_steps": 180,
                        "sampler.final_denoise": "greedy",
                    },
                )
            )
        elif family in ("uniform-ce", "uniform-plugin"):
            for freeze in (True, False):
                jobs.append(
                    job(
                        "uniform_entropy_bounded",
                        **{
                            "sampler.entropy_bound": bound,
                            "sampler.steps": 180,
                            "sampler.freeze": freeze,
                        },
                    )
                )
        else:
            for freeze in (True, False) if mixture == "uniform" else (True,):
                jobs.append(
                    job(
                        "parallel_entropy_bounded",
                        **{
                            "sampler.entropy_bound": bound,
                            "sampler.steps": 180,
                            "sampler.freeze": freeze,
                        },
                    )
                )
    return tuple(jobs)


def build_jobs(args, *, require_data=False):
    jobs = []
    for dataset in args.datasets:
        metadata_path = args.data_cache / dataset / "metadata.json"
        metadata = (
            json.loads(metadata_path.read_text()) if metadata_path.exists() else None
        )
        if metadata is None and require_data:
            raise FileNotFoundError(f"Prepare data before launching: {metadata_path}")
        task_cfg = compose(overrides={"task": "single_cell", "data": dataset})
        task_cfg.data.cache_dir = str(args.data_cache.resolve())
        if metadata:
            task_cfg.data.num_genes = metadata["num_genes"]
            task_cfg.data.vocab_size = metadata["vocab_size"]
        for mode in args.modes:
            for seed in args.seeds:
                for family, mixture, feedback in variants():
                    cfg = (
                        continuous_config(family, mixture)
                        if family in CONTINUOUS
                        else compose(
                            "sedd_absorb" if family == "sedd" else DISCRETE[family]
                        )
                    )
                    label = family + (
                        f"-{mixture}" if mixture not in (None, "none") else ""
                    )
                    name = f"{dataset}-{mode}-{label}-{feedback}-seed-{seed}"
                    with open_dict(cfg):
                        cfg.task = OmegaConf.to_container(task_cfg.task, resolve=False)
                        cfg.data = OmegaConf.to_container(task_cfg.data, resolve=False)
                        cfg.task.conditional = mode == "conditional"
                        cfg.task.length = int(task_cfg.data.num_genes) + int(
                            cfg.task.conditional
                        )
                        cfg.data.max_train = args.train_examples
                        cfg.seed = seed
                        cfg.trainer.max_steps = args.train_steps
                        cfg.trainer.accumulate_grad_batches = (
                            args.batch_size // args.microbatch_size
                        )
                        cfg.trainer.val_check_interval = 1.0
                        cfg.trainer.check_val_every_n_epoch = args.eval_every_epochs
                        cfg.trainer.enable_progress_bar = False
                        cfg.trainer.log_every_n_steps = 1
                        cfg.checkpoint.every_n_steps = 0
                        cfg.checkpoint.every_n_epochs = 1
                        cfg.checkpoint.save_top_k = 0
                        cfg.loader.global_batch_size = args.batch_size
                        cfg.optimizer.lr = args.learning_rate
                        cfg.loader.eval_batch_size = args.eval_batch_size
                        cfg.loader.num_workers = args.data_workers
                        cfg.callback.every_n_steps = 0
                        cfg.callback.every_n_epochs = args.eval_every_epochs
                        cfg.callback.evaluation_name = "validation"
                        cfg.callback.save_records = False
                        cfg.callback.subset_size = 128
                        cfg.self_conditioning.enabled = feedback == "hidden"
                        cfg.self_conditioning.source = "hidden"
                        cfg.self_conditioning.probability = 0.9
                        cfg.token_feedback.enabled = False
                        # Match Sudoku's backbone across families, including SEDD.
                        model = (
                            cfg.model.backbone if family in CONTINUOUS else cfg.model
                        )
                        model.hidden_size = 256 if args.model_size == "paper" else 512
                        model.n_blocks = 4 if args.model_size == "paper" else 8
                        model.n_heads = 4 if args.model_size == "paper" else 8
                        if family == "masked":
                            cfg.sampler = compose(
                                overrides={"sampler": "ancestral"}
                            ).sampler
                        cfg.sampler.steps = 180
                        cfg.wandb = (
                            None
                            if args.wandb_mode == "disabled"
                            else {
                                "project": args.wandb_project,
                                "group": args.name,
                                "name": name,
                                "job_type": "train",
                                "mode": args.wandb_mode,
                                **(
                                    {"entity": args.wandb_entity}
                                    if args.wandb_entity
                                    else {}
                                ),
                            }
                        )
                    if metadata:
                        OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
                    inference = tuple(
                        InferenceJob(j.sampler, {**j.overrides, "seed": seed})
                        for j in inference_jobs(family, mixture, args.max_examples)
                    )
                    jobs.append(
                        TrainingJob(
                            name,
                            config=OmegaConf.to_container(cfg, resolve=False),
                            inference=inference,
                        )
                    )
    return jobs


def parser():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--run", action="store_true")
    cli.add_argument("--resume", action="store_true")
    cli.add_argument("--write-plan", type=Path)
    cli.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    cli.add_argument(
        "--modes",
        nargs="+",
        choices=("unconditional", "conditional"),
        default=["unconditional", "conditional"],
    )
    cli.add_argument("--seeds", nargs="+", type=int, default=[1])
    cli.add_argument(
        "--data-cache", type=Path, default=Path("core/data_cache/single_cell")
    )
    cli.add_argument("--train-steps", type=int, default=5000)
    cli.add_argument("--train-examples", type=int, default=48000)
    cli.add_argument("--max-examples", type=int, default=2000)
    cli.add_argument("--batch-size", type=int, default=512)
    cli.add_argument("--learning-rate", type=float, default=3e-4)
    cli.add_argument("--eval-every-epochs", type=int, default=100)
    cli.add_argument("--microbatch-size", type=int, default=1)
    cli.add_argument("--eval-batch-size", type=int, default=1)
    cli.add_argument("--data-workers", type=int, default=2)
    cli.add_argument("--model-size", choices=("sudoku", "paper"), default="sudoku")
    cli.add_argument("--gpus", type=int, default=1)
    cli.add_argument("--storage", type=Path, default=Path("core/outputs/ray"))
    cli.add_argument("--name", default="single-cell-sudoku-matched-5k")
    cli.add_argument("--wandb-project", default="scCANDI")
    cli.add_argument("--wandb-entity")
    cli.add_argument(
        "--wandb-mode", choices=("online", "offline", "disabled"), default="disabled"
    )
    return cli


def main(argv=None):
    cli = parser()
    args = cli.parse_args(argv)
    if any(
        x <= 0
        for x in (
            args.train_steps,
            args.train_examples,
            args.max_examples,
            args.batch_size,
            args.microbatch_size,
            args.eval_batch_size,
            args.learning_rate,
            args.eval_every_epochs,
            args.gpus,
        )
    ):
        cli.error("Steps, counts and resources must be positive")
    if args.batch_size % args.microbatch_size or args.data_workers < 0:
        cli.error(
            "Batch size must be divisible by microbatch size; workers must be nonnegative"
        )
    if any(
        len(values) != len(set(values))
        for values in (args.datasets, args.modes, args.seeds)
    ):
        cli.error("Duplicate datasets, modes or seeds")
    if Path(args.name).name != args.name or args.name in (".", ".."):
        cli.error("--name must be one directory name")
    jobs = build_jobs(args, require_data=args.run)
    plan = [asdict(job) for job in jobs]
    if args.write_plan:
        args.write_plan.parent.mkdir(parents=True, exist_ok=True)
        args.write_plan.write_text(json.dumps(plan, indent=2) + "\n")
    print(
        f"{len(jobs)} training jobs; {sum(len(j.inference) for j in jobs)} dependent inference jobs"
    )
    print("22 variants per dataset/mode/seed; SEDD is analytic-only with no SC or EB.")
    print(
        "Regular: 32/180 steps; continuous-only: 4/8 NFE; EB: .1/.01, frozen and uniform revisable."
    )
    missing = [
        name
        for name in args.datasets
        if not (args.data_cache / name / "metadata.json").exists()
    ]
    if missing:
        print(
            f"Unprepared data: {', '.join(missing)}. Vocabulary remains ??? until preparation."
        )
    if not args.run:
        print("Plan only; no downloads or training. Use --run after preparing data.")
        return
    from hydra.utils import instantiate
    from ray import tune
    import ray
    from core.sweeps.queue import TrainingInferenceSearch, run_job

    # Validate every dataset/mode before launching GPU jobs.
    checked = set()
    for job in jobs:
        cfg = OmegaConf.create(job.config)
        key = (cfg.data.dataset, cfg.task.conditional)
        if key not in checked:
            instantiate(cfg.task).prepare_data()
            checked.add(key)
    output = args.storage.resolve() / args.name
    digest = hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest()
    manifest = output / "job-plan.json"
    if args.resume:
        if not tune.Tuner.can_restore(str(output)) or not manifest.exists():
            cli.error("No restorable sweep at this location")
        if json.loads(manifest.read_text())["fingerprint"] != digest:
            cli.error("Plan changed; use a new --name")
    else:
        output.mkdir(parents=True, exist_ok=False)
        manifest.write_text(json.dumps({"fingerprint": digest, "jobs": plan}, indent=2))
    ray.init(
        address="local",
        num_cpus=args.gpus * 4,
        num_gpus=args.gpus,
        object_store_memory=2 * 1024**3,
        include_dashboard=False,
    )
    try:
        trainable = tune.with_resources(run_job, {"cpu": 4, "gpu": 1})
        tuner = (
            tune.Tuner.restore(str(output), trainable=trainable)
            if args.resume
            else tune.Tuner(
                trainable,
                tune_config=tune.TuneConfig(
                    search_alg=TrainingInferenceSearch(jobs),
                    num_samples=-1,
                    max_concurrent_trials=args.gpus,
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
            raise RuntimeError(f"{results.num_errors} jobs failed")
    finally:
        ray.shutdown()


if __name__ == "__main__":
    main()

"""Backfill the same fixed GSM8K subset used by periodic training evaluations."""

import argparse
import json
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf
import ray


def evaluate_checkpoint(
    config_path,
    checkpoint,
    destination,
    subset_size=128,
    seed=42,
    steps=None,
    temperature=None,
):
    import torch
    from hydra.utils import instantiate
    from core.training.module import TrainingModule
    from core.training.checkpoints import load_weights
    from core.evaluation.generation import evaluate
    from core.tasks.sandbox import execute_python

    torch.set_num_threads(4)
    probe = execute_python(
        "def simple_math_problem():\n    return 2+2", entrypoint="simple_math_problem"
    )
    if probe.result != 4 or probe.sandbox_error:
        raise RuntimeError("GSM8K code-scoring sandbox is not functional")
    cfg = OmegaConf.load(config_path)
    if steps is not None:
        cfg.sampler.steps = steps
    if temperature is not None:
        cfg.parameterization.temperature = temperature
    module = TrainingModule(cfg)
    step = load_weights(module, checkpoint)
    module.to("cuda:0")
    task = instantiate(cfg.task)
    task.prepare_evaluation("gsm8k")
    dataset = task.evaluation_dataset("gsm8k")
    indices = sorted(
        np.random.default_rng(seed)
        .choice(len(dataset), min(len(dataset), subset_size), replace=False)
        .tolist()
    )
    torch.manual_seed(seed)
    result = evaluate(
        module,
        task,
        cfg.sampler,
        batch_size=4,
        indices=indices,
        precision=cfg.trainer.precision,
        use_ema=True,
    )
    result.update(
        checkpoint_step=step,
        checkpoint=str(checkpoint),
        indices=indices,
        evaluation_seed=seed,
        use_ema=True,
        sampler=OmegaConf.to_container(cfg.sampler, resolve=True),
        temperature=cfg.parameterization.temperature,
    )
    destination = Path(destination)
    temporary = destination.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, indent=2))
    temporary.replace(destination)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--temperature", type=float)
    parser.add_argument(
        "--direct",
        action="store_true",
        help="Use the GPU assigned by the caller without Ray scheduling",
    )
    parser.add_argument("--run-suffix", default="gsm8k-checkpoints")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    config = OmegaConf.load(args.run / "config.yaml")
    # Include every saved periodic checkpoint and separately score best/last.
    checkpoints = (
        [args.checkpoint]
        if args.checkpoint
        else sorted((args.run / "checkpoints").glob("*.ckpt"))
    )
    if not checkpoints:
        raise RuntimeError("No saved checkpoints found")
    if not args.direct:
        ray.init(address="auto")
        worker = ray.remote(num_gpus=1, num_cpus=4, max_retries=0)(evaluate_checkpoint)
    pending, results = {}, []
    for checkpoint in checkpoints:
        output = args.output / f"{checkpoint.stem}.json"
        if output.exists():
            results.append(json.loads(output.read_text()))
        elif args.direct:
            results.append(
                evaluate_checkpoint(
                    str(args.run / "config.yaml"),
                    str(checkpoint),
                    str(output),
                    steps=args.steps,
                    temperature=args.temperature,
                )
            )
        else:
            ref = worker.remote(
                str(args.run / "config.yaml"),
                str(checkpoint),
                str(output),
                steps=args.steps,
                temperature=args.temperature,
            )
            pending[ref] = checkpoint.name
    while pending:
        ready, _ = ray.wait(list(pending), num_returns=1)
        ref = ready[0]
        result = ray.get(ref)
        print(
            f"{pending.pop(ref)}: step={result['checkpoint_step']}, accuracy={result['accuracy']:.4f}, n={result['num_total']}",
            flush=True,
        )
        results.append(result)
    # Historical points go in a separate run, never backdate the live training history.
    import wandb
    from core.evaluation.reporting import scalar_metrics

    options = dict(config.wandb)
    options.update(
        id=options["id"] + "-" + args.run_suffix,
        name=options["name"] + "/" + args.run_suffix,
        job_type="evaluate",
        resume="allow",
    )
    with wandb.init(**options, dir=str(args.output)) as tracking:
        tracking.define_metric("checkpoint_step")
        tracking.define_metric("gsm8k/*", step_metric="checkpoint_step")
        for result in sorted(
            results, key=lambda r: (r["checkpoint_step"], r["checkpoint"])
        ):
            tracking.log(
                {
                    "checkpoint_step": result["checkpoint_step"],
                    **{
                        f"gsm8k/{k}": v
                        for k, v in scalar_metrics(result)
                        if k
                        not in (
                            "checkpoint_step",
                            "checkpoint",
                            "evaluation_seed",
                            "use_ema",
                        )
                        and not k.startswith(("sampler/", "indices/"))
                    },
                }
            )
        artifact = wandb.Artifact(
            "tinygsm-masked-sc-" + args.run_suffix, type="evaluation"
        )
        samples = wandb.Table(
            columns=[
                "checkpoint_step",
                "checkpoint",
                "index",
                "prompt",
                "response",
                "response_ground_truth",
                "correct",
            ]
        )
        for result in sorted(results, key=lambda r: r["checkpoint_step"]):
            for row in result["records"]:
                samples.add_data(
                    result["checkpoint_step"],
                    Path(result["checkpoint"]).name,
                    row["index"],
                    row["prompt"],
                    row["response"],
                    row["response_ground_truth"],
                    row["correct"],
                )
        tracking.log({"gsm8k/generated_answers": samples})
        for path in sorted(args.output.glob("*.json")):
            artifact.add_file(str(path), name=path.name)
        tracking.log_artifact(artifact)
        url = tracking.url
    (args.output / "summary.json").write_text(
        json.dumps(
            {
                "wandb_url": url,
                "results": [
                    {
                        "checkpoint": r["checkpoint"],
                        "step": r["checkpoint_step"],
                        "accuracy": r["accuracy"],
                        "num_total": r["num_total"],
                    }
                    for r in results
                ],
            },
            indent=2,
        )
    )
    print(f"Results: {url}", flush=True)
    if not args.direct:
        ray.shutdown()


if __name__ == "__main__":
    main()

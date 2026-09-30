"""Compare current/EMA checkpoint weights with identical SDM sampling seeds."""

import argparse
import json
from pathlib import Path
import sys

import torch
import wandb
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import main  # noqa: E402 (register Hydra resolvers)
from algos.simplex import SimplexDiffusion  # noqa: E402
from algos.simplex_utils import corruption_parameters, log_dirichlet_sample  # noqa: E402
from dataloader import get_tokenizer  # noqa: E402
from tasks.sudoku import get_sudoku_dataset  # noqa: E402
from tasks.sudoku_callback import (  # noqa: E402
    generate_sudoku_records,
    summarize_sudoku_records,
)


@torch.no_grad()
def audit(args):
    config = OmegaConf.load(args.config)
    config.eval.checkpoint_path = str(args.checkpoint.resolve())
    tokenizer = get_tokenizer(config)
    dataset = get_sudoku_dataset(config, tokenizer)["validation"]
    count = min(args.examples, len(dataset))
    batch = torch.stack([dataset[i]["input_ids"] for i in range(count)])
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    step = checkpoint["global_step"]
    decay = checkpoint["ema"]["decay"]
    report = {"checkpoint_step": step, "ema_decay": decay, "comparisons": {}}
    if checkpoint["ema"]["num_updates"] is None:
        report["ema_initialization_coefficient"] = decay**step
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with wandb.init(
        dir=str(args.output.parent),
        project=args.project,
        entity=args.entity,
        mode="online",
        name=args.name,
        job_type="sampler-diagnostic",
        config={
            "checkpoint": str(args.checkpoint),
            "examples": count,
            "steps": args.steps,
            "checkpoint_step": step,
            "seed": 123,
        },
    ) as run:
        for weights in ("current", "ema"):
            config.eval.disable_ema = weights == "current"
            model = main._load_from_checkpoint(SimplexDiffusion, config, tokenizer)
            model.to(args.device).eval()
            x = batch.to(model.device)
            clean = torch.nn.functional.one_hot(x, model.vocab_size).double()
            target = torch.arange(x.shape[1], device=x.device) >= tokenizer.prompt_len
            metrics = {}
            for time in (0.01, 0.5, 0.99, 1.0):
                torch.manual_seed(123)
                t = torch.full((count,), time, device=x.device, dtype=torch.float64)
                beta = corruption_parameters(
                    clean, t, model.concentration(t), model.reference.double()
                )
                state = torch.where(
                    target[None, :, None], log_dirichlet_sample(beta), clean.log()
                )
                pred = model(state, t.float())
                metrics[f"denoise_t{time}/accuracy"] = (
                    (pred.argmax(-1)[:, target] == x[:, target]).float().mean().item()
                )
                metrics[f"denoise_t{time}/nll"] = (
                    -pred.gather(-1, x[..., None]).squeeze(-1)[:, target].mean().item()
                )
            for churn in (0.0, 0.2, 1.0):
                torch.manual_seed(123)
                records = generate_sudoku_records(
                    model,
                    dataset,
                    tokenizer,
                    count,
                    args.steps,
                    range(count),
                    sampler_overrides={"churn": churn},
                )
                summary = summarize_sudoku_records(records)
                key = f"{weights}/churn{churn}"
                report["comparisons"][key] = summary
                for field in ("solve_rate", "cell_accuracy", "valid_grid_rate"):
                    metrics[f"churn{churn}/{field}"] = summary[field]
                run.log(
                    {
                        f"{key}/examples": wandb.Table(
                            columns=["generated", "ground_truth"],
                            data=[
                                [r["generated"], r["ground_truth"]] for r in records[:8]
                            ],
                        )
                    }
                )
            report[weights] = metrics
            run.log({f"{weights}/{key}": value for key, value in metrics.items()})
            print(weights, json.dumps(metrics), flush=True)
            del model
        args.output.write_text(json.dumps(report, indent=2))
        artifact = wandb.Artifact("simplex-sampler-audit", type="diagnostic")
        artifact.add_file(str(args.output))
        run.log_artifact(artifact)
        print(run.url, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--examples", type=int, default=32)
    parser.add_argument("--steps", type=int, default=180)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--project", default="simplex-sudoku")
    parser.add_argument(
        "--entity", help="W&B entity; defaults to the authenticated account"
    )
    parser.add_argument("--name", default="simplex-sampler-current-vs-ema")
    audit(parser.parse_args())

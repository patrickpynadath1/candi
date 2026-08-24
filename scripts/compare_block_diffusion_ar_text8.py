"""Matched Text8 training run for AR vs zero-noise block diffusion."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import torch
from hydra import compose, initialize_config_dir

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import main  # noqa: F401 - register OmegaConf resolvers
import dataloader
from algos import algo_registry


def evaluate(model, loader, batches):
    model.eval()
    total_nll = 0.0
    total_tokens = 0.0
    with torch.no_grad():
        iterator = iter(loader)
        for _ in range(batches):
            batch = next(iterator)
            tokens = batch["input_ids"].to(model.device)
            valid = batch["attention_mask"].to(model.device)
            result = model._loss(tokens, valid)
            total_nll += float(result.nlls)
            total_tokens += float(result.num_tokens)
    model.train()
    return total_nll / total_tokens


def main_experiment(args):
    root = ROOT
    overrides = [
        f"algo={args.algo}",
        "data=text8",
        "model=tiny",
        "strategy=ddp",
        "training.ema=0",
        "loader.num_workers=0",
        f"loader.global_batch_size={args.batch_size}",
        f"loader.eval_global_batch_size={args.batch_size}",
        f"loader.batch_size={args.batch_size}",
        f"loader.eval_batch_size={args.batch_size}",
        "trainer.devices=1",
        "trainer.num_nodes=1",
        "trainer.accumulate_grad_batches=1",
        f"model.length={args.length}",
        f"model.hidden_size={args.hidden_size}",
        f"model.cond_dim={args.hidden_size}",
        f"model.n_blocks={args.layers}",
        f"model.n_heads={args.heads}",
        "model.dropout=0.0",
        "+model.normalize_embeddings=true",
    ]
    if args.algo == "block_diffusion":
        overrides.extend(
            [
                f"algo.block_size={args.block_size}",
                "algo.max_noise_level=0.0",
                "algo.noise_curriculum_steps=0",
                "sampling.predictor=block_diffusion",
            ]
        )
    else:
        overrides.append("sampling.predictor=ar")

    with initialize_config_dir(config_dir=str(root / "configs"), version_base=None):
        config = compose(config_name="config_text8_frontier", overrides=overrides)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    tokenizer = dataloader.get_tokenizer(config)
    train_loader, valid_loader = dataloader.get_dataloaders(config, tokenizer)
    model = algo_registry.get(config.algo.name)(config, tokenizer).cuda().train()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-8
    )

    history = []
    initial = evaluate(model, valid_loader, args.validation_batches)
    history.append({"step": 0, "valid_nll": initial, "valid_bpc": initial / math.log(2)})
    print(json.dumps(history[-1]), flush=True)

    iterator = iter(train_loader)
    for step in range(1, args.steps + 1):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            batch = next(iterator)
        tokens = batch["input_ids"].cuda(non_blocking=True)
        valid = batch["attention_mask"].cuda(non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        result = model._loss(tokens, valid, train_mode=True)
        result.loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step == 1 or step % args.eval_every == 0 or step == args.steps:
            valid_nll = evaluate(model, valid_loader, args.validation_batches)
            record = {
                "step": step,
                "train_nll": float(result.loss.detach()),
                "valid_nll": valid_nll,
                "valid_bpc": valid_nll / math.log(2),
            }
            history.append(record)
            print(json.dumps(record), flush=True)

    output = {
        "algo": args.algo,
        "seed": args.seed,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "sequence_length": args.length,
        "hidden_size": args.hidden_size,
        "layers": args.layers,
        "heads": args.heads,
        "block_size": args.block_size if args.algo == "block_diffusion" else None,
        "max_noise_level": 0.0 if args.algo == "block_diffusion" else None,
        "history": history,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", choices=("ar", "block_diffusion"), required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--eval-every", type=int, default=200)
    parser.add_argument("--validation-batches", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--length", type=int, default=128)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--block-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=17)
    main_experiment(parser.parse_args())

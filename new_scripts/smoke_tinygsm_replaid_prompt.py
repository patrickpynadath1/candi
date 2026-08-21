#!/usr/bin/env python3
"""Run a short conditional RePlaid generation without requiring a checkpoint."""

import argparse
from pathlib import Path
import sys

import torch
from hydra import compose, initialize

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import dataloader
import main  # noqa: F401 - register config resolvers
from algos import algo_registry


def main_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--steps", type=int, default=2)
    args = parser.parse_args()
    with initialize(version_base=None, config_path="../configs"):
        config = compose(
            config_name="config",
            overrides=[
                "+experiment=tinygsm_replaid_50m",
                "trainer.devices=1",
                "training.ema=0",
            ],
        )
    tokenizer = dataloader.get_tokenizer(config)
    model = algo_registry.get("replaid")(config, tokenizer).to(args.device).eval()
    encoded = tokenizer(
        "Write Python to solve: Alice has 3 apples and buys 2 more.\n",
        add_special_tokens=False,
    ).input_ids
    prefix = torch.tensor(
        [[tokenizer.bos_token_id, *encoded]], dtype=torch.long, device=args.device
    )
    prompt_tokens = torch.zeros(
        1, model.num_tokens, dtype=torch.long, device=args.device
    )
    prompt_mask = torch.zeros_like(prompt_tokens, dtype=torch.bool)
    prompt_tokens[:, : prefix.shape[1]] = prefix
    prompt_mask[:, : prefix.shape[1]] = True

    with torch.no_grad():
        generated = model.generate_samples(
            1,
            num_steps=args.steps,
            prompt_tokens=prompt_tokens,
            prompt_mask=prompt_mask,
        )
    if not torch.equal(generated[prompt_mask], prompt_tokens[prompt_mask]):
        raise RuntimeError("conditional sampler changed prompt tokens")
    print(f"prompt_tokens={int(prompt_mask.sum())}")
    print(f"generated_shape={tuple(generated.shape)}")
    print("prompt_preserved=true")


if __name__ == "__main__":
    main_cli()

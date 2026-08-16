#!/usr/bin/env python3
"""Instantiate a TinyGSM experiment and report its exact parameterization."""

import argparse
from pathlib import Path
import sys

from hydra import compose, initialize

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import dataloader
import main  # noqa: F401 - registers OmegaConf resolvers
from algos import algo_registry


def main_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", default="tinygsm_ar_20m")
    args = parser.parse_args()
    with initialize(version_base=None, config_path="../configs"):
        config = compose(
            config_name="config",
            overrides=[f"+experiment={args.experiment}", "trainer.devices=1"],
        )
    tokenizer = dataloader.get_tokenizer(config)
    model = algo_registry.get(config.algo.name)(config, tokenizer)
    trainable = sum(parameter.numel() for parameter in model.parameters())
    tied = (
        model.backbone.output_layer.linear.weight
        is model.backbone.vocab_embed.embedding
    )
    print(f"experiment={args.experiment}")
    print(f"parameters={trainable:,}")
    print(f"vocabulary={model.vocab_size:,}")
    print(f"tied_embeddings={tied}")
    print(f"layers={config.model.n_blocks}")
    print(f"hidden_size={config.model.hidden_size}")
    print(f"context={config.model.length}")


if __name__ == "__main__":
    main_cli()

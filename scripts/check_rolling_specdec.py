"""Smoke-test rolling speculative decoding from a trained block checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import dataloader
from algos import algo_registry
from main import _load_from_checkpoint


def main(args):
    config = OmegaConf.load(args.config)
    config.eval.checkpoint_path = args.checkpoint
    config.eval.disable_ema = args.disable_ema
    config.sampling.predictor = args.sampler
    config.algo.rolling_update = args.update
    config.algo.renoise_correlation = args.noise_correlation
    config.algo.temp = args.temperature
    OmegaConf.update(
        config, "algo.euler_tokens_per_step", args.tokens_per_step, force_add=True
    )
    OmegaConf.update(
        config, "algo.euler_nfe_budget", args.nfe_budget, force_add=True
    )
    OmegaConf.update(
        config, "algo.rolling_warmup_steps", args.warmup_steps, force_add=True
    )
    OmegaConf.update(
        config,
        "algo.rolling_refinement_steps",
        args.refinement_steps,
        force_add=True,
    )
    OmegaConf.update(
        config,
        "algo.rolling_x0_projection",
        args.x0_projection,
        force_add=True,
    )

    tokenizer = dataloader.get_tokenizer(config)
    model_cls = algo_registry.get(config.algo.name)
    model = _load_from_checkpoint(model_cls, config, tokenizer).cuda().eval()
    torch.manual_seed(args.seed)
    result = model.generate_samples(
        args.num_samples,
        return_result=True,
        sampler_name=args.sampler,
    )
    samples = result.samples
    if samples.shape != (args.num_samples, config.model.length):
        raise RuntimeError(f"unexpected sample shape: {tuple(samples.shape)}")
    if samples.min() < 0 or samples.max() >= model.vocab_size:
        raise RuntimeError("sample contains an out-of-vocabulary token")
    if model.mask_index is not None and (samples == model.mask_index).any():
        raise RuntimeError("sample contains the reserved mask token")
    if not all(torch.isfinite(parameter).all() for parameter in model.parameters()):
        raise RuntimeError("model contains non-finite parameters")

    report = {
        "update": args.update,
        "temperature": args.temperature,
        "sampler": args.sampler,
        "shape": list(samples.shape),
        "token_min": int(samples.min()),
        "token_max": int(samples.max()),
        "metadata": result.metadata,
        "text": tokenizer.batch_decode(samples),
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--update", choices=("euler", "renoise"), required=True)
    parser.add_argument(
        "--sampler",
        choices=("block_diffusion", "block_diffusion_euler"),
        default="block_diffusion",
    )
    parser.add_argument("--num-samples", type=int, default=4)
    parser.add_argument("--noise-correlation", type=float, default=0.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--tokens-per-step", type=int, default=4)
    parser.add_argument("--nfe-budget", type=int)
    parser.add_argument("--warmup-steps", type=int, default=8)
    parser.add_argument("--refinement-steps", type=int, default=0)
    parser.add_argument(
        "--x0-projection", choices=("soft", "argmax"), default="soft"
    )
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--disable-ema", action="store_true")
    main(parser.parse_args())

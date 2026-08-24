#!/usr/bin/env python3
"""Dump deterministic first-step MDLM tensors from upstream or local code."""

import argparse
import os
import sys
from pathlib import Path

import torch
from hydra import compose, initialize_config_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("implementation", choices=("upstream", "local"))
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    root = Path("/home/patrick/s-flm" if args.implementation == "upstream" else "/home/patrick/duo")
    sys.path.insert(0, str(root))
    os.chdir(root)

    if args.implementation == "upstream":
        import algo
        import dataloader

        with initialize_config_dir(version_base=None, config_dir=str(root / "configs")):
            config = compose(config_name="config", overrides=[
                "data=gsm8k-test",
                "data.tokenizer_name_or_path=HuggingFaceTB/SmolLM-135M",
                "model=small", "model.length=512", "algo=mdlm",
                "sampler=ancestral", "sampler.temperature=0.1", "sampler.steps=32",
                "eval.checkpoint_path=/home/patrick/.cache/discrete_diffusion/pretrained/s-flm/tinygsm/mdlm.ckpt",
                "trainer.devices=1",
            ])
        tokenizer = dataloader.get_tokenizer(config)
        model = algo.MDLM.load_from_checkpoint(
            config.eval.checkpoint_path, tokenizer=tokenizer, config=config, strict=True
        ).cuda()
        model._eval_mode()
        prompt = [tokenizer.bos_token_id] + tokenizer(
            "A boy has 5 cards.\n", add_special_tokens=False
        ).input_ids
        xt = model.prior_sample(1, model.num_tokens)
        xt[0, :len(prompt)] = torch.tensor(prompt, device=xt.device)
        t = torch.ones(1, 1, device=xt.device)
        _, alpha = model.noise(t)
        sigma = model._sigma_from_alphat(alpha)
        import samplers
        logits = model.forward(
            xt=xt, sigma=sigma,
            context=samplers.AncestralContext(temperature=0.1),
        )
    else:
        import algos
        import dataloader
        import main as local_main

        with initialize_config_dir(version_base=None, config_dir=str(root / "configs")):
            config = compose(config_name="config", overrides=[
                "data=gsm8k-test-sflm", "model=small", "model.length=512",
                "+model.sflm_compat=true", "algo=mdlm", "algo.temp=0.1",
                "sampling.predictor=ancestral", "sampling.steps=32",
                "eval.checkpoint_path=/home/patrick/.cache/discrete_diffusion/pretrained/s-flm/tinygsm/mdlm.ckpt",
            ])
        tokenizer = dataloader.get_tokenizer(config)
        model = local_main._load_from_checkpoint(
            algos.algo_registry.get("mdlm"), config, tokenizer
        ).cuda()
        model._eval_mode()
        prompt = [tokenizer.bos_token_id] + tokenizer(
            "A boy has 5 cards.\n", add_special_tokens=False
        ).input_ids
        xt = model.prior_sample(1, model.num_tokens)
        xt[0, :len(prompt)] = torch.tensor(prompt, device=xt.device)
        t = torch.ones(1, 1, device=xt.device)
        _, alpha = model.noise(t)
        sigma = model._sigma_from_alphat(alpha)
        logits = model.forward(xt, sigma)

    state = model.state_dict()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "logits": logits[0, len(prompt):len(prompt) + 4].float().cpu(),
        "vocab_embedding": state["backbone.vocab_embed.embedding"][:4, :16].float().cpu(),
        "output_weight": state["backbone.output_layer.linear.weight"][:4, :16].float().cpu(),
        "prompt": prompt,
    }, args.output)
    print(args.output)


if __name__ == "__main__":
    main()

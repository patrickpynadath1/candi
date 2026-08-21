"""Strict deterministic memorization test for the base RePlaid method."""

import argparse
from pathlib import Path
import sys

import torch
from hydra import compose, initialize_config_dir

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import main  # noqa: F401 -- register Hydra resolvers
from algos import algo_registry
from dataloader import get_tokenizer


TEXT = """Ryan plants 2 flowers a day for 15 days and 5 do not grow.
def simple_math_problem() -> int:
    return 2 * 15 - 5
"""


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--length", type=int, default=128)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def component_values(model, tokens, valid, gamma, gamma_prime, noise):
    clean = model.backbone.embed_tokens(tokens)
    z_t = model.q_sample(clean, gamma, noise=noise)
    with torch.no_grad():
        bootstrap = model.denoise(z_t, gamma.float())
        self_condition = model.backbone.embed_tokens(bootstrap.exp())
    log_probs = model.denoise(z_t, gamma.float(), self_condition)
    predicted = model.backbone.embed_tokens(log_probs.exp())
    ce = -log_probs.gather(-1, tokens.unsqueeze(-1)).squeeze(-1).double()
    mse = (predicted - clean).double().square().sum(-1)
    weights = valid.double()

    recon = (ce[0] * weights[0]).sum() / weights[0].sum().clamp_min(1)
    # SNR = exp(-gamma), hence SNR' = -gamma' * exp(-gamma).  The
    # diffusion ELBO integrand is -0.5 * SNR' * squared_error.
    snr_prime = -gamma_prime[1:] * torch.exp(-gamma[1:])
    diffusion_weight = -0.5 * snr_prime
    diffusion = (
        diffusion_weight[:, None] * mse[1:] * weights[1:]
    ).sum() / weights[1:].sum().clamp_min(1)
    mean_ce = (ce * weights).sum() / weights.sum().clamp_min(1)
    mean_mse = (mse * weights).sum() / weights.sum().clamp_min(1)

    gamma1 = gamma[-1]
    alpha1_sq = torch.sigmoid(-gamma1)
    sigma1_sq = 1 - alpha1_sq
    prior = 0.5 * (
        alpha1_sq * clean.double().square().sum(-1)
        + model.config.algo.embedding_dim
        * (sigma1_sq - 1 - torch.log(sigma1_sq))
    )
    prior = (prior * weights).sum() / weights.sum().clamp_min(1)
    return recon, diffusion, mean_ce, mean_mse, prior


def main_strict():
    args = parse_args()
    torch.manual_seed(1)
    with initialize_config_dir(
        config_dir=str(REPO_ROOT / "configs"), version_base=None
    ):
        config = compose(
            config_name="config",
            overrides=[
                "algo=replaid",
                "data=tiny-gsm",
                "model=tiny",
                "training.ema=0",
                f"model.length={args.length}",
                "model.hidden_size=128",
                "model.cond_dim=64",
                "model.n_blocks=2",
                "model.n_heads=4",
                "model.dropout=0",
                "optim.lr=3e-4",
                "optim.weight_decay=0",
                "lr_scheduler=constant_warmup",
                "lr_scheduler.num_warmup_steps=25",
                "sampling.predictor=replaid_ddpm",
            ],
        )
    tokenizer = get_tokenizer(config)
    model = algo_registry.get("replaid")(config, tokenizer).to(args.device)
    for parameter in model.gamma_schedule.parameters():
        parameter.requires_grad_(False)

    encoded = tokenizer(
        [TEXT] * args.batch_size,
        padding="max_length",
        truncation=True,
        max_length=args.length,
        return_tensors="pt",
    )
    tokens = encoded.input_ids.to(args.device)
    valid = encoded.attention_mask.to(args.device)
    times = torch.linspace(0, 1, args.batch_size, device=args.device)
    with torch.no_grad():
        gamma, gamma_prime = model._gamma(times)
    noise = torch.randn(
        args.batch_size,
        args.length,
        config.algo.embedding_dim,
        device=args.device,
    )
    optimizers, scheduler_configs = model.configure_optimizers()
    optimizer = optimizers[0]
    scheduler = scheduler_configs[0]["scheduler"]

    with torch.no_grad():
        initial = component_values(
            model, tokens, valid, gamma, gamma_prime, noise
        )
    for step in range(1, args.steps + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        recon, diffusion, mean_ce, mean_mse, prior = component_values(
            model, tokens, valid, gamma, gamma_prime, noise
        )
        loss = recon + diffusion
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        model.backbone.project_embeddings_()
        if step == 1 or step % 250 == 0:
            print(
                f"step={step} recon={recon.item():.8f} "
                f"diffusion={diffusion.item():.8f} ce={mean_ce.item():.8f} "
                f"mse={mean_mse.item():.8f} prior={prior.item():.8f}",
                flush=True,
            )

    with torch.no_grad():
        final = component_values(model, tokens, valid, gamma, gamma_prime, noise)
    labels = ("recon", "diffusion", "ce", "mse", "prior")
    for label, before, after in zip(labels, initial, final):
        print(
            f"{label}: initial={before.item():.10f} final={after.item():.10f}",
            flush=True,
        )
    if final[0] >= 1e-3 or final[3] >= 1e-3:
        raise RuntimeError(
            "strict overfit did not drive reconstruction loss and MSE near zero"
        )


if __name__ == "__main__":
    main_strict()

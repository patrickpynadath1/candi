"""Overfit a small RePlaid model on a fixed TinyGSM-style batch."""

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


EXAMPLES = (
    """Diego saves $100 each month for a $500 phone.\n"
    "def simple_math_problem() -> int:\n    return 500 // 100\n""",
    """Ryan plants 2 flowers daily for 15 days and 5 fail to grow.\n"
    "def simple_math_problem() -> int:\n    return 2 * 15 - 5\n""",
    """Six potatoes make 36 hash browns. Find the amount from 96 potatoes.\n"
    "def simple_math_problem() -> int:\n    return (36 // 6) * 96\n""",
    """Three rose bushes have 25 roses each and every rose has 8 thorns.\n"
    "def simple_math_problem() -> int:\n    return 3 * 25 * 8\n""",
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--length", type=int, default=128)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def fixed_batch(tokenizer, batch_size, length, device):
    texts = [EXAMPLES[index % len(EXAMPLES)] for index in range(batch_size)]
    encoded = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=length,
        return_tensors="pt",
    )
    return encoded.input_ids.to(device), encoded.attention_mask.to(device)


@torch.no_grad()
def noisy_metrics(model, tokens, valid, noise):
    model.eval()
    # Hold the channel fixed while the learned training schedule changes.
    gamma = torch.full(
        (tokens.shape[0],), 2.0, device=tokens.device, dtype=torch.float64
    )
    clean = model.backbone.embed_tokens(tokens)
    z_t = model.q_sample(clean, gamma, noise=noise)
    bootstrap = model.denoise(z_t, gamma.float())
    self_condition = model.backbone.embed_tokens(bootstrap.exp())
    log_probs = model.denoise(z_t, gamma.float(), self_condition)
    predicted = model.backbone.embed_tokens(log_probs.exp())
    ce = -log_probs.gather(-1, tokens.unsqueeze(-1)).squeeze(-1)
    mse = (predicted - clean).square().sum(-1)
    denominator = valid.sum().clamp_min(1)
    return (
        (ce * valid).sum().item() / denominator.item(),
        (mse * valid).sum().item() / denominator.item(),
    )


def main_overfit():
    args = parse_args()
    torch.manual_seed(1)
    config_dir = str(REPO_ROOT / "configs")
    with initialize_config_dir(config_dir=config_dir, version_base=None):
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
    model.train()
    tokens, valid = fixed_batch(
        tokenizer, args.batch_size, args.length, args.device
    )
    fixed_noise = torch.randn(
        *tokens.shape, config.algo.embedding_dim, device=args.device
    )
    optimizers, scheduler_configs = model.configure_optimizers()
    optimizer = optimizers[0]
    scheduler = scheduler_configs[0]["scheduler"]

    initial_ce, initial_mse = noisy_metrics(
        model, tokens, valid, fixed_noise
    )
    for step in range(1, args.steps + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        result = model._loss(tokens, valid, train_mode=True)
        result.loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        model.backbone.project_embeddings_()
        if step == 1 or step % 100 == 0:
            ce, mse = noisy_metrics(model, tokens, valid, fixed_noise)
            print(
                f"step={step} nelbo={result.loss.item():.6f} "
                f"noisy_ce={ce:.6f} noisy_mse={mse:.6f}",
                flush=True,
            )
    final_ce, final_mse = noisy_metrics(model, tokens, valid, fixed_noise)
    print(
        f"initial_noisy_ce={initial_ce:.6f} final_noisy_ce={final_ce:.6f} "
        f"initial_noisy_mse={initial_mse:.6f} final_noisy_mse={final_mse:.6f}",
        flush=True,
    )
    if not (final_ce < initial_ce * 0.5 and final_mse < initial_mse * 0.9):
        raise RuntimeError(
            "fixed-batch overfit did not reduce CE by 50% and MSE by 10%"
        )


if __name__ == "__main__":
    main_overfit()

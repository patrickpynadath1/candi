#!/usr/bin/env python3
"""Small-set overfit and device profiling; never a benchmark result."""

import argparse
import json
from pathlib import Path
import sys
import time
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import main  # noqa: F401 -- registers Hydra resolvers
from algos import algo_registry
from tasks.reasoning import get_reasoning_tokenizer, load_prepared
from reasoning_ablation import ROOT, overrides


def pilot(
    matrix, profiles, name, variant, prepared, output, steps, device, production=False
):
    job = dict(profile=name, variant=variant, mode="hybrid", seed=1)
    _, values = overrides(matrix, profiles, job, output, prepared, smoke=not production)
    values += ["training.ema=0"]
    if not production:
        values += ["model.hidden_size=32", "model.cond_dim=32", "data.limit_train=2"]
    else:
        values += [f"data.limit_train={profiles[name].batch_size}"]
    with initialize_config_dir(config_dir=str(ROOT / "configs"), version_base=None):
        config = compose(config_name="config", overrides=values)
    tok = get_reasoning_tokenizer(profiles[name].task)
    data = load_prepared(config, tok, "train")
    ids = torch.stack([x["input_ids"] for x in data]).to(device)
    mask = torch.stack([x["attention_mask"] for x in data]).to(device)
    torch.manual_seed(1)
    model = algo_registry.get(config.algo.name)(config, tok).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003)

    def measure():
        model.eval()
        losses = []
        with (
            torch.no_grad(),
            torch.random.fork_rng(
                devices=[torch.cuda.current_device()] if device == "cuda" else []
            ),
        ):
            for seed in range(5):
                torch.manual_seed(seed)
                losses.append(float(model._loss(ids, mask).loss))
        return sum(losses) / len(losses)

    initial = measure()
    model.train()
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    started = time.perf_counter()
    for _ in range(steps):
        optimizer.zero_grad()
        loss = model._loss(ids, mask).loss
        if not torch.isfinite(loss):
            raise ValueError("nonfinite training loss")
        loss.backward()
        optimizer.step()
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    final = measure()
    model.eval()
    result = model.generate_samples(
        len(ids),
        num_steps=16,
        prompt_tokens=ids,
        prompt_mask=~mask.bool(),
        return_result=True,
    )
    preserved = torch.equal(result.samples[~mask.bool()], ids[~mask.bool()])
    record = dict(
        profile=name,
        variant=variant,
        steps=steps,
        device=device,
        production=production,
        precision="float32",
        initial_loss=initial,
        final_loss=final,
        improved=final < initial,
        conditioning_preserved=preserved,
        seconds=elapsed,
        steps_per_second=steps / elapsed,
        parameters=sum(p.numel() for p in model.parameters()),
        sequence_length=ids.shape[1],
        batch_size=len(ids),
        peak_allocated_bytes=torch.cuda.max_memory_allocated()
        if device == "cuda"
        else None,
        training_target_accuracy=float(
            (result.samples[mask.bool()] == ids[mask.bool()]).float().mean()
        ),
    )
    output.mkdir(parents=True, exist_ok=True)
    (output / f"{name}_{variant}.json").write_text(json.dumps(record, indent=2))
    print(json.dumps(record), flush=True)
    if (not production and not record["improved"]) or not preserved:
        raise RuntimeError("pilot acceptance failed")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prepared", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument(
        "--profiles",
        nargs="+",
        default=["sudoku_extreme", "sudoku_shah", "zebra_4x4", "maze_unique"],
    )
    p.add_argument(
        "--variants",
        nargs="+",
        default=["candi_geometric_adaptive", "slerp_geometric_rank_adaptive"],
    )
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    p.add_argument(
        "--production",
        action="store_true",
        help="profile configured model and batch size, in float32 without EMA; not an overfit gate",
    )
    args = p.parse_args()
    torch.set_num_threads(1)
    if args.device == "cuda" and not torch.cuda.is_available():
        p.error("CUDA is unavailable")
    matrix = OmegaConf.load(ROOT / "configs/sweeps/sudoku_rank.yaml")
    profiles = OmegaConf.load(ROOT / "configs/sweeps/reasoning_tasks.yaml").profiles
    for name in args.profiles:
        for variant in args.variants:
            pilot(
                matrix,
                profiles,
                name,
                variant,
                args.prepared,
                args.output,
                args.steps,
                args.device,
                args.production,
            )

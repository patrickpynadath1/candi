"""Small cached-Sudoku smoke and PCA timing; never submits training jobs.

Run via uv from the repository root, with CUDA_VISIBLE_DEVICES set to a free GPU.
The truncated/reused-basis timings are alternatives to assess, not production modes.
"""

import argparse
import json
import time
from pathlib import Path

import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate

from core.processes.pca_blurring import PCABlurringProcess


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    device = torch.device(args.device)
    torch.manual_seed(42)
    torch.set_num_threads(4)
    process = PCABlurringProcess()
    x = torch.randn(16, 512, 128, device=device)
    mask = torch.ones(16, 512, dtype=torch.bool, device=device)

    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize()

    def measure(fn):
        for _ in range(3):
            fn()
        sync()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
            baseline = torch.cuda.memory_allocated()
        start = time.perf_counter()
        for _ in range(20):
            fn()
        sync()
        return dict(
            ms=(time.perf_counter() - start) * 1000 / 20,
            extra_peak_mb=(torch.cuda.max_memory_allocated() - baseline) / 2**20
            if device.type == "cuda"
            else None,
        )

    def truncated():
        return torch.pca_lowrank(x, q=32, center=True, niter=2)

    bench = {
        "shape": list(x.shape),
        "device": str(device),
        "gpu": torch.cuda.get_device_name() if device.type == "cuda" else None,
        "full_covariance_eigh": measure(lambda: process.basis(x, mask)),
        "randomized_rank32": measure(truncated),
    }
    # Amortization estimate, not a claim that stale bases preserve the kernel.
    bench["basis_every_4_calls_estimated_ms"] = bench["full_covariance_eigh"]["ms"] / 4
    from core.noise_schedules.linear_interpolation import LinearInterpolationSchedule

    path = LinearInterpolationSchedule()(torch.full((16,), 0.5, device=device))
    bench["full_corruption_including_pca"] = measure(
        lambda: process.corrupt(x, path, mask)
    )

    with initialize_config_dir(
        version_base=None,
        config_dir=str(Path(__file__).resolve().parents[2] / "configs"),
    ):
        cfg = compose(
            config_name="config",
            overrides=[
                "+experiment=sudoku_pca_blurring",
                "model.backbone.hidden_size=64",
                "model.backbone.n_heads=4",
                "model.backbone.n_blocks=2",
                "model.backbone.cond_dim=32",
                "model.backbone.dropout=0",
                "time_sampler.warmup=0",
                "sampler.steps=7",
            ],
        )
    # Guard explicitly against cache regeneration.
    cache = Path(cfg.data.cache_dir) / "sudoku_hard_train48000_valid2000_seed42"
    if not (cache / "dataset_dict.json").exists():
        raise FileNotFoundError(cache)
    task = instantiate(cfg.task)
    ds = task.training_dataset()
    batch = task.collate_training([ds[i] for i in range(8)])
    tokens = batch["input_ids"].to(device)
    loss_mask = batch["loss_mask"].to(device)
    algo = instantiate(cfg.algorithm).to(device).train()
    optimizer = torch.optim.AdamW(algo.parameters(), lr=3e-4)
    losses = []
    for step in range(20):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
        ):
            loss = algo.loss(tokens, loss_mask, training=True, step=step)["loss"]
        loss.backward()
        assert loss.isfinite() and all(
            p.grad.isfinite().all() for p in algo.parameters() if p.grad is not None
        )
        optimizer.step()
        for component in algo.modules():
            hook = getattr(component, "post_optimizer_step", None)
            if callable(hook):
                hook()
        losses.append(float(loss.detach()))
    assert algo.time_sampler.ready
    algo.eval()
    with (
        torch.no_grad(),
        torch.autocast(
            device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
        ),
    ):
        output, info = algo.generate(instantiate(cfg.sampler), tokens, ~loss_mask)
    assert torch.equal(output[~loss_mask], tokens[~loss_mask])
    assert output.shape == tokens.shape and output.min() >= 0 and output.max() < 12
    report = {
        "benchmark": bench,
        "smoke": {
            "steps": 20,
            "losses": losses,
            "nfe": info,
            "prompt_preserved": True,
            "cdcd_updates": int(algo.time_sampler.updates),
            "note": "Small two-layer execution/gradient test, not a solve-rate evaluation. Uses 128-dimensional learnable embeddings, BF16 and cached real Sudoku.",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

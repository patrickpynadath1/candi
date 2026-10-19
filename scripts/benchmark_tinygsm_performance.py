"""Measure the real TinyGSM loss/backward/AdamW/EMA path on synthetic token batches."""

import argparse
from contextlib import nullcontext
import json
import time
from pathlib import Path

import torch
from core.sweeps.configuration import compose
from core.training.module import TrainingModule
from core.training.performance import configure_performance
from core.tests.test_entrypoint import TinyTask


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--experiment", default="tinygsm_uniform_sc")
    p.add_argument(
        "--variant",
        choices=[
            "eager",
            "fused",
            "hidden",
            "compile",
            "checkpoint",
            "optimized",
            "memory",
        ],
        default="eager",
    )
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--iterations", type=int, default=16)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--profile", action="store_true")
    p.add_argument("--attention", choices=["auto", "flash", "cudnn"], default="auto")
    args = p.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(42)
    cfg = compose(args.experiment)
    cfg.training.performance = dict(
        activation_checkpointing=args.variant in ["checkpoint", "memory"],
        hidden_only_bootstrap=args.variant in ["hidden", "optimized", "memory"],
        compile=dict(
            enabled=args.variant in ["compile", "checkpoint", "optimized", "memory"],
            dynamic=True,
            mode="default",
            objective=args.variant in ["optimized", "memory"],
        ),
    )
    cfg.optimizer.fused = True if args.variant != "eager" else None
    module = TrainingModule(cfg).cuda().train()
    module.ema.move_shadow_params_to_device("cuda")
    # Nonzero output weights exercise meaningful backbone gradients from the start.
    torch.nn.init.normal_(module.algorithm.model.output_layer.linear.weight, std=0.01)
    configure_performance(module)
    optimizer = module.configure_optimizers()["optimizer"]
    tokens = torch.randint(
        0, cfg.task.vocab_size - 1, (args.batch, cfg.task.length), device="cuda"
    )
    mask = torch.ones_like(tokens, dtype=torch.bool)
    mask[:, :64] = False
    accumulation = 8
    iteration = 0
    from torch.nn.attention import SDPBackend, sdpa_kernel

    attention_context = (
        nullcontext
        if args.attention == "auto"
        else lambda: sdpa_kernel(
            {"flash": SDPBackend.FLASH_ATTENTION, "cudnn": SDPBackend.CUDNN_ATTENTION}[
                args.attention
            ]
        )
    )

    def step():
        nonlocal iteration
        with attention_context(), torch.autocast("cuda", dtype=torch.bfloat16):
            result = module.algorithm.loss(
                tokens, mask, corruption_mask=mask, training=True
            )
            loss = result["loss"] / accumulation
        loss.backward()
        iteration += 1
        if iteration % accumulation == 0:
            torch.nn.utils.clip_grad_norm_(module.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            module.ema.update(module.algorithm.parameters())
        return result["loss"].detach()

    start = time.monotonic()
    print(f"Warmup {args.variant} {args.experiment} batch={args.batch}", flush=True)
    for _ in range(8):
        step()
    torch.cuda.synchronize()
    cold = time.monotonic() - start
    print(f"Warmup completed in {cold:.2f}s", flush=True)
    torch.cuda.reset_peak_memory_stats()
    start = time.monotonic()
    for _ in range(args.iterations):
        loss = step()
    torch.cuda.synchronize()
    elapsed = time.monotonic() - start
    report = dict(
        experiment=args.experiment,
        attention=args.attention,
        variant=args.variant,
        batch=args.batch,
        seconds_per_microbatch=elapsed / args.iterations,
        tokens_per_second=args.batch * cfg.task.length * args.iterations / elapsed,
        peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
        peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30,
        warmup_seconds=cold,
        loss=loss.item(),
        torch_version=torch.__version__,
    )
    if args.profile:
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ]
        ) as profile:
            step()
            torch.cuda.synchronize()
        report["attention_operators"] = [
            e.key for e in profile.key_averages() if "attention" in e.key.lower()
        ]
        report["cuda_top"] = profile.key_averages().table(
            sort_by="self_cuda_time_total", row_limit=15
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()


# Importable through Hydra for short, isolated Lightning/DDP validation runs.


class BenchmarkTask(TinyTask):
    def training_dataset(self):
        return super().training_dataset() * 32

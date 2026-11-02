"""Measure isolated CUDA forward/backward/optimizer peaks for resolved configs.

Each config runs in a fresh subprocess. Synthetic token batches isolate execution
memory; this is not a task-quality evaluation or a Lightning/EMA memory estimate.
"""

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def measure(args):
    import torch
    from omegaconf import OmegaConf
    from core.training.module import TrainingModule
    from core.training.performance import configure_performance

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for allocator memory measurements")
    config = OmegaConf.load(args.configs[0])
    config.training.ema = 0
    config.training.performance.compile.enabled = False
    torch.manual_seed(args.seed)
    module = TrainingModule(config).cuda().train()
    configure_performance(module)
    model = module.algorithm
    compile_target = args.compile_target
    if compile_target == "transformer-layer":
        from core.models.layers import DDiTBlock

        for block in model.modules():
            if isinstance(block, DDiTBlock):
                block.forward = torch.compile(block.forward, fullgraph=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    # Task vocab includes specials; synthetic IDs exclude mask and other specials.
    vocab_size = int(config.task.vocab_size)
    length = args.length or int(config.task.length)
    torch.manual_seed(args.seed)
    tokens = torch.randint(
        0, max(2, vocab_size - 3), (args.batch_size, length), device="cuda"
    )
    mask = torch.ones_like(tokens, dtype=torch.bool)

    def step():
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = model.loss(tokens, mask, training=True)["loss"]
        if not torch.isfinite(loss):
            raise RuntimeError("nonfinite loss during memory benchmark")
        loss.backward()
        optimizer.step()
        return loss.detach()

    for _ in range(args.warmup):
        step()
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start_memory = torch.cuda.memory_allocated()
    start = time.perf_counter()
    for _ in range(args.steps):
        loss = step()
    torch.cuda.synchronize()
    record = dict(
        config=str(Path(args.configs[0]).resolve()),
        batch_size=args.batch_size,
        length=length,
        steps=args.steps,
        parameters=sum(p.numel() for p in model.parameters()),
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        starting_allocated_bytes=start_memory,
        seconds_per_step=(time.perf_counter() - start) / args.steps,
        loss=float(loss),
        device=torch.cuda.get_device_name(),
        torch_version=torch.__version__,
        cuda_version=torch.version.cuda,
        precision="bf16-mixed",
        compile=compile_target != "none",
        compile_target=compile_target,
        ema=False,
        synthetic_data=True,
        block_diffusion=config.get("block_diffusion", {}).get("enabled", False),
        block_size=config.get("block_diffusion", {}).get("block_size")
        if config.get("block_diffusion", {}).get("enabled", False)
        else None,
        activation_checkpointing=config.training.performance.get(
            "activation_checkpointing", False
        ),
    )
    Path(args.output).write_text(json.dumps(record, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("configs", nargs="+")
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--length", type=int)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--compile-target",
        choices=["none", "transformer-layer"],
        default="none",
        help="Compile repeated layers; warmup excludes compilation from timing",
    )
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.batch_size < 1 or args.steps < 1 or args.warmup < 1:
        parser.error("batch size, steps and warmup must be positive")
    if args.worker:
        measure(args)
        return
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    records = []
    for index, config in enumerate(args.configs):
        destination = output / f"{index:02d}-{Path(config).stem}.json"
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            config,
            "--worker",
            "--output",
            str(destination),
            "--batch-size",
            str(args.batch_size),
            "--warmup",
            str(args.warmup),
            "--steps",
            str(args.steps),
            "--seed",
            str(args.seed),
            "--compile-target",
            args.compile_target,
        ]
        if args.length:
            command += ["--length", str(args.length)]
        subprocess.run(command, check=True)
        records.append(json.loads(destination.read_text()))
        print(json.dumps(records[-1]), flush=True)
    (output / "summary.json").write_text(json.dumps(records, indent=2) + "\n")


if __name__ == "__main__":
    main()

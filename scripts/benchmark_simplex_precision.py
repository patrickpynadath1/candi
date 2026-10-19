"""Isolated, single-A6000 benchmark workers; JSON includes OOM/error cases.

Run from repository root with PYTHONPATH=. and CUDA_VISIBLE_DEVICES set to
one idle GPU UUID. Compilation/warmup is excluded from reported timings.
"""

import argparse
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time
from types import SimpleNamespace


def worker(case, repeats, warmup):
    import torch
    import triton
    from core.processes.dirichlet import DirichletProcess
    from core.models.simplex_projection import simplex_projection

    assert torch.cuda.device_count() == 1, "Expose exactly one GPU"
    assert torch.cuda.get_device_name(0) == "NVIDIA RTX A6000"
    torch.manual_seed(20261019)
    # Explicit numerical settings, recorded below. No TF32 FP32 baseline.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    b, length, vocab, width = case["shape"]
    precision, backend = {
        "fp64": ("fp64", "torch"),
        "fp32": ("fp32", "torch"),
        "fused": ("fp32", "triton"),
    }[case["sampler"]]
    process = DirichletProcess(
        vocab, precision=precision, sampler_backend=backend, chunk_size=case["chunk"]
    ).cuda()
    tokens = torch.randint(vocab, (b, length), device="cuda")
    path = SimpleNamespace(
        time=torch.full((b,), 0.5, device="cuda"),
        concentration=torch.ones(b, device="cuda"),
    )
    table = torch.randn(vocab, width, device="cuda", requires_grad=True)
    state = None
    if case["operation"].startswith("projection"):
        # Dense log probabilities avoid making the projection error test trivial.
        state = torch.randn(b, length, vocab, device="cuda").log_softmax(-1)

    def run():
        table.grad = None
        value = state if state is not None else process.corrupt(tokens, path)
        if case["operation"] == "corruption":
            return value
        out = simplex_projection(value, table, case["projection"], case["chunk"])
        if case["operation"] in ("projection_backward", "combined"):
            out.sum().backward()
        return out

    meta = dict(
        torch=torch.__version__,
        triton=triton.__version__,
        cuda=torch.version.cuda,
        python=platform.python_version(),
        gpu=torch.cuda.get_device_name(0),
        visible_device=os.environ.get("CUDA_VISIBLE_DEVICES"),
        total_memory=torch.cuda.get_device_properties(0).total_memory,
        tf32=False,
        bf16_reduced_precision_reduction=False,
        state_storage_bytes=b * length * vocab * (8 if precision == "fp64" else 4),
    )
    try:
        for _ in range(warmup):
            run()
        torch.cuda.synchronize()
        table.grad = None
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        timings = []
        for _ in range(repeats):
            torch.cuda.synchronize()
            start = time.perf_counter()
            run()
            torch.cuda.synchronize()
            timings.append((time.perf_counter() - start) * 1000)
        return dict(
            case=case,
            status="ok",
            metadata=meta,
            milliseconds=timings,
            median_ms=statistics.median(timings),
            peak_allocated=torch.cuda.max_memory_allocated(),
            peak_reserved=torch.cuda.max_memory_reserved(),
        )
    except torch.cuda.OutOfMemoryError as e:
        return dict(
            case=case,
            status="oom",
            metadata=meta,
            error=str(e),
            peak_allocated=torch.cuda.max_memory_allocated(),
            peak_reserved=torch.cuda.max_memory_reserved(),
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker")
    parser.add_argument("--output", default="results/simplex_precision/benchmark.json")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--small-only", action="store_true")
    args = parser.parse_args()
    if args.worker:
        print(json.dumps(worker(json.loads(args.worker), args.repeats, args.warmup)))
        return
    shapes = [[2, 32, 4097, 768]]
    if not args.small_only:
        shapes.append([16, 512, 49153, 768])
    cases = []
    for shape in shapes:
        for chunk in [0, 128, 512]:
            for sampler in ["fp64", "fp32", "fused"]:
                cases.append(
                    dict(
                        shape=shape,
                        chunk=chunk,
                        sampler=sampler,
                        projection="fp32",
                        operation="corruption",
                    )
                )
                for projection in ["fp32", "bf16"]:
                    cases.append(
                        dict(
                            shape=shape,
                            chunk=chunk,
                            sampler=sampler,
                            projection=projection,
                            operation="combined",
                        )
                    )
            for projection in ["fp32", "bf16"]:
                for operation in ["projection_forward", "projection_backward"]:
                    cases.append(
                        dict(
                            shape=shape,
                            chunk=chunk,
                            sampler="fp32",
                            projection=projection,
                            operation=operation,
                        )
                    )
    result = dict(
        command=sys.argv,
        environment={
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
        },
        git_head=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        warmup=args.warmup,
        repeats=args.repeats,
        cases=[],
    )
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    for case in cases:
        command = [
            sys.executable,
            __file__,
            "--worker",
            json.dumps(case),
            "--repeats",
            str(args.repeats),
            "--warmup",
            str(args.warmup),
        ]
        try:
            p = subprocess.run(command, capture_output=True, text=True, timeout=300)
            row = (
                json.loads(p.stdout.strip().splitlines()[-1])
                if p.returncode == 0
                else dict(case=case, status="error", error=p.stderr[-8000:])
            )
        except subprocess.TimeoutExpired:
            row = dict(case=case, status="timeout", limit_seconds=300)
        row["command"] = command
        result["cases"].append(row)
        out.write_text(json.dumps(result, indent=2))
        print(
            len(result["cases"]),
            "/",
            len(cases),
            case,
            row["status"],
            row.get("median_ms"),
            flush=True,
        )


if __name__ == "__main__":
    main()

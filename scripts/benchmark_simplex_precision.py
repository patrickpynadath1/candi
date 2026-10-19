#!/usr/bin/env python3
"""Compare FP32/FP64 simplex corruption and reverse sampling on synthetic tokens.

Run on an explicitly reserved GPU, not on a GPU used by a training job:
  python -m scripts.benchmark_simplex_precision --device cuda --output result.json
Small CPU smoke check:
  python -m scripts.benchmark_simplex_precision --device cpu --batch 1 --length 8 --vocab 257 --iterations 2
This measures corruption only, not transformer or end-to-end training throughput.
"""
import argparse
import json
from pathlib import Path
import time
import torch
from core.processes.dirichlet import DirichletProcess
from core.noise_schedules.simplex import SimplexSchedule


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--length', type=int, default=512)
    parser.add_argument('--vocab', type=int, default=49153)
    parser.add_argument('--iterations', type=int, default=5)
    parser.add_argument('--warmup', type=int, default=1)
    parser.add_argument('--precision', nargs='+', choices=['float32', 'float64'], default=['float64', 'float32'])
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if min(args.batch, args.length, args.vocab, args.iterations) <= 0 or args.warmup < 0:
        parser.error('Dimensions/iterations must be positive and warmup nonnegative')
    device = torch.device(args.device)
    torch.set_num_threads(4)
    def sync():
        if device.type == 'cuda': torch.cuda.synchronize(device)
    rows = []
    for precision in args.precision:
        torch.manual_seed(42)
        process = DirichletProcess(args.vocab, precision=precision).to(device)
        tokens = torch.randint(args.vocab, (args.batch, args.length), device=device)
        schedule = SimplexSchedule().to(device)
        source = schedule(torch.linspace(.1, .99, args.batch, device=device))
        destination = schedule(source.time / 2)
        def measure(phase, fn):
            for _ in range(args.warmup):
                value = fn(); del value
            sync()
            if device.type == 'cuda': torch.cuda.reset_peak_memory_stats(device)
            start = time.perf_counter()
            for _ in range(args.iterations):
                value = fn(); del value
            sync()
            elapsed = (time.perf_counter() - start) / args.iterations
            row = dict(precision=precision, phase=phase, seconds_per_call=elapsed)
            if device.type == 'cuda': row['peak_allocated_gib'] = torch.cuda.max_memory_allocated(device) / 2**30
            print(json.dumps(row), flush=True); rows.append(row)
        with torch.no_grad():
            measure('corrupt', lambda: process.corrupt(tokens, source))
            state = process.corrupt(tokens, source)
            assert state.dtype == getattr(torch, precision)
            measure('reverse_churn_1', lambda: process.reverse(state, tokens, destination, source, 1.0))
            del state
        del process, tokens, schedule, source, destination
        if device.type == 'cuda': torch.cuda.empty_cache()
    report = dict(device=str(device), batch=args.batch, length=args.length, vocab=args.vocab, iterations=args.iterations, results=rows)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Evaluate completed rank-min sweep checkpoints with paired Sudoku Hard samplers."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import json
from pathlib import Path
import time

from eval_sudoku_hard_paired_samplers import evaluate


ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN = ROOT / 'outputs/candi_linear_rank_min_sudoku_hard_20k_20260919/campaign.json'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', type=Path, default=CAMPAIGN)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--gpus', nargs='+', default=['4', '5', '6'])
    parser.add_argument('--steps', type=int, default=180)
    parser.add_argument('--max-examples', type=int, default=2000)
    parser.add_argument('--entropy-bound', type=float, default=0.1)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--wait-for-training', action='store_true')
    args = parser.parse_args()
    campaign = json.loads(args.campaign.read_text())
    args.output = (args.output or args.campaign.parent / 'full_eval' / 'sudoku_hard_eb0p1_greedy_steps180').resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    specs = []
    for job in campaign['jobs']:
        directory = Path(job['directory'])
        if args.wait_for_training:
            while not (directory / 'train.complete').exists():
                status_path = directory / 'train.status.json'
                if status_path.exists() and json.loads(status_path.read_text())['returncode']:
                    raise RuntimeError(f'training failed: {directory}')
                time.sleep(30)
        if not (directory / 'train.complete').exists():
            raise RuntimeError(f'training incomplete: {directory}')
        specs.append(dict(label=job['label'], overrides=job['overrides'],
                          checkpoint=directory / 'checkpoints/last.ckpt',
                          standard='candi', eb='candi_entropy_bounded', trained_on='hard'))
    work = [(spec, sampler) for spec in specs for sampler in ('standard', 'eb')]
    errors = []

    def worker(gpu, assigned):
        for spec, sampler in assigned:
            try:
                out, state = evaluate(spec, sampler, gpu, args.output, args)
                print(f'{state}: {out}', flush=True)
            except Exception as exc:
                errors.append(str(exc))
                print(f'FAILED: {spec["label"]}/{sampler}: {exc}', flush=True)

    with ThreadPoolExecutor(max_workers=len(args.gpus)) as pool:
        futures = [pool.submit(worker, gpu, work[i::len(args.gpus)])
                   for i, gpu in enumerate(args.gpus)]
        for future in as_completed(futures):
            future.result()
    rows = []
    for result_path in sorted(args.output.glob('*/*/results.json')):
        value = json.loads(result_path.read_text())
        meta = json.loads((result_path.parent / 'evaluation.json').read_text())
        rows.append({key: meta[key] for key in ('label', 'sampler', 'predictor', 'trained_on', 'evaluated_on', 'checkpoint')} |
                    {key: value.get(key) for key in ('num_correct', 'num_total', 'accuracy', 'cell_accuracy', 'mean_nfe', 'evaluation_seconds')})
    if rows:
        with (args.output / 'summary.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=rows[0])
            writer.writeheader()
            writer.writerows(rows)
    print(f'{len(rows)}/{len(work)} results; {len(errors)} failures', flush=True)
    if errors or len(rows) != len(work):
        raise SystemExit(1)


if __name__ == '__main__':
    main()

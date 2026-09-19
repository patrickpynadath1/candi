#!/usr/bin/env python3
"""Compare standard reverse and greedy entropy-bound sampling on the same checkpoints."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import json
import os
from pathlib import Path
import subprocess
import time


ROOT = Path(__file__).resolve().parents[1]
CACHE = Path('/home/patrick/.cache/duo/new_sudoku_runs')
CAMPAIGN = ROOT / 'outputs/sudoku_core_ablation_20k_seed1_20260915/campaign.json'
CDCD_CAMPAIGN = ROOT / 'outputs/sudoku_hard_onehot_cdcd_adaptive_v2_20k_seed1_20260917/campaign.json'
DUO_INTEGRAL = ROOT / 'outputs/sudoku_hard_paired_samplers_20260919/sudoku_duo_integral.pkl'


def jobs():
    campaign = json.loads(CAMPAIGN.read_text())
    ranks = {
        job['method_name']: job for job in campaign['jobs']
        if job['task'] == 'sudoku_hard'
    }
    cdcd = {
        job['method_name']: job for job in json.loads(CDCD_CAMPAIGN.read_text())['jobs']
    }
    specs = [
        ('candi_linear_rank_cdcd', cdcd['candi_linear_cdcd_adaptive_hybrid'], 'candi', 'candi_entropy_bounded', 'hard'),
        ('candi_geometric_rank_cdcd', cdcd['candi_geometric_cdcd_adaptive_hybrid'], 'candi', 'candi_entropy_bounded', 'hard'),
        ('candi_linear_rank_fixed', ranks['candi_orig_hybrid'], 'candi', 'candi_entropy_bounded', 'hard'),
        ('candi_geometric_rank_fixed', ranks['candi_geometric_hybrid'], 'candi', 'candi_entropy_bounded', 'hard'),
        ('candi_geometric_adaptive', ranks['candi_geometric_adaptive_hybrid'], 'candi', 'candi_entropy_bounded', 'hard'),
    ]
    baselines = [
        ('mdlm', 'sudoku_mdlm_hard_eb', CACHE / 'hard_eb_0p1/mdlm/checkpoints/last.ckpt', 'ancestral_cache', 'entropy_bounded', 'hard'),
        ('candi_original_legacy', 'sudoku_candi_hard_eb', CACHE / 'hard_eb_0p1/candi/checkpoints/last.ckpt', 'candi', 'candi_entropy_bounded', 'hard'),
        ('udlm_easy_trained', 'sudoku_duo', CACHE / 'duo/checkpoints/last.ckpt', 'ancestral_cache', 'uniform_entropy_bounded', 'easy'),
    ]
    for label, job, standard, eb, trained_on in specs:
        yield dict(label=label, overrides=job['overrides'], checkpoint=Path(job['directory']) / 'checkpoints/last.ckpt', standard=standard, eb=eb, trained_on=trained_on)
    for label, experiment, checkpoint, standard, eb, trained_on in baselines:
        overrides = [f'+experiment={experiment}', 'data.difficulty=hard', 'data.cache_dir=/home/patrick/.cache/duo/data_cache']
        if label == 'mdlm':
            overrides.append('algo.adaLN=false')
        if label == 'candi_original_legacy':
            overrides.append('+algo.legacy_architecture=true')
        if label == 'udlm_easy_trained':
            overrides.append(f'algo.integral_cache_path={DUO_INTEGRAL}')
        yield dict(label=label, overrides=overrides, checkpoint=checkpoint, standard=standard, eb=eb, trained_on=trained_on)


def evaluate(spec, sampler_kind, gpu, output_root, args):
    out = output_root / spec['label'] / sampler_kind
    result = out / 'results.json'
    if result.exists():
        return out, 'cached'
    out.mkdir(parents=True, exist_ok=True)
    checkpoint = spec['checkpoint']
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    predictor = spec['standard'] if sampler_kind == 'standard' else spec['eb']
    overrides = list(spec['overrides']) + [
        'mode=sudoku_eval',
        'data.difficulty=hard',
        f'sampling.predictor={predictor}',
        f'sampling.steps={args.steps}',
        'sampling.noise_removal=greedy',
        f'sampling.entropy_bound={args.entropy_bound}',
        f'sampling.greedy_clean_estimate={str(sampler_kind == "eb").lower()}',
        f'sudoku.max_examples={args.max_examples}',
        f'sudoku.batch_size={args.batch_size}',
        f'loader.eval_global_batch_size={args.batch_size}',
        'loader.num_workers=1',
        f'eval.checkpoint_path={checkpoint}',
        f'sudoku.output_dir={out}',
        f'hydra.run.dir={out}',
        'wandb=null',
    ]
    meta = dict(label=spec['label'], sampler=sampler_kind, predictor=predictor,
                trained_on=spec['trained_on'], evaluated_on='hard', checkpoint=str(checkpoint),
                steps=args.steps, entropy_bound=args.entropy_bound if sampler_kind == 'eb' else None,
                greedy_clean_estimate=sampler_kind == 'eb', gpu=gpu, command=overrides)
    (out / 'evaluation.json').write_text(json.dumps(meta, indent=2))
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='1', TOKENIZERS_PARALLELISM='false')
    started = time.time()
    with (out / 'eval.log').open('w') as log:
        proc = subprocess.run([str(ROOT / '.venv/bin/python'), str(ROOT / 'main.py'), '--config-name=config', *overrides],
                              cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
    meta['seconds'] = time.time() - started
    meta['exit_code'] = proc.returncode
    (out / 'evaluation.json').write_text(json.dumps(meta, indent=2))
    if proc.returncode or not result.exists():
        raise RuntimeError(f'{out}: exit={proc.returncode}; see {out / "eval.log"}')
    return out, 'complete'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs/sudoku_hard_paired_samplers_20260919')
    parser.add_argument('--gpus', nargs='+', default=[str(i) for i in range(7)])
    parser.add_argument('--steps', type=int, default=180)
    parser.add_argument('--max-examples', type=int, default=2000)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--entropy-bound', type=float, default=0.1)
    parser.add_argument('--models', nargs='+')
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    work = [(spec, sampler) for spec in jobs() if args.models is None or spec['label'] in args.models for sampler in ('standard', 'eb')]
    errors = []
    def queue(gpu, assigned):
        for spec, sampler in assigned:
            try:
                out, status = evaluate(spec, sampler, gpu, args.output, args)
                print(f'{status}: {out}', flush=True)
            except Exception as exc:
                print(f'FAILED: {spec["label"]}/{sampler}: {exc}', flush=True)
                errors.append(str(exc))
    with ThreadPoolExecutor(max_workers=len(args.gpus)) as pool:
        futures = [pool.submit(queue, gpu, work[i::len(args.gpus)]) for i, gpu in enumerate(args.gpus)]
        for future in as_completed(futures):
            future.result()
    rows = []
    for result in sorted(args.output.glob('*/*/results.json')):
        value = json.loads(result.read_text())
        meta = json.loads((result.parent / 'evaluation.json').read_text())
        rows.append({key: meta[key] for key in ('label', 'sampler', 'predictor', 'trained_on', 'evaluated_on', 'checkpoint')} |
                    {key: value.get(key) for key in ('num_correct', 'num_total', 'accuracy', 'cell_accuracy', 'mean_nfe', 'evaluation_seconds')})
    if rows:
        with (args.output / 'summary.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=rows[0])
            writer.writeheader()
            writer.writerows(rows)
    print(f'{len(rows)} results total; {len(work)} requested; {len(errors)} failures; {args.output / "summary.csv"}', flush=True)
    if errors:
        raise SystemExit(1)


if __name__ == '__main__':
    main()

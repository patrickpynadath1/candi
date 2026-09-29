#!/usr/bin/env python3
"""Queue final easy-trained checkpoints for evaluation on hard validation data."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]


def write_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)


def sampler_for(job, duo_sampler='uniform_entropy_bounded', sampler_mode='eb'):
    experiment = job['method']['experiment']
    if sampler_mode == 'standard':
        return {
            'sudoku_candi_rank': 'candi',
            'sudoku_candi_embedding': 'candi_embedding',
            'sudoku_candi_slerp': 'candi_embedding',
            'sudoku_mdlm': 'ancestral',
            'sudoku_sfm_truncated_adaptive': 'sfm',
            'sudoku_duo': 'ancestral',
        }[experiment]
    return {
        'sudoku_candi_rank': 'candi_entropy_bounded',
        'sudoku_candi_embedding': 'candi_embedding_entropy_bounded',
        'sudoku_candi_slerp': 'candi_embedding_entropy_bounded',
        'sudoku_mdlm': 'entropy_bounded',
        'sudoku_sfm_truncated_adaptive': 'sfm',
        'sudoku_duo': duo_sampler,
    }[experiment]


def final_checkpoint(job):
    directory = Path(job['directory']) / 'checkpoints'
    candidates = list(directory.glob('last*.ckpt'))
    if len(candidates) <= 1:
        return directory / 'last.ckpt'
    # Lightning versions the last checkpoint after an interrupted/resumed run.
    import torch
    def step(path):
        checkpoint = torch.load(path, map_location='cpu', mmap=True, weights_only=False)
        return checkpoint['global_step']
    return max(candidates, key=step)


def eval_overrides(job, output, cache, args):
    # Preserve the training architecture and learned adaptive schedule buffers.
    sampler = sampler_for(job, args.duo_sampler, args.sampler_mode)
    return list(job['overrides']) + [
        'mode=sudoku_eval', 'data.difficulty=hard',
        f'data.cache_dir={cache}', 'data.num_train=200000',
        'data.num_valid=10000', 'data.data_seed=42',
        f'sampling.predictor={sampler}',
        f'sampling.steps={args.steps}',
        f'sampling.entropy_bound={args.entropy_bound}',
        f'sampling.greedy_clean_estimate={str(args.sampler_mode == "eb" and sampler != "ancestral").lower()}',
        f'sampling.noise_removal={"ancestral" if sampler == "ancestral" else "greedy"}',
        'eval.disable_ema=false',
        f'eval.checkpoint_path={Path(job["directory"]) / "checkpoints/last.ckpt"}',
        f'sudoku.max_examples={args.max_examples}',
        f'sudoku.batch_size={args.batch_size}', 'loader.num_workers=0',
        f'sudoku.output_dir={output}', f'hydra.run.dir={output}', 'wandb=null',
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gpus', nargs='+', required=True)
    parser.add_argument('--steps', type=int, default=180)
    parser.add_argument('--max-examples', type=int, default=10000)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--entropy-bound', type=float, default=0.1)
    parser.add_argument('--methods', nargs='+')
    parser.add_argument('--trained-on', choices=['easy', 'hard'], default='easy')
    parser.add_argument('--sampler-mode', choices=['eb', 'standard'], default='eb')
    parser.add_argument('--clues', type=int, choices=range(31),
                        help='Thin hard validation prompts to this many clues.')
    parser.add_argument('--duo-sampler', choices=['uniform_entropy_bounded', 'ancestral'],
                        default='uniform_entropy_bounded')
    parser.add_argument('--fixed-linear-clock', action='store_true',
                        help='Evaluate only adaptive-trained methods, bypassing their learned time warp.')
    parser.add_argument('--after', type=Path,
                        help='Wait until this evaluation queue has no actively running evaluations.')
    parser.add_argument('--baseline-checkpoints', type=Path,
                        help='Pinned available checkpoints: NAME/last.ckpt for S-FLM and DUO.')
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--detach', action='store_true')
    args = parser.parse_args()
    args.campaign = args.campaign.resolve()
    args.output = args.output.resolve()
    if args.detach:
        if not args.execute:
            parser.error('--detach requires --execute')
        args.output.mkdir(parents=True, exist_ok=True)
        with (args.output / 'supervisor.log').open('a') as log:
            process = subprocess.Popen(
                [sys.executable, *[a for a in sys.argv if a != '--detach']],
                cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log,
                stderr=subprocess.STDOUT, start_new_session=True,
            )
        print(process.pid)
        return
    campaign = json.loads(args.campaign.read_text())
    jobs = [j for j in campaign['jobs'] if j['task'] == f'sudoku_{args.trained_on}']
    if args.methods:
        missing = set(args.methods) - {j['method_name'] for j in jobs}
        if missing:
            raise ValueError(f'Unknown methods: {sorted(missing)}')
        jobs = [j for j in jobs if j['method_name'] in args.methods]
    if args.fixed_linear_clock:
        jobs = [j for j in jobs if '_adaptive_' in j['method_name']
                or j['method']['experiment'] == 'sudoku_sfm_truncated_adaptive']
    if not jobs:
        raise ValueError(f'No {args.trained_on}-trained jobs found')
    if len({j['method_name'] for j in jobs}) != len(jobs):
        raise ValueError('Expected one training seed per method')
    source = args.campaign.parent / 'source'
    # Checkpoints pickle the Sudoku tokenizer from the frozen training source.
    sys.path.insert(0, str(source))
    cache = ROOT / 'data_cache'
    work = []
    for job in jobs:
        output = args.output / job['method_name']
        checkpoint = final_checkpoint(job)
        work.append(dict(
            method=job['method_name'], trained_on=args.trained_on, evaluated_on='hard',
            split='validation', checkpoint=str(checkpoint),
            sampler=sampler_for(job, args.duo_sampler, args.sampler_mode), source=str(source),
            command=[sys.executable, *(
                [str(ROOT / 'new_scripts/eval_sudoku_fixed_clock.py'), '--source', str(source)]
                if args.fixed_linear_clock else [str(source / 'main.py')]),
                     *eval_overrides(job, output, cache, args),
                     f'eval.checkpoint_path={checkpoint}'],
        ))
        if args.clues is not None:
            if args.fixed_linear_clock:
                raise ValueError('Clue thinning cannot be combined with the clock override')
            work[-1]['command'] = [sys.executable,
                str(ROOT / 'new_scripts/eval_sudoku_fewer_clues.py'),
                '--source', str(source), '--clues', str(args.clues),
                *work[-1]['command'][2:]]
            work[-1]['clues'] = args.clues
        if args.baseline_checkpoints and job['method_name'] in {'sfm_baseline', 'duo_baseline'}:
            checkpoint = args.baseline_checkpoints.resolve() / job['method_name'] / 'last.ckpt'
            if not checkpoint.exists():
                raise FileNotFoundError(checkpoint)
            work[-1]['checkpoint'] = str(checkpoint)
            work[-1]['command'].append(f'eval.checkpoint_path={checkpoint}')
    plan = dict(campaign=str(args.campaign), gpus=args.gpus, jobs=work,
                steps=args.steps, max_examples=args.max_examples,
                entropy_bound=args.entropy_bound, ema=True, batch_size=args.batch_size,
                fixed_linear_clock=args.fixed_linear_clock,
                baseline_checkpoints=str(args.baseline_checkpoints.resolve()) if args.baseline_checkpoints else None,
                after=str(args.after.resolve()) if args.after else None)
    if not args.execute:
        print(json.dumps(plan, indent=2))
        return
    args.output.mkdir(parents=True, exist_ok=True)
    lock = (args.output / 'supervisor.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    plan_path = args.output / 'plan.json'
    if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
        raise ValueError('Evaluation plan differs; use a new output directory')
    write_json(plan_path, plan)
    from datasets import load_from_disk
    data = load_from_disk(str(cache / 'sudoku_hard_train200000_valid10000_seed42'))
    if len(data['validation']) != 10000:
        raise ValueError('Expected 10,000 hard validation puzzles')
    states = {w['method']: ('complete' if (args.output / w['method'] / 'results.json').exists()
                            else 'waiting for final checkpoint') for w in work}
    mutex = threading.Lock()

    def save():
        write_json(args.output / 'status.json', dict(pid=os.getpid(), jobs=states))
        rows = []
        for w in work:
            path = args.output / w['method'] / 'results.json'
            if path.exists():
                result = json.loads(path.read_text())
                metadata = json.loads((path.parent / 'evaluation.json').read_text())
                rows.append(dict(method=w['method'], trained_on=args.trained_on, evaluated_on='hard',
                                 sampler=w['sampler'], checkpoint_step=metadata['checkpoint_step'],
                                 **{k: result.get(k) for k in (
                                     'accuracy', 'digit_accuracy', 'valid_solution_rate',
                                     'clue_consistency_rate', 'mean_nfe', 'num_correct',
                                     'num_total', 'evaluation_seconds')}))
        if rows:
            with (args.output / 'summary.csv').open('w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)

    def worker(gpu):
        while True:
            if args.after:
                prior = json.loads((args.after / 'status.json').read_text())['jobs']
                if any(s.startswith(('running', 'starting')) for s in prior.values()):
                    time.sleep(15)
                    continue
            with mutex:
                pending = [i for i, w in enumerate(work)
                           if states[w['method']] == 'waiting for final checkpoint']
                if not pending:
                    return
                index = next((i for i in pending if
                              (Path(jobs[i]['directory']) / 'train.complete').exists()
                              or (args.baseline_checkpoints and work[i]['method'] in
                                  {'sfm_baseline', 'duo_baseline'})), None)
                if index is not None:
                    states[work[index]['method']] = f'starting on GPU {gpu}'
                    save()
            if index is None:
                time.sleep(30)
                continue
            w = work[index]
            output = args.output / w['method']
            output.mkdir(parents=True, exist_ok=True)
            try:
                import torch
                checkpoint = torch.load(w['checkpoint'], map_location='cpu', mmap=True,
                                        weights_only=False)
                step = checkpoint['global_step']
                del checkpoint
                if step != 50000 and not (args.baseline_checkpoints and
                                         w['method'] in {'sfm_baseline', 'duo_baseline'}):
                    raise ValueError(f'Expected final step 50000, got {step}')
                metadata = dict(w, gpu=gpu, checkpoint_step=step, ema=True)
                write_json(output / 'evaluation.json', metadata)
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='1',
                           TOKENIZERS_PARALLELISM='false', PYTHONUNBUFFERED='1')
                with (output / 'eval.log').open('a') as log:
                    process = subprocess.Popen(w['command'], cwd=source, env=env,
                                               stdout=log, stderr=subprocess.STDOUT)
                    with mutex:
                        states[w['method']] = f'running on GPU {gpu}, pid {process.pid}'
                        save()
                    code = process.wait()
                if code:
                    raise RuntimeError(f'Evaluation exited {code}; see {output / "eval.log"}')
                result = json.loads((output / 'results.json').read_text())
                if result['num_total'] != min(args.max_examples, 10000):
                    raise ValueError('Evaluation did not cover the requested split')
                status = 'complete'
            except Exception as error:
                status = f'failed: {error}'
            with mutex:
                states[w['method']] = status
                save()
            print(w['method'], status, flush=True)

    save()
    with ThreadPoolExecutor(max_workers=len(args.gpus)) as pool:
        list(pool.map(worker, args.gpus))
    if any(s != 'complete' for s in states.values()):
        raise SystemExit(1)


if __name__ == '__main__':
    main()

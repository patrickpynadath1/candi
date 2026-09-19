#!/usr/bin/env python3
"""Run the Sudoku Hard linear-rank minimum sweep against the frozen .01 baseline."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'outputs/sudoku_core_ablation_20k_seed1_20260915'
SOURCE = BASE / 'source'
DEFAULT_ROOT = ROOT / 'outputs/candi_linear_rank_min_sudoku_hard_20k_20260919'


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def make_jobs(output):
    campaign = json.loads((BASE / 'campaign.json').read_text())
    baseline = next(j for j in campaign['jobs'] if
                    j['task'] == 'sudoku_hard' and j['method_name'] == 'candi_orig_hybrid')
    jobs = []
    for rank_min in (0.01, 0.10, 0.15, 0.20):
        label = f'rank_min_{rank_min:.2f}'.replace('.', 'p')
        directory = output / 'runs' / label / 'seed_1'
        run_name = f'sudoku_hard_candi_linear_rank_min_{rank_min:.2f}_seed1'
        run_id = hashlib.sha256(f'{output.name}:{run_name}'.encode()).hexdigest()[:16]
        replace = {
            'checkpointing.save_dir': str(directory),
            'hydra.run.dir': str(directory),
            'wandb.id': run_id,
            'wandb.name': run_name,
            'wandb.group': 'sudoku_hard_candi_linear_rank_min',
            'wandb.tags': f'[sudoku_hard,candi,linear_rank,rank_min_{rank_min:.2f}]',
        }
        overrides = []
        for value in baseline['overrides']:
            key = value.split('=', 1)[0].lstrip('+')
            overrides.append(f'{key}={replace[key]}' if key in replace else value)
        overrides += [f'algo.min_percentile={rank_min}', 'algo.max_percentile=0.49']
        jobs.append(dict(rank_min=rank_min, rank_max=0.49, label=label,
                         directory=str(directory), run_id=run_id, run_name=run_name,
                         wandb_url=f'https://wandb.ai/rz-text-dlp/reasoning-sudoku-core-ablations/runs/{run_id}',
                         overrides=overrides))
    return baseline, jobs


def gpu_idle(gpu):
    value = subprocess.check_output([
        'nvidia-smi', '-i', str(gpu), '--query-gpu=memory.used,utilization.gpu',
        '--format=csv,noheader,nounits'], text=True).strip()
    memory, utilization = (int(part.strip()) for part in value.split(','))
    return memory < 1024 and utilization < 10


def run_campaign(root, gpus):
    root.mkdir(parents=True, exist_ok=True)
    with (root / 'supervisor.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        baseline, jobs = make_jobs(root)
        manifest = dict(baseline_directory=baseline['directory'],
                        baseline_rank_min=0.01, source=str(SOURCE),
                        project='reasoning-sudoku-core-ablations',
                        steps=20000, task='sudoku_hard', seed=1,
                        gpus=gpus, jobs=jobs)
        campaign_path = root / 'campaign.json'
        if campaign_path.exists() and json.loads(campaign_path.read_text()) != manifest:
            raise ValueError('existing sweep manifest differs')
        atomic_json(campaign_path, manifest)
        states = {job['label']: ('complete' if (Path(job['directory']) / 'train.complete').exists() else 'queued') for job in jobs}
        mutex = threading.Lock()

        def save():
            atomic_json(root / 'status.json', dict(pid=os.getpid(),
                updated_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), jobs=states))

        def worker(gpu):
            while True:
                while not gpu_idle(gpu):
                    time.sleep(30)
                with mutex:
                    pending = next((j for j in jobs if states[j['label']] == 'queued'), None)
                    if pending is None:
                        return
                    states[pending['label']] = f'starting on GPU {gpu}'
                    save()
                directory = Path(pending['directory'])
                directory.mkdir(parents=True, exist_ok=True)
                try:
                    command = [sys.executable, str(SOURCE / 'main.py'), *pending['overrides']]
                    atomic_json(directory / 'run.json', pending | dict(gpu=gpu, command=command))
                    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='1',
                               TOKENIZERS_PARALLELISM='false', WANDB_MODE='online',
                               WANDB_RESUME='allow', PYTHONUNBUFFERED='1')
                    started = time.monotonic()
                    with (directory / 'train.log').open('a') as stream:
                        process = subprocess.Popen(command, cwd=SOURCE, env=env,
                                                   stdout=stream, stderr=subprocess.STDOUT)
                        with mutex:
                            states[pending['label']] = f'running on GPU {gpu}, pid {process.pid}'
                            save()
                        code = process.wait()
                    atomic_json(directory / 'train.status.json',
                                dict(returncode=code, seconds=time.monotonic() - started))
                    if code:
                        raise RuntimeError(f'training exit {code}')
                    (directory / 'train.complete').touch()
                    status = 'complete'
                except Exception as exc:
                    status = f'failed: {exc}'
                with mutex:
                    states[pending['label']] = status
                    save()

        save()
        with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
            futures = [pool.submit(worker, gpu) for gpu in gpus]
            for future in futures:
                future.result()
        if any(state != 'complete' for state in states.values()):
            raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    parser.add_argument('--gpus', nargs='+', default=['0', '1', '2', '3'])
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--detach', action='store_true')
    args = parser.parse_args()
    root = args.root.resolve()
    baseline, jobs = make_jobs(root)
    if not args.execute:
        print(json.dumps(dict(baseline=baseline['directory'], jobs=jobs), indent=2))
        return
    if args.detach:
        root.mkdir(parents=True, exist_ok=True)
        command = [sys.executable, str(Path(__file__).resolve()), '--root', str(root),
                   '--gpus', *args.gpus, '--execute']
        with (root / 'supervisor.log').open('a') as stream:
            process = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL,
                                       stdout=stream, stderr=subprocess.STDOUT,
                                       start_new_session=True)
        print(process.pid)
        return
    run_campaign(root, args.gpus)


if __name__ == '__main__':
    main()

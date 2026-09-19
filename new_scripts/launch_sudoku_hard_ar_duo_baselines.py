#!/usr/bin/env python3
"""Launch matched 20k-step Sudoku Hard AR and curriculum-free DUO baselines."""

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

from launch_reasoning_method_smoke import snapshot


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = ROOT / 'outputs/sudoku_hard_ar_duo_nocurriculum_20k_20260919'
DATA_CACHE = ROOT / 'outputs/sudoku_core_ablation_20k_seed1_20260915/data'
PROJECT = 'reasoning-sudoku-core-ablations'
ENTITY = 'rz-text-dlp'


def atomic_json(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)


def jobs(root):
    specifications = [
        ('ar', 'sudoku_ar_adaptive_control', []),
        ('duo_no_curriculum', 'sudoku_duo',
         ['algo.name=duo_base', '+trainer.check_val_every_n_epoch=null']),
    ]
    result = []
    for label, experiment, special in specifications:
        directory = root / 'runs' / label / 'seed_1'
        run_name = f'sudoku_hard_{label}_20k_seed1'
        run_id = hashlib.sha256(f'{root.name}:{run_name}'.encode()).hexdigest()[:16]
        overrides = [
            f'+experiment={experiment}', *special,
            '+callbacks=reasoning_eval',
            'seed=1', 'data=sudoku', 'data.difficulty=hard',
            f'data.cache_dir={DATA_CACHE}',
            'model.length=180', 'model.hidden_size=512',
            'loader.global_batch_size=256',
            'loader.eval_global_batch_size=128',
            'loader.num_workers=4',
            'reasoning.batch_size=64',
            'trainer.devices=1', 'trainer.num_nodes=1',
            'trainer.max_steps=20000', 'trainer.limit_val_batches=0',
            'callbacks.reasoning_eval.every_n_train_steps=1000',
            'callbacks.reasoning_eval.subset_size=128',
            'callbacks.reasoning_eval.num_steps=45',
            'callbacks.checkpoint_every_n_steps.every_n_train_steps=5000',
            'callbacks.checkpoint_every_n_steps.save_top_k=0',
            'callbacks.checkpoint_every_n_steps.save_last=true',
            'callbacks.checkpoint_monitor.save_top_k=0',
            'callbacks.checkpoint_monitor.monitor=null',
            'checkpointing.resume_from_ckpt=true',
            f'checkpointing.save_dir={directory}',
            f'hydra.run.dir={directory}',
            f'wandb.project={PROJECT}', f'+wandb.entity={ENTITY}',
            f'wandb.id={run_id}', f'wandb.name={run_name}',
            'wandb.group=sudoku_hard_baselines_20k',
            'wandb.job_type=train',
            f'wandb.tags=[sudoku_hard,baseline,{label},seed1]',
        ]
        result.append(dict(label=label, directory=str(directory),
                           run_name=run_name, run_id=run_id,
                           wandb_url=f'https://wandb.ai/{ENTITY}/{PROJECT}/runs/{run_id}',
                           overrides=overrides))
    return result


def run(root, gpus):
    root.mkdir(parents=True, exist_ok=True)
    with (root / 'supervisor.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        source = snapshot(root)
        work = jobs(root)
        manifest = dict(task='sudoku_hard', seed=1, max_steps=20000,
                        data_cache=str(DATA_CACHE), project=PROJECT,
                        source=str(source), gpus=gpus, jobs=work,
                        note='DUO_BASE uses the full uniform-diffusion objective from step zero.')
        manifest_path = root / 'campaign.json'
        if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
            raise ValueError('existing campaign manifest differs')
        atomic_json(manifest_path, manifest)
        states = {job['label']: ('complete' if (Path(job['directory'])/'train.complete').exists() else 'queued') for job in work}
        mutex = threading.Lock()

        def save():
            atomic_json(root / 'status.json', dict(pid=os.getpid(),
                        updated_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                        jobs=states))

        def worker(gpu):
            while True:
                with mutex:
                    job = next((job for job in work if states[job['label']] == 'queued'), None)
                    if job is None:
                        return
                    states[job['label']] = f'starting on GPU {gpu}'
                    save()
                directory = Path(job['directory'])
                directory.mkdir(parents=True, exist_ok=True)
                try:
                    command = [sys.executable, str(source / 'main.py'), *job['overrides']]
                    atomic_json(directory / 'run.json', job | dict(gpu=gpu, command=command))
                    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='1',
                               TOKENIZERS_PARALLELISM='false', WANDB_MODE='online',
                               WANDB_RESUME='allow', PYTHONUNBUFFERED='1')
                    started = time.monotonic()
                    with (directory / 'train.log').open('a') as stream:
                        process = subprocess.Popen(command, cwd=source, env=env,
                                                   stdout=stream, stderr=subprocess.STDOUT)
                        with mutex:
                            states[job['label']] = f'running on GPU {gpu}, pid {process.pid}'
                            save()
                        code = process.wait()
                    atomic_json(directory / 'train.status.json', dict(returncode=code,
                                seconds=time.monotonic()-started))
                    if code:
                        raise RuntimeError(f'training exit {code}')
                    (directory / 'train.complete').touch()
                    status = 'complete'
                except Exception as exc:
                    status = f'failed: {exc}'
                with mutex:
                    states[job['label']] = status
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
    parser.add_argument('--gpus', nargs='+', default=['0', '1'])
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--detach', action='store_true')
    args = parser.parse_args()
    root = args.root.resolve()
    if not args.execute:
        print(json.dumps(jobs(root), indent=2))
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
    run(root, args.gpus)


if __name__ == '__main__':
    main()

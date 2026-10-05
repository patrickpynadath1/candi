"""Resume matched hint-removal evaluations from a completed sampler comparison."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, is_dataclass
import csv
import json
import os
from pathlib import Path
import queue
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DEFAULT_SOURCE = ROOT / 'core/outputs/adaptive-sweep-evaluation-20261005'
DEFAULT_OUTPUT = ROOT / 'core/outputs/hint-removal-20261005'
FRACTIONS = (1., .75, .5, .25, 0.)


def save(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2,
        default=lambda obj: asdict(obj) if is_dataclass(obj) else str(obj)))
    temporary.replace(path)


def prepare(args):
    from omegaconf import OmegaConf
    args.output.mkdir(parents=True, exist_ok=True)
    tasks = []
    for row in json.loads((args.source / 'summary.json').read_text()):
        source = Path(row['result_file']).parent
        config_path = source / 'config.yaml'
        if not config_path.exists():
            config_path = Path(json.loads((source / 'source.json').read_text())['config'])
        dest = args.output / row['run'] / row['sampler']
        dest.mkdir(parents=True, exist_ok=True)
        config = OmegaConf.load(config_path)
        config.wandb = None
        OmegaConf.save(OmegaConf.create(OmegaConf.to_container(config, resolve=True)), dest / 'config.yaml')
        tasks.append({'directory': str(dest), 'run': row['run'], 'sampler': row['sampler']})
    manifest = {'puzzles': args.puzzles, 'repetitions': args.repetitions,
                'seed': 1729, 'fractions': FRACTIONS, 'tasks': tasks,
                'unique_definition': 'distinct well-formed 81-digit boards (including invalid Sudoku grids)',
                'valid_definition': 'valid rows, columns, boxes, output format, and remaining clues; original answer match not required'}
    path = args.output / 'manifest.json'
    if path.exists() and json.loads(path.read_text()) != json.loads(json.dumps(manifest)):
        raise ValueError('Existing sweep manifest differs; use a different output directory')
    save(path, manifest)
    print(f'{len(tasks)} configurations, {args.puzzles * args.repetitions * len(FRACTIONS)} generations each', flush=True)


def worker(args):
    import lightning as L
    import torch
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    from core.training.module import TrainingModule
    from core.training.checkpoints import load_weights
    from core.evaluation.generation import evaluate
    from core.tasks.sudoku_diversity import HintRemovalTask, diversity_summary
    manifest = json.loads((args.output / 'manifest.json').read_text())
    dest = Path(args.worker)
    config = OmegaConf.load(dest / 'config.yaml')
    L.seed_everything(manifest['seed'], workers=True)
    base = instantiate(config.task)
    module = TrainingModule(config)
    if not config.evaluation.use_ema:
        module.ema = None
    step = load_weights(module, config.evaluation.checkpoint, legacy=config.evaluation.legacy)
    module.to('cuda')
    indices = torch.randperm(len(base.validation_dataset()), generator=torch.Generator().manual_seed(manifest['seed']))[:manifest['puzzles']].tolist()
    for fraction in manifest['fractions']:
        path = dest / f'hints_{fraction:g}.json'
        if path.exists():
            continue
        L.seed_everything(manifest['seed'], workers=True)
        task = HintRemovalTask(base, indices, manifest['repetitions'], fraction, manifest['seed'])
        result = evaluate(module, task, config.sampler, batch_size=64,
                          use_ema=config.evaluation.use_ema, precision=config.trainer.precision)
        summary = diversity_summary(result['records'], task)
        save(path, {'fraction_retained': fraction, 'checkpoint_step': step,
                    'use_ema': config.evaluation.use_ema, 'seed': manifest['seed'],
                    'mean_nfe': result['mean_nfe'], 'seconds': result['evaluation_seconds'],
                    **summary, 'records': result['records']})
        print(f"DONE {dest.parent.name}/{dest.name} hints={fraction} valid={summary['valid_rate']:.3%} distinct_valid/prompt={summary['mean_valid_unique_per_prompt']:.3f}", flush=True)


def compile_results(output):
    rows = []
    for path in output.glob('*/*/hints_*.json'):
        result = json.loads(path.read_text())
        rows.append({'run': path.parents[1].name, 'sampler': path.parent.name,
                     **{k:v for k,v in result.items() if k not in ('records', 'prompts')},
                     'result_file': str(path)})
    rows.sort(key=lambda r: (r['run'], r['sampler'], -r['fraction_retained']))
    save(output / 'summary.json', rows)
    if rows:
        with (output / 'summary.csv').open('w') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
    return rows


def launch(args):
    manifest = json.loads((args.output / 'manifest.json').read_text())
    pending = queue.Queue()
    for task in manifest['tasks']:
        if any(not (Path(task['directory']) / f'hints_{f:g}.json').exists() for f in manifest['fractions']):
            pending.put(task)
    failures = []
    def consume(gpu):
        while True:
            try:
                task = pending.get_nowait()
            except queue.Empty:
                return
            dest = Path(task['directory'])
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='2', PYTHONUTF8='1')
            print('START', gpu, task['run'], task['sampler'], flush=True)
            with (dest / 'evaluation.log').open('a') as log:
                result = subprocess.run([sys.executable, __file__, '--output', str(args.output), '--worker', str(dest)],
                                        cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
            print('FINISH', gpu, task['run'], task['sampler'], result.returncode, flush=True)
            if result.returncode:
                failures.append(task)
    with ThreadPoolExecutor(max_workers=len(args.gpus.split(','))) as executor:
        list(executor.map(consume, args.gpus.split(',')))
    save(args.output / 'failures.json', failures)
    compile_results(args.output)
    if failures:
        raise RuntimeError(f'{len(failures)} configurations failed; see failures.json')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=DEFAULT_SOURCE)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--puzzles', type=int, default=64)
    parser.add_argument('--repetitions', type=int, default=32)
    parser.add_argument('--gpus', default='1,2')
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--compile', action='store_true')
    parser.add_argument('--worker')
    args = parser.parse_args()
    if args.worker:
        worker(args)
    else:
        if args.prepare: prepare(args)
        if args.run: launch(args)
        if args.compile: compile_results(args.output)

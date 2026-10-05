"""Evaluate the completed adaptive sweep; restartable per GPU and result file."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import json
import os
from pathlib import Path
import queue
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
OUTPUT=ROOT/'core/outputs/adaptive-sweep-evaluation-20261005'
SOURCE=Path('/home/patrick/.cache/ray/ray/session_2026-10-04_17-22-09_209409_1503986/artifacts')


def prepare():
    from omegaconf import OmegaConf
    inventory=[]
    for p in SOURCE.glob('**/run/config.yaml'):
        config=OmegaConf.load(p)
        checkpoint=p.parent/'checkpoints/step_00005000.ckpt'
        if config.mode=='train' and config.get('wandb') and checkpoint.exists():
            inventory.append({'name':config.wandb.name,'config':str(p),'checkpoint':str(checkpoint)})
    if len(inventory)!=24:
        raise ValueError(f'Expected 24 completed checkpoints, found {len(inventory)}')
    OUTPUT.mkdir(parents=True,exist_ok=True)
    tasks=[]
    for record in inventory:
        name=record['name']
        policies=['frozen','revisable'] if '-uniform-' in name else ['frozen']
        for policy in policies:
            for bound in (.1,.01):
                label=f'eb_{policy}_{bound:g}'
                dest=OUTPUT/name/label
                dest.mkdir(parents=True,exist_ok=True)
                c=OmegaConf.load(record['config'])
                c.mode='evaluate';c.wandb=None;c.output_dir=str(dest)
                c.evaluation.checkpoint=record['checkpoint'];c.evaluation.use_ema=True
                c.evaluation.device='cuda';c.evaluation.max_examples=2000
                c.loader.eval_batch_size=64
                c.sampler={'_target_':'core.samplers.parallel_entropy_bounded.ParallelEntropyBoundedSampler',
                           'entropy_bound':bound,'freeze':policy=='frozen','steps':180,
                           'greedy_clean_estimate':True}
                OmegaConf.save(OmegaConf.create(OmegaConf.to_container(c,resolve=True)),dest/'config.yaml')
                tasks.append({'name':name,'label':label,'directory':str(dest)})
    baselines=[]
    for p in SOURCE.glob('**/run/results.json'):
        c=OmegaConf.load(p.parent/'config.yaml')
        name=c.wandb.name if c.get('wandb') else None
        if name not in {x['name'] for x in inventory}:continue
        r=json.loads(p.read_text())
        if r.get('checkpoint_step')!=5000 or r.get('num_total')!=2000 or not c.evaluation.use_ema:continue
        steps=r.get('sampler',{}).get('steps')
        if steps not in (32,180):continue
        dest=OUTPUT/name/f'regular_{steps}'
        dest.mkdir(parents=True,exist_ok=True)
        (dest/'results.json').write_text(p.read_text())
        (dest/'source.json').write_text(json.dumps({'results':str(p),'config':str(p.parent/'config.yaml')},indent=2))
        baselines.append((name,steps))
    assert len(set(baselines))==48, f'Missing baseline results: {len(set(baselines))}/48'
    (OUTPUT/'inventory.json').write_text(json.dumps(inventory,indent=2))
    (OUTPUT/'tasks.json').write_text(json.dumps(tasks,indent=2))
    print(f'{len(inventory)} checkpoints; {len(set(baselines))} existing evaluations; {len(tasks)} EB evaluations',flush=True)


def add_low_nfe():
    from omegaconf import OmegaConf
    tasks=json.loads((OUTPUT/'tasks.json').read_text())
    for record in json.loads((OUTPUT/'inventory.json').read_text()):
        if any(x in record['name'] for x in ('-masked-', '-uniform-')):
            continue
        for nfe in (4,8):
            label=f'regular_{nfe}nfe'
            if any(t['name']==record['name'] and t['label']==label for t in tasks):
                continue
            dest=OUTPUT/record['name']/label
            dest.mkdir(parents=True,exist_ok=True)
            c=OmegaConf.load(record['config'])
            c.mode='evaluate';c.wandb=None;c.output_dir=str(dest)
            c.evaluation.checkpoint=record['checkpoint'];c.evaluation.use_ema=True
            c.evaluation.device='cuda';c.evaluation.max_examples=2000
            c.sampler.steps=nfe-1
            OmegaConf.save(OmegaConf.create(OmegaConf.to_container(c,resolve=True)),dest/'config.yaml')
            tasks.append({'name':record['name'],'label':label,'directory':str(dest)})
    (OUTPUT/'tasks.json').write_text(json.dumps(tasks,indent=2))


def compile_results():
    rows=[]
    for p in OUTPUT.glob('*/*/results.json'):
        r=json.loads(p.read_text())
        experimental = p.parent.name.startswith('eb_') and not any(x in p.parents[1].name for x in ('-masked-', '-uniform-'))
        rows.append({'run':p.parents[1].name,'sampler':('experimental_' if experimental else '') + p.parent.name,
                     'experimental':experimental,
                     'solved_percent':100*r['valid_solution_rate'],
                     'digit_percent':100*r['digit_accuracy'],'correct':r['num_correct'],
                     'puzzles':r['num_total'],'mean_nfe':r['mean_nfe'],'max_nfe':r['max_nfe'],
                     'at_step_limit':sum(x.get('nfe',0)>=180 for x in r.get('records',[])),
                     'seconds':r['evaluation_seconds'],'checkpoint_step':r['checkpoint_step'],
                     'result_file':str(p)})
    rows.sort(key=lambda r:(r['run'],r['sampler']))
    (OUTPUT/'summary.json').write_text(json.dumps(rows,indent=2))
    if rows:
        with (OUTPUT/'summary.csv').open('w') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    labels=['regular_4nfe','regular_8nfe','regular_32','regular_180','eb_frozen_0.1','eb_frozen_0.01','eb_revisable_0.1','eb_revisable_0.01']
    header=['Run','Regular 4 NFE','Regular 8 NFE','Regular 32 steps','Regular 180 steps','Frozen .1','Frozen .01','Revisable .1','Revisable .01']
    lines=['# Adaptive Sudoku sweep evaluation','',
           '5,000-step EMA checkpoints; 48,000 training puzzles; 2,000 validation puzzles; seed 1. Cells show solved percentage / mean model calls. Regular 32/180-step runs use 33/181 calls including final readout; the 4/8-NFE runs use exactly 4/8 calls. EB uses greedy commitment and at most 180 transition steps (plus an endpoint readout when unfinished).', '',
           'Regular sampling retains each model’s configured update: Gaussian Euler, spherical manifold update, or simplex stochastic bridge. Frozen EB is cumulative; revisable uniform-hybrid EB re-corrupts released continuous commitments at destination time and stops only on simultaneous acceptance. No hidden-feedback resets were applied.', '',
           '| '+' | '.join(header)+' |','|'+'|'.join(['---']*len(header))+'|']
    for name in sorted({r['run'] for r in rows}):
        subset={r['sampler']:r for r in rows if r['run']==name}
        cells=[name]
        for label in labels:
            r=subset.get(label);cells.append(f"{r['solved_percent']:.2f}% / {r['mean_nfe']:.2f}" if r else '—')
        lines.append('| '+' | '.join(cells)+' |')
    lines+=['','`hidden` is recurrent hidden self-conditioning; `token` is intermediate token feedback. Continuous-only frozen EB is an added inference policy, not a paper-native sampler. New EB evaluations and cached regular evaluations retain their full per-puzzle results and source paths.','',f'{len(rows)} evaluation results currently available.']
    lines += ['', 'Continuous-only commitment experiments are excluded from the main EB columns; see [experimental results](experimental-continuous-commitment.md).']
    (OUTPUT/'report.md').write_text('\n'.join(lines)+'\n')
    experimental_lines = ['# Experimental continuous-only commitment sampling', '',
        'These are not native Gaussian/spherical/simplex samplers and contain no discrete hybrid view. An external entropy-selected mask freezes continuous states at clean token representations. This changes the continuous process sampling dynamics. Retained for transparency, separately from the main comparison.', '',
        '| Model | Bound | Solved (%) | Mean model calls |', '|---|---:|---:|---:|']
    for r in rows:
        if r['experimental']:
            experimental_lines.append(f"| {r['run']} | {r['sampler'].rsplit('_',1)[-1]} | {r['solved_percent']:.2f} | {r['mean_nfe']:.2f} |")
    (OUTPUT/'experimental-continuous-commitment.md').write_text('\n'.join(experimental_lines)+'\n')
    return rows


def launch(gpus):
    tasks=json.loads((OUTPUT/'tasks.json').read_text());pending=queue.Queue()
    for task in tasks:
        if not (Path(task['directory'])/'results.json').exists():pending.put(task)
    failures=[]
    def worker(gpu):
        while True:
            try:task=pending.get_nowait()
            except queue.Empty:return
            dest=Path(task['directory']);env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='2',PYTHONUTF8='1')
            print('START',gpu,task['name'],task['label'],flush=True)
            with (dest/'evaluation.log').open('w') as log:
                r=subprocess.run([sys.executable,__file__,'--worker',str(dest/'config.yaml')],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
            print('FINISH',gpu,task['name'],task['label'],r.returncode,flush=True)
            if r.returncode:failures.append(task)
            pending.task_done()
    with ThreadPoolExecutor(max_workers=len(gpus)) as executor:
        list(executor.map(worker,gpus))
    (OUTPUT/'failures.json').write_text(json.dumps(failures,indent=2))
    compile_results()
    if failures:raise RuntimeError(f'{len(failures)} evaluation failures')

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare',action='store_true');parser.add_argument('--run',action='store_true')
    parser.add_argument('--compile',action='store_true');parser.add_argument('--worker')
    parser.add_argument('--low-nfe',action='store_true')
    parser.add_argument('--gpus',default='1,2,3,4,5,6');args=parser.parse_args()
    if args.worker:
        from omegaconf import OmegaConf
        from core.training.runner import run
        run(OmegaConf.load(args.worker))
    else:
        if args.prepare:prepare()
        if args.low_nfe:add_low_nfe()
        if args.run:launch([int(x) for x in args.gpus.split(',')])
        if args.compile:compile_results()

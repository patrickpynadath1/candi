#!/usr/bin/env python3
"""Train official FMLM+ unchanged, then evaluate our shared Sudoku battery."""
import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'external/posterior-refinement'
DEPS=ROOT/'external/posterior-refinement-deps'
PYTHON=ROOT/'.venv/bin/python'


def write_json(p,data):
    temp=p.with_suffix('.tmp');temp.write_text(json.dumps(data,indent=2));temp.replace(p)


def train_command(output,cache,difficulty,steps=50000):
    # Method-specific settings copied from upstream scripts/train/sudoku.sh.
    return [str(PYTHON),'-u',str(SOURCE/'main.py'),
        'mode=train','seed=1','data=sudoku',f'data.cache_dir={cache}',
        f'data.difficulty={difficulty}','data.num_train=200000','data.num_valid=10000',
        'data.data_seed=42','model=mini','model.length=180','algo=fmlm_plus',
        'algo.sample_clean=uniform','algo.double_temb=True','algo.skip_time_reparam=True',
        'algo.diagonal_fraction=0.5','algo.boundary_prob=32','algo.add_boundary=True',
        'algo.distillation_method=PSD','algo.p_distill=0.0',
        'algo.distill_teacher_ckpt_path=null','algo.init_teacher_ckpt_path=null',
        'algo.init_teacher_kind=flm_plus','algo.diag_teacher_ckpt_path=null',
        'strategy.find_unused_parameters=false','training.antithetic_sampling=True',
        'loader.global_batch_size=256','loader.batch_size=256','loader.eval_batch_size=256',
        'loader.num_workers=2','trainer.devices=1','trainer.num_nodes=1',
        'sampling.num_sample_batches=1','sampling.method=block-diffusion',
        'sampling.block_length=180','sampling.steps_per_block=180',
        f'trainer.max_steps={steps}','trainer.precision=bf16',
        'trainer.val_check_interval=5000','trainer.limit_val_batches=1',
        'trainer.num_sanity_val_steps=1','+trainer.check_val_every_n_epoch=null',
        'optim.lr=3e-4','eval.compute_generative_perplexity=False',
        'eval.generate_samples=True','+eval.sflm_sudoku_num_eval=256','wandb=null',
        'callbacks.checkpoint_every_n_steps.every_n_train_steps=5000',
        'callbacks.checkpoint_every_n_steps.save_last=True',
        f'checkpointing.save_dir={output}','checkpointing.resume_from_ckpt=True',
        f'hydra.run.dir={output}']


def environment(gpu):
    return dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='1',
        TOKENIZERS_PARALLELISM='false',PYTHONUNBUFFERED='1',
        PYTHONPATH=str(DEPS),HF_HUB_OFFLINE='1',DIT_USE_COMPILE='0')


def report(output, states, revision):
    rows=[]
    for path in sorted((output/'evaluations').glob('*/*/results.json')):
        result=json.loads(path.read_text())
        meta=json.loads((path.parent/'evaluation.json').read_text())
        rows.append(dict(trained_on=meta['trained_on'],clues=meta['clues'],
            checkpoint_step=meta['checkpoint_step'],num_total=result['num_total'],
            **{k:result[k] for k in ('accuracy','valid_solution_rate','clue_consistency_rate',
                'mean_nfe','unique_outputs','unique_valid_outputs')},result_path=str(path)))
    if rows:
        with (output/'summary.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    lines=['# Official Posterior Refinement on our Sudoku datasets','',
        'Official repository: https://github.com/MananAg007/posterior-refinement',
        f'Pinned revision: `{revision}`. Looped Flows was skipped at the user’s request because official code was unavailable.','',
        'Two from-scratch FMLM+ runs: 200,000 training examples each (easy and hard), 50,000 optimizer steps, batch 256, seed 1. The official mini DiT architecture, FMLM+ objective, optimizer and learning-rate settings are retained. Native Sudoku training uses PSD, uniform clean-context sampling, diagonal fraction 0.5 and boundary probability parameter 32.','',
        'Adaptations: existing dataset paths and sizes, 50k instead of the upstream script’s 20k steps, a single GPU, two loader workers, local logging, and checkpoints every 5k steps. No upstream Python source modifications. Missing dependencies are isolated in external/posterior-refinement-deps.','',
        'Inference exactly follows scripts/eval/sudoku.sh: EMA, posterior refinement, four rounds, one flow step per round, confidence threshold 0.999, uniform-budget fallback, fresh noise. Batch size 200. No EB or custom ancestral sampler and no inference-clock overrides. NFE is measured by a passive backbone hook.','',
        'Each trained checkpoint is evaluated on all 10,000 hard validation puzzles with 30, 16, 12, 8, and 0 clues. Thinning uses the same nested subsets as the previous sweep (seed 420000 + validation index). Targets are hidden from generation. Validity includes row/column/box rules, retained-clue consistency, and output format. Zero-clue evaluation also records distinct outputs and distinct valid grids.','',
        f'[Queue status]({output}/status.json) · [Full plan]({output}/plan.json) · [Results CSV]({output}/summary.csv)','',
        '| Job | Status |','|---|---|']
    lines += [f'| {name} | {state} |' for name,state in states.items()]
    lines += ['','## Results','','| Training | Clues | Exact accuracy | Valid solutions | Mean NFE | Distinct valid grids |','|---|---:|---:|---:|---:|---:|']
    for r in rows:
        lines.append(f"| {r['trained_on']} | {r['clues']} | {100*r['accuracy']:.2f}% | {100*r['valid_solution_rate']:.2f}% | {r['mean_nfe']:.2f} | {r['unique_valid_outputs']} |")
    if not rows:lines+=['','No full evaluation results yet.']
    (ROOT/'docs/posterior_refinement_sudoku_results.md').write_text('\n'.join(lines)+'\n')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--gpu',default='6')
    p.add_argument('--smoke',action='store_true')
    p.add_argument('--detach',action='store_true')
    args=p.parse_args();args.output=args.output.absolute();args.output.mkdir(parents=True,exist_ok=True)
    if args.detach:
        with (args.output/'supervisor.log').open('a') as f:
            proc=subprocess.Popen([str(PYTHON),__file__,*[x for x in sys.argv[1:] if x!='--detach']],cwd=ROOT,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        print(proc.pid);return
    import fcntl
    lock=(args.output/'supervisor.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    cache=args.output/'data';cache.mkdir(exist_ok=True)
    for difficulty in ('easy','hard'):
        target=ROOT/f'data_cache/sudoku_{difficulty}_train200000_valid10000_seed42'
        assert (target/'train').is_dir() and (target/'validation').is_dir()
        link=cache/f'sflm_sudoku_{difficulty}_train200000_valid10000_seed42'
        if not link.exists():link.symlink_to(target,target_is_directory=True)
        assert link.resolve()==target.resolve()
    revision=subprocess.check_output(['git','-C',str(SOURCE),'rev-parse','HEAD'],text=True).strip()
    tasks=[]
    for difficulty in ('easy','hard'):
        out=args.output/'runs'/difficulty
        train=train_command(out,cache,difficulty,2 if args.smoke else 50000)
        if args.smoke:
            train += ['trainer.val_check_interval=2','callbacks.checkpoint_every_n_steps.every_n_train_steps=2']
        tasks.append(dict(name=f'train_{difficulty}',command=train,cwd=str(SOURCE),output=str(out)))
        for clues in (30,16,12,8,0):
            evaluation=args.output/'evaluations'/difficulty/f'clues_{clues}'
            cmd=[str(PYTHON),str(ROOT/'new_scripts/eval_posterior_refinement_sudoku.py'),
                 '--checkpoint',str(out/'checkpoints/last.ckpt'),'--output',str(evaluation),
                 '--clues',str(clues),'--expected-step',str(2 if args.smoke else 50000),
                 '--max-examples',str(32 if args.smoke else 10000)]
            tasks.append(dict(name=f'eval_{difficulty}_{clues}',command=cmd,cwd=str(SOURCE),output=str(evaluation)))
        if args.smoke:break
    plan=dict(repository='https://github.com/MananAg007/posterior-refinement',revision=revision,
        source=str(SOURCE),gpu=args.gpu,smoke=args.smoke,tasks=tasks,
        adaptation='existing 200k/10k datasets, 50k steps, local logging/checkpoint frequency; upstream model/loss/sampler unchanged',
        inference=dict(method='refinement',rounds=4,flow_steps=1,threshold=0.999,top_k='uniform_budget',fresh_noise=True))
    planfile=args.output/'plan.json'
    if planfile.exists() and json.loads(planfile.read_text())!=plan:raise ValueError('Existing plan differs')
    write_json(planfile,plan)
    states={t['name']:'queued' for t in tasks}
    def save():
        write_json(args.output/'status.json',dict(pid=os.getpid(),jobs=states))
        if not args.smoke:report(args.output,states,revision)
    save()
    for task in tasks:
        out=Path(task['output']);out.mkdir(parents=True,exist_ok=True)
        done=out/'complete.json'
        if done.exists():states[task['name']]='complete';save();continue
        states[task['name']]='running';save();print(task['name'],'starting',flush=True)
        with (out/'run.log').open('a') as f:
            result=subprocess.run(task['command'],cwd=task['cwd'],env=environment(args.gpu),stdout=f,stderr=subprocess.STDOUT)
        if result.returncode:
            states[task['name']]=f'failed: exit {result.returncode}';save()
            raise SystemExit(result.returncode)
        write_json(done,dict(completed_at=time.time(),command=task['command']))
        states[task['name']]='complete';save();print(task['name'],'complete',flush=True)

if __name__=='__main__':main()

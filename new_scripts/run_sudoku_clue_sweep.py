#!/usr/bin/env python3
"""Run paired easy/hard-trained clue sweeps on four specified GPUs."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--campaign',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--gpus',nargs=4,required=True)
    p.add_argument('--detach',action='store_true')
    args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    if args.detach:
        with (args.output/'supervisor.log').open('a') as log:
            proc=subprocess.Popen([sys.executable,__file__,*[a for a in sys.argv[1:] if a!='--detach']],cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        print(proc.pid)
        return
    def worker(pair):
        clues,gpu=pair
        for trained in ('hard','easy'):
            out=args.output/f'clues_{clues}'/trained
            cmd=[sys.executable,str(ROOT/'new_scripts/eval_sudoku_generalization.py'),
                 '--campaign',str(args.campaign),'--output',str(out),'--gpus',gpu,
                 '--trained-on',trained,'--clues',str(clues),'--entropy-bound','0.01']
            if trained=='easy':
                cmd+=['--baseline-checkpoints',str(args.campaign.parent/'generalization/available_baseline_checkpoints')]
            print('Starting',clues,trained,flush=True)
            result=subprocess.run(cmd+['--execute'],cwd=ROOT)
            print('Finished',clues,trained,'exit',result.returncode,flush=True)
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(worker,zip((16,12,8,0),args.gpus)))

if __name__=='__main__':
    main()

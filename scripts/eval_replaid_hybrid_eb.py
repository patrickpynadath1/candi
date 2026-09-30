"""Paired checkpoint evaluation of ancestral and entropy-bounded hybrid sampling."""
import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import datasets
from hydra import compose, initialize_config_dir
import numpy as np
import torch
import wandb

import main
from algos.replaid_hybrid import RePlaidHybrid
from tasks.sudoku import SudokuTokenizer
from tasks.sudoku_callback import generate_sudoku_records, summarize_sudoku_records


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--data', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    with initialize_config_dir(config_dir=str(Path(__file__).resolve().parents[1]/'configs'),version_base=None):
        config = compose(config_name='config',overrides=['+experiment=sudoku_replaid_hybrid_matched'])
    config.eval.checkpoint_path = args.checkpoint
    tokenizer=SudokuTokenizer()
    checkpoint=torch.load(args.checkpoint,map_location='cpu',weights_only=False,mmap=True)
    step=int(checkpoint['global_step'])
    del checkpoint
    model=main._load_from_checkpoint(RePlaidHybrid,config,tokenizer).cuda().eval()
    data=datasets.load_from_disk(args.data)['validation'].with_format('torch')
    indices=np.random.default_rng(42).choice(len(data),size=min(128,len(data)),replace=False).tolist()
    run=wandb.init(project='replaid-sudoku',entity='rz-text-dlp',job_type='sampler-eval',
        name=f'hybrid-eb-comparison-step{step}',config={'checkpoint':args.checkpoint,'checkpoint_step':step,
        'ema':True,'seed':42,'indices':indices,'steps':180,'self_condition_rate':float(config.algo.self_condition_rate)})
    print('WANDB',run.url,flush=True)
    results=[]
    for name,bound in [('ancestral',None),('eb_0',0.0),('eb_0.1',0.1),('eb_1',1.0)]:
        torch.manual_seed(42)
        sampler='replaid_hybrid_ddpm' if bound is None else 'replaid_hybrid_entropy_bounded'
        overrides=None if bound is None else {'entropy_bound':bound}
        started=time.monotonic()
        records=generate_sudoku_records(model,data,tokenizer,16,180,indices,sampler,overrides)
        metrics=summarize_sudoku_records(records)
        metrics.pop("records", None)
        result={'sampler':name,'checkpoint_step':step,'seconds':time.monotonic()-started,**metrics}
        results.append(result)
        run.log({f'{name}/{k}':v for k,v in result.items() if isinstance(v,(int,float))})
        (output/f'{name}.json').write_text(json.dumps(records,indent=2))
        (output/'summary.json').write_text(json.dumps({'wandb':run.url,'results':results},indent=2))
        print(json.dumps(result),flush=True)
    run.finish()


if __name__ == '__main__':
    run()

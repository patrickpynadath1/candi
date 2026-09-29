#!/usr/bin/env python3
"""Score the unmodified official FMLM+ posterior-refinement sampler on our data."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'external/posterior-refinement'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--clues',type=int,choices=(30,16,12,8,0),required=True)
    p.add_argument('--expected-step',type=int,default=50000)
    p.add_argument('--max-examples',type=int,default=10000)
    args=p.parse_args()
    sys.path.insert(0,str(SOURCE));sys.path.append(str(ROOT))
    import main as upstream
    import torch
    from omegaconf import OmegaConf
    from datasets import load_from_disk
    from tqdm import trange
    from tasks.sudoku import sudoku_metrics
    from new_scripts.eval_sudoku_fewer_clues import thin_tokens
    import lightning as L
    L.seed_everything(1)
    torch.backends.cuda.matmul.allow_tf32=True
    torch.backends.cudnn.allow_tf32=True
    config=OmegaConf.load(args.checkpoint.parent.parent/'.hydra/config.yaml')
    config.eval.checkpoint_path=str(args.checkpoint)
    config.eval.disable_ema=False
    config.sampling.method='refinement'
    config.sampling.steps=4
    config.sampling.refinement_rounds=4
    config.sampling.refinement_flow_steps=1
    config.sampling.refinement_threshold=0.999
    config.sampling.refinement_fresh_noise=True
    config.sampling.refinement_top_k='uniform_budget'
    checkpoint=torch.load(args.checkpoint,map_location='cpu',mmap=True)
    step=checkpoint['global_step']
    assert step==args.expected_step,(step,args.expected_step)
    assert checkpoint.get('ema') is not None
    del checkpoint
    tokenizer=upstream.dataloader.get_tokenizer(config)
    model=upstream._load_from_checkpoint(upstream.algo.FMLMPlus,config,tokenizer).cuda()
    model.eval();model._eval_mode()
    # Count network calls without changing the sampler or its outputs.
    calls=[0]
    def count_calls(module,inputs,output):calls[0]+=1
    handle=model.backbone.register_forward_hook(count_calls)
    dataset=load_from_disk(str(ROOT/'data_cache/sudoku_hard_train200000_valid10000_seed42'))['validation'].with_format(None)
    assert len(dataset)==10000
    count=min(args.max_examples,len(dataset));batch_size=200  # official eval recipe
    args.output.mkdir(parents=True,exist_ok=True)
    OmegaConf.save(config,args.output/'config.yaml')
    metadata=dict(checkpoint=str(args.checkpoint),checkpoint_step=step,trained_on=config.data.difficulty,
        evaluated_on='hard',clues=args.clues,seed=1,batch_size=batch_size,ema=True,
        sampler='official FMLM+ sample_refinement',rounds=4,flow_steps=1,
        threshold=0.999,top_k='uniform_budget',fresh_noise=True,
        data_source=str(ROOT/'data_cache/sudoku_hard_train200000_valid10000_seed42'),
        clue_selection='same nested subsets as prior sweep: Random(420000 + index)',
        source=str(SOURCE),command=sys.argv)
    (args.output/'evaluation.json').write_text(json.dumps(metadata,indent=2))
    records=[];nfes=[];digest=hashlib.sha256();torch.cuda.synchronize();started=time.monotonic()
    for start in trange(0,count,batch_size,desc=f'FMLM+ {config.data.difficulty}→hard, {args.clues} clues'):
        examples=[]
        for i in range(start,min(start+batch_size,count)):
            raw=dataset[i]['input_ids'];tokens=thin_tokens(raw,i,args.clues)
            assert tokens[90:]==raw[90:]
            assert sum(1<=v<=9 for v in tokens[1:90])==args.clues
            digest.update(bytes(tokens[:90]));examples.append(tokens)
        batch=torch.tensor(examples,dtype=torch.long,device='cuda')
        # Match upstream conditional evaluation: first 91 tokens are the prompt.
        # Zero unknown positions so target answers cannot be exposed to inference.
        prompt=batch.clone();prompt[:,91:]=0
        before=calls[0]
        with torch.inference_mode():
            samples=model.generate_samples(num_samples=len(examples),num_steps=4,
                prompt_tokens=prompt,prompt_lens=torch.full((len(examples),),91,device='cuda',dtype=torch.long))
        nfe=calls[0]-before
        assert nfe==4, f'Native 4-round sampler used {nfe} forward passes'
        nfes.extend([nfe]*len(examples))
        assert torch.equal(samples[:,:91],batch[:,:91])
        for generated,target,example in zip(samples[:,91:].cpu(),batch[:,91:].cpu(),batch.cpu()):
            records.append(dict(generated=tokenizer.decode(generated),ground_truth=tokenizer.decode(target),
                **sudoku_metrics(generated,target,example[1:90])))
    handle.remove();torch.cuda.synchronize();elapsed=time.monotonic()-started
    outputs=Counter(r['generated'] for r in records)
    result=dict(num_total=count,num_correct=sum(r['exact'] for r in records),
        accuracy=sum(r['exact'] for r in records)/count,
        valid_solution_rate=sum(r['valid_solution'] for r in records)/count,
        valid_grid_rate=sum(r['valid_grid'] for r in records)/count,
        clue_consistency_rate=sum(r['clue_consistency'] for r in records)/count,
        digit_accuracy=sum(r['digit_accuracy'] for r in records)/count,
        mean_nfe=sum(nfes)/count,evaluation_seconds=elapsed,
        unique_outputs=len(outputs),unique_valid_outputs=len({r['generated'] for r in records if r['valid_solution']}),
        most_frequent_output_count=outputs.most_common(1)[0][1],prompt_sha256=digest.hexdigest(),records=records)
    temp=args.output/'results.tmp';temp.write_text(json.dumps(result));temp.replace(args.output/'results.json')
    print(json.dumps({k:v for k,v in result.items() if k!='records'},indent=2),flush=True)

if __name__=='__main__':main()

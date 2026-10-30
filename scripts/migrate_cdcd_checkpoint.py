"""Create an explicit CDCD-resume checkpoint and matching launch config.

python -m scripts.migrate_cdcd_checkpoint SOURCE DEST --sampler-config CONFIG
The source is retained. Both destination paths must be new. Training stays off.
"""
import argparse
from pathlib import Path
import torch
from omegaconf import OmegaConf
from core.training.cdcd_migration import migrate_checkpoint


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source',type=Path)
    parser.add_argument('destination',type=Path)
    parser.add_argument('--sampler-config',type=Path,required=True)
    args=parser.parse_args()
    yaml_path=args.destination.with_suffix('.yaml')
    if args.destination.exists() or yaml_path.exists():
        parser.error('destination checkpoint/config already exists')
    sampler=OmegaConf.load(args.sampler_config)
    if not sampler.get('adaptive',False):
        parser.error('sampler config must explicitly enable adaptive: true')
    original=torch.load(args.source,map_location='cpu',weights_only=False,mmap=True)
    result=migrate_checkpoint(original,sampler)
    config=OmegaConf.create(result['hyper_parameters']['config'])
    config.checkpoint.resume=str(args.destination.resolve())
    result['hyper_parameters']['config']=OmegaConf.to_container(config,resolve=True)
    args.destination.parent.mkdir(parents=True,exist_ok=True)
    # Exclusive create: never overwrite a training checkpoint.
    with args.destination.open('xb') as stream:torch.save(result,stream)
    with yaml_path.open('x') as stream:OmegaConf.save(config,stream)
    print(f'Created {args.destination} and {yaml_path}; sampler reset, model step {result["global_step"]} retained.')


if __name__=='__main__':main()

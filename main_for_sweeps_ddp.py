from tqdm import tqdm

import json
import os

from typing import List, Dict, Tuple
import fsspec
import hydra
import lightning as L
import omegaconf
import rich.syntax
import rich.tree
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import algo
import other_dataloader as dataloader
import utils
import pdb

omegaconf.OmegaConf.register_new_resolver("cwd", os.getcwd)
omegaconf.OmegaConf.register_new_resolver("device_count", torch.cuda.device_count)
omegaconf.OmegaConf.register_new_resolver("eval", eval)
omegaconf.OmegaConf.register_new_resolver("div_up", lambda x, y: (x + y - 1) // y)

def setup_ddp_torchrun():
    """Initialize DDP when using torchrun"""
    dist.init_process_group("nccl")
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    return rank, local_rank, world_size

def cleanup_ddp():
    """Clean up the process group"""
    dist.destroy_process_group()

def _load_from_checkpoint(diffusion_model, config, tokenizer, device):
    if "hf" in config.algo.backbone:
        return diffusion_model(config, tokenizer=tokenizer).to(device)
    
    model = diffusion_model.load_from_checkpoint(
        config.eval.checkpoint_path, tokenizer=tokenizer, config=config
    )
    return model.to(device)

def get_distributed_dataloader(config, tokenizer, rank, world_size):
    """Create distributed dataloader"""
    train_ds, valid_ds = dataloader.get_dataloaders(config, tokenizer)
    
    # Create distributed sampler for validation dataset
    sampler = DistributedSampler(
        valid_ds.dataset if hasattr(valid_ds, 'dataset') else valid_ds,
        num_replicas=world_size,
        rank=rank,
        shuffle=False  # For evaluation, we don't want to shuffle
    )
    
    # Create new dataloader with distributed sampler
    if hasattr(valid_ds, 'dataset'):
        dataset = valid_ds.dataset
        batch_size = valid_ds.batch_size
        num_workers = getattr(valid_ds, 'num_workers', 0)
        collate_fn = getattr(valid_ds, 'collate_fn', None)
    else:
        dataset = valid_ds
        batch_size = 1
        num_workers = 0
        collate_fn = None
    
    distributed_valid_ds = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    return train_ds, distributed_valid_ds

@L.pytorch.utilities.rank_zero_only
def _print_config(
    config: omegaconf.DictConfig, resolve: bool = True, save_cfg: bool = True
) -> None:
    """Prints content of DictConfig using Rich library and its tree structure."""
    style = "dim"
    tree = rich.tree.Tree("CONFIG", style=style, guide_style=style)

    fields = config.keys()
    for field in fields:
        branch = tree.add(field, style=style, guide_style=style)
        config_section = config.get(field)
        branch_content = str(config_section)
        if isinstance(config_section, omegaconf.DictConfig):
            branch_content = omegaconf.OmegaConf.to_yaml(
                config_section, resolve=resolve
            )
        branch.add(rich.syntax.Syntax(branch_content, "yaml"))
    rich.print(tree)
    if save_cfg:
        with fsspec.open(
            "{}/config_tree.txt".format(config.checkpointing.save_dir), "w"
        ) as fp:
            rich.print(tree, file=fp)

def _conditional_gen_loop_ddp(model, config, logger, tokenizer, valid_ds, rank, world_size):
    """DDP version of conditional generation loop"""
    if rank == 0:
        logger.info("Starting Conditional Eval with DDP.")
    
    # Calculate samples per GPU
    total_samples = config.eval.num_conditional_samples
    samples_per_gpu = total_samples // world_size
    if rank < total_samples % world_size:
        samples_per_gpu += 1
    
    model_length = config.model.length
    counter = 0
    
    # Compile model
    model.module.backbone = torch.compile(model.module.backbone)
    
    # Use smaller batch size per GPU
    samples_to_eval = []
    reference_samples = []
    
    # Only show progress bar on rank 0
    pbar = None
    if rank == 0:
        pbar = tqdm(total=total_samples, desc="Generating samples (all GPUs)")
    
    local_pbar = tqdm(total=samples_per_gpu, desc=f"GPU {rank} generating", position=rank)
    
    while counter < samples_per_gpu:
        batch = next(iter(valid_ds))
        clean_sequences = batch['input_ids'].cuda(rank)
        eval_batch_size = clean_sequences.shape[0]
        batch_size = eval_batch_size
        
        # Calculate how many samples to generate this iteration
        remaining_samples = min(batch_size, samples_per_gpu - counter)
        
        # Generate prompt tokens and mask
        prompt_tokens, prompt_mask = mask_clean_samples(
            clean_sequences, config.eval.prompt_type, 
            remaining_samples, model_length, config.eval.revealed_fraction
        )
        prompt_tokens = prompt_tokens.cuda(rank)
        prompt_mask = prompt_mask.cuda(rank)
        
        # Generate samples on this GPU
        samples = model.module.generate_samples(
            num_samples=remaining_samples,
            num_steps=config.sampling.steps,
            prompt_tokens=prompt_tokens,
            prompt_mask=prompt_mask
        )
        
        # Record metrics
        model.module.metrics.record_entropy(samples)
        
        gen_sequences = tokenizer.batch_decode(samples)
        
        clean_samples = tokenizer.batch_decode(clean_sequences)
        clean_samples = [s.replace("<|endoftext|>", "") for s in clean_samples]
        
        # Update counters and collect samples
        counter += remaining_samples
        samples_to_eval.extend(gen_sequences)
        reference_samples.extend(clean_samples[:remaining_samples])
        
        local_pbar.update(remaining_samples)
        
        # Synchronize progress across all GPUs for global progress bar
        if rank == 0 and pbar:
            # Get progress from all GPUs
            progress_tensor = torch.tensor(remaining_samples, device=rank)
            dist.all_reduce(progress_tensor, op=dist.ReduceOp.SUM)
            pbar.update(progress_tensor.item())
    
    local_pbar.close()
    if pbar:
        pbar.close()
    
    # Compute local metrics
    generative_ppl = model.module.metrics.gen_ppl.compute().item()
    entropy = model.module.metrics.sample_entropy.compute().item()
    
    # Gather results from all GPUs
    local_results = {
        'samples': samples_to_eval,
        'references': reference_samples,
        'entropy': entropy,
        'rank': rank,
        'num_samples': len(samples_to_eval)
    }
    
    # Collect results from all processes
    all_results = [None] * world_size
    dist.all_gather_object(all_results, local_results)
    
    # Only rank 0 processes and returns the final results
    if rank == 0:
        combined_samples = []
        combined_references = []
        total_ppl = 0
        total_entropy = 0
        total_sample_count = 0
        
        for result in all_results:
            combined_samples.extend(result['samples'])
            combined_references.extend(result['references'])
            total_sample_count += result['num_samples']
        
        # Weighted average for metrics
        avg_entropy = total_entropy / total_sample_count if total_sample_count > 0 else 0
        
        metrics = {
            'entropy': avg_entropy,
            'gen_samples': combined_samples,
            'ref_samples': combined_references
        }
        
        print(f"Sample entropy: {avg_entropy}")
        print(f"Total samples generated: {total_sample_count}")
        
        return metrics
    
    return None

def _sweep_ddp(diffusion_model, config, tokenizer, logger, rank, world_size, is_conditional=False):
    """DDP version of parameter sweep"""
    temps_to_use = torch.linspace(.7, 1.0, steps=15).tolist()
    steps_to_use = [8, 16, 32, 64, 128, 256, 512]

    # Get distributed dataloader
    _, valid_ds = get_distributed_dataloader(config, tokenizer, rank, world_size)
    
    # Load model on this GPU
    model = _load_from_checkpoint(
        diffusion_model=diffusion_model, config=config, tokenizer=tokenizer, device=rank
    )
    
    # Wrap with DDP
    model = DDP(model, device_ids=[rank], output_device=rank)
    
    for steps in steps_to_use:
        config.sampling.steps = steps
        
        for temp in temps_to_use:
            model.module.temp = temp
            model.module.metrics.reset()
            
            res = _conditional_gen_loop_ddp(
                model, config, logger, tokenizer, valid_ds, rank, world_size
            )
            
            # Only rank 0 saves results
            if rank == 0 and res is not None:
                res['steps'] = steps
                res['temperature'] = temp
                samples_path = config.eval.generated_samples_path
                samples_path = samples_path.replace('.json', f'_steps{steps}_temp{temp:.3f}.json')
                with fsspec.open(samples_path, "w") as f:
                    json.dump(res, f, indent=4)
                print("Samples saved at:", samples_path)

def run_distributed_main(rank, world_size, config):
    """Main function for each distributed process"""
    setup_ddp(rank, world_size)
    
    try:
        # Set same seed for all processes for reproducibility
        L.seed_everything(config.seed)
        
        # Set up logging (only on rank 0)
        if rank == 0:
            logger = utils.get_logger(__name__)
            logger.info(f"Starting DDP evaluation on {world_size} GPUs")
        else:
            logger = None
            
        tokenizer = dataloader.get_tokenizer(config)
        
        # Get diffusion model class
        if config.algo.name == "ar":
            diffusion_model = algo.AR
        elif config.algo.name == "mdlm":
            diffusion_model = algo.MDLM
        elif config.algo.name == "duo_base":
            diffusion_model = algo.DUO_BASE
        elif config.algo.name == "d3pm":
            diffusion_model = algo.D3PMAbsorb
        elif config.algo.name == "sedd":
            diffusion_model = algo.SEDDAbsorb
        elif config.algo.name == "duo":
            diffusion_model = algo.DUO
        elif config.algo.name == "distillation":
            diffusion_model = algo.Distillation
        elif config.algo.name == "ot-finetune":
            diffusion_model = algo.OptimalTransportFinetune
        elif config.algo.name == "candi":
            diffusion_model = algo.CANDI
        else:
            raise ValueError(f"Invalid algorithm name: {config.algo.name}")
        
        # Run the sweep
        if config.mode == "conditional_sweep":
            _sweep_ddp(diffusion_model, config, tokenizer, logger, rank, world_size, is_conditional=True)
        else:
            _sweep_ddp(diffusion_model, config, tokenizer, logger, rank, world_size, is_conditional=False)
            
    finally:
        cleanup_ddp()

def mask_clean_samples(clean_sequence, prompt_type, batch_size, model_length, revealed_fraction=.1): 

    # Get the actual batch size from clean_sequence
    actual_batch_size = clean_sequence.shape[0]

    # Use the minimum of requested batch_size and actual batch size
    effective_batch_size = min(batch_size, actual_batch_size)

    prompt_tokens = torch.zeros((effective_batch_size, model_length), dtype=torch.long)
    prompt_mask = torch.zeros((effective_batch_size, model_length), dtype=torch.float)

    positions_to_reveal = int(revealed_fraction * model_length)

    if prompt_type == 'prefix':
        prompt_tokens[:, :positions_to_reveal] = clean_sequence[:, :positions_to_reveal]
        prompt_mask[:, :positions_to_reveal] = 1

    if prompt_type == 'suffix': 
        prompt_tokens[:, -positions_to_reveal:] = clean_sequence[:, -positions_to_reveal:]
        prompt_mask[:, -positions_to_reveal:] = 1

    if prompt_type == 'infill': 
        positions_to_reveal = positions_to_reveal // 2
        prompt_tokens[:, :positions_to_reveal] = clean_sequence[:, :positions_to_reveal]
        prompt_tokens[:, -positions_to_reveal:] = clean_sequence[:, -positions_to_reveal:]
        prompt_mask[:, :positions_to_reveal] = 1
        prompt_mask[:, -positions_to_reveal:] = 1
        
    return prompt_tokens, prompt_mask

@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(config):
    """Main entry point for torchrun"""
    # Check if we're running with torchrun
    if "RANK" in os.environ:
        # Running with torchrun - each process handles its own setup
        rank, local_rank, world_size = setup_ddp_torchrun()
        
        try:
            L.seed_everything(config.seed)
            
            if rank == 0:
                logger = utils.get_logger(__name__)
                logger.info(f"Starting DDP evaluation on {world_size} GPUs with torchrun")
            else:
                logger = None
                
            tokenizer = dataloader.get_tokenizer(config)
            
            # Get diffusion model class
            if config.algo.name == "ar":
                diffusion_model = algo.AR
            elif config.algo.name == "mdlm":
                diffusion_model = algo.MDLM
            elif config.algo.name == "duo_base":
                diffusion_model = algo.DUO_BASE
            elif config.algo.name == "d3pm":
                diffusion_model = algo.D3PMAbsorb
            elif config.algo.name == "sedd":
                diffusion_model = algo.SEDDAbsorb
            elif config.algo.name == "duo":
                diffusion_model = algo.DUO
            elif config.algo.name == "distillation":
                diffusion_model = algo.Distillation
            elif config.algo.name == "ot-finetune":
                diffusion_model = algo.OptimalTransportFinetune
            elif config.algo.name == "candi":
                diffusion_model = algo.CANDI
            else:
                raise ValueError(f"Invalid algorithm name: {config.algo.name}")
            
            # Run the sweep
            if config.mode == "conditional_sweep":
                _sweep_ddp(diffusion_model, config, tokenizer, logger, rank, world_size, is_conditional=True)
            else:
                _sweep_ddp(diffusion_model, config, tokenizer, logger, rank, world_size, is_conditional=False)
                
        finally:
            cleanup_ddp()
    else:
        # Running without torchrun - use mp.spawn approach
        world_size = torch.cuda.device_count()
        
        if world_size > 1:
            print(f"Starting DDP with {world_size} GPUs using mp.spawn")
            mp.set_start_method('spawn', force=True)
            mp.spawn(
                run_distributed_main,
                args=(world_size, config),
                nprocs=world_size,
                join=True
            )
        else:
            print("Single GPU detected, running without DDP")
            L.seed_everything(config.seed)
            run_distributed_main(0, 1, config)

if __name__ == "__main__":
    # Ensure we're in the main process
    main()

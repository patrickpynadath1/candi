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
import mauve
import evaluate
import algo
import other_dataloader as dataloader
import utils
import pdb
import numpy as np
from text_metrics import evaluate_text_metrics
# from lm_eval.models.huggingface import HFLM
# from lm_eval import evaluator
# from hugging_face_wrappers import DiffusionEvalWrapper, CandiEvalWrapper


omegaconf.OmegaConf.register_new_resolver("cwd", os.getcwd)
omegaconf.OmegaConf.register_new_resolver("device_count", torch.cuda.device_count)
omegaconf.OmegaConf.register_new_resolver("eval", eval)
omegaconf.OmegaConf.register_new_resolver("div_up", lambda x, y: (x + y - 1) // y)

def _load_from_checkpoint(diffusion_model, config, tokenizer):
    if "hf" in config.algo.backbone:
        return diffusion_model(config, tokenizer=tokenizer).to("cuda")
    
    
    model = diffusion_model.load_from_checkpoint(
        config.eval.checkpoint_path, tokenizer=tokenizer, config=config
    )
    return model



@L.pytorch.utilities.rank_zero_only
def _print_config(
    config: omegaconf.DictConfig, resolve: bool = True, save_cfg: bool = True
) -> None:
    """Prints content of DictConfig using Rich library and its tree structure.

    Args:
      config (DictConfig): Configuration composed by Hydra.
      resolve (bool): Whether to resolve reference fields of DictConfig.
      save_cfg (bool): Whether to save the configuration tree to a file.
    """

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


@L.pytorch.utilities.rank_zero_only
def _print_batch(train_ds, valid_ds, tokenizer, k=64, vocab_offset=False):
    for dl_type, dl in [("train", train_ds), ("valid", valid_ds)]:
        print(f"Printing {dl_type} dataloader batch.")
        batch = next(iter(dl))
        print("Batch input_ids.shape", batch["input_ids"].shape)
        first = batch["input_ids"][0, :k]
        last = batch["input_ids"][0, -k:]
        print(f"First {k} tokens:", tokenizer.decode(first))
        print("ids:", first)
        print(f"Last {k} tokens:", tokenizer.decode(last))
        print("ids:", last)


def _generate_samples(model, config, logger, tokenizer):
    logger.info("Starting Sample Eval.")
    if config.eval.disable_ema:
        logger.info("Disabling EMA.")
        model.ema = None
    all_samples = [] 
    device_count = torch.cuda.device_count()
    if device_count > 1:
        print(f"Using {device_count} GPUs for parallel generation")
        model = torch.nn.DataParallel(model)

    else: 
        print("Using single GPU for generation")


    # using compile 
    if hasattr(model, 'module'):
        # If using DataParallel, the actual model is in .module
        model.module.backbone = torch.compile(model.module.backbone)
    else:
        # Single GPU case
        model.backbone = torch.compile(model.backbone)

    for _ in range(config.sampling.num_sample_batches):
        samples = model.restore_model_and_sample(num_steps=config.sampling.steps)

        model.metrics.record_entropy(samples)
        text_samples = model.tokenizer.batch_decode(samples)
        model.metrics.record_generative_perplexity(
            text_samples, config.model.length, model.device
        )
        all_samples.extend(list(text_samples))

        generative_ppl = 0.0
        entropy = 0.0
        if hasattr(model, 'module'):
            model.module.metrics.record_entropy(samples)
        else:
            model.metrics.record_entropy(samples)
    
        gen_sequences = tokenizer.batch_decode(samples)
        
        if hasattr(model, 'module'):
            model.module.metrics.record_generative_perplexity(gen_sequences, config.model.length, device)
        else:
            device = torch.device('cuda:0')
            model.metrics.record_generative_perplexity(gen_sequences, config.model.length, device)
            
        gen_samples = tokenizer.batch_decode(gen_sequences)
        all_samples.extend(list(gen_samples))

    if hasattr(model, 'module'):
        generative_ppl = model.module.metrics.gen_ppl.compute().item()
        entropy = model.module.metrics.sample_entropy.compute().item()
    else:
        generative_ppl = model.metrics.gen_ppl.compute().item()
        entropy = model.metrics.sample_entropy.compute().item()

    res = {
                "generative_ppl": generative_ppl,
                "entropy": entropy,
                "generated_seqs": all_samples,
    }
    return res

def _sweep(diffusion_model, config, tokenizer, logger, is_conditional=False):

    temps_to_use = torch.linspace(.7, 1.0, steps=15).tolist()
    steps_to_use = [8, 16, 32, 64, 128, 256, 512]

    _, valid_ds = dataloader.get_dataloaders(config, tokenizer)
    model = _load_from_checkpoint(
        diffusion_model=diffusion_model, config=config, tokenizer=tokenizer
    )
    for steps in steps_to_use:
        config.sampling.steps = steps

        for temp in temps_to_use:
            config.sampling.temperature = temp
            model.metrics.reset()
            res = _conditional_gen_loop(model, config, logger, tokenizer, valid_ds, is_conditional=is_conditional)
            res['steps'] = steps
            res['temperature'] = temp
            samples_path = config.eval.generated_samples_path
            samples_path = samples_path.replace('.json', f'_steps{steps}_temp{temp:.3f}.json')
            with fsspec.open(samples_path, "w") as f:
                json.dump(res, f,
                    indent=4,
                )
            print("Samples saved at:", samples_path) 
    return




def _conditional_gen_loop(model, config, logger, tokenizer, valid_ds):

    logger.info("Starting Conditional Eval.")
    # load dataset 
    # only use the validation ds 

    # load diffusion model from checkpoint 

    # sample first 1000 examples 
    num_samples = min(config.eval.num_conditional_samples, len(valid_ds))
    model_length = config.model.length
    counter = 0
    device_count = torch.cuda.device_count()
    if device_count > 1:
        print(f"Using {device_count} GPUs for parallel generation")
        model = torch.nn.DataParallel(model)

    else: 
        print("Using single GPU for generation")


    # using compile 
    if hasattr(model, 'module'):
        # If using DataParallel, the actual model is in .module
        model.module.backbone = torch.compile(model.module.backbone)
    else:
        # Single GPU case
        model.backbone = torch.compile(model.backbone)

    batch_size = config.eval.batch_size * device_count
    samples_to_eval = []
    reference_samples = []
    with tqdm(total=num_samples, desc="Generating samples") as pbar:
        while counter < num_samples:
            batch = next(iter(valid_ds))
            clean_sequences = batch['input_ids'].to(model.device)

            eval_batch_size = clean_sequences.shape[0]
            # repeat over batch size 
            clean_sequences_repeated = clean_sequences


            prompt_tokens, prompt_mask = mask_clean_samples(clean_sequences_repeated, config.eval.prompt_type, batch_size, model_length, config.eval.revealed_fraction)
            prompt_tokens = prompt_tokens.to(model.device)
            prompt_mask = prompt_mask.to(model.device)
            # generate samples from diffusion model 
            samples = model.generate_samples(
                num_samples=batch_size, 
                num_steps=config.sampling.steps, 
                prompt_tokens=prompt_tokens, 
                prompt_mask=prompt_mask
            )

            if hasattr(model, 'module'):
                model.module.metrics.record_entropy(samples)
            else:
                model.metrics.record_entropy(samples)
        
            gen_sequences = tokenizer.batch_decode(samples)
            
            if hasattr(model, 'module'):
                device = torch.device('cuda:0')  
                model.module.metrics.record_generative_perplexity(gen_sequences, config.model.length, device)
            else:
                model.metrics.record_generative_perplexity(gen_sequences, config.model.length, model.device)
                
            clean_samples = tokenizer.batch_decode(clean_sequences)
            clean_samples = [s.replace("<|endoftext|>", "") for s in clean_samples]
            
            # Your original counting logic is fine
            counter += eval_batch_size
            samples_to_eval.extend(list(gen_sequences))
            reference_samples.extend(list(clean_samples))
            pbar.update(eval_batch_size)

    if hasattr(model, 'module'):
        generative_ppl = model.module.metrics.gen_ppl.compute().item()
        entropy = model.module.metrics.sample_entropy.compute().item()
    else:
        generative_ppl = model.metrics.gen_ppl.compute().item()
        entropy = model.metrics.sample_entropy.compute().item()

    metrics = {
        'generative_ppl': generative_ppl,
        'entropy': entropy,
        'gen_samples': samples_to_eval,
        'ref_samples': reference_samples
    }
    print("Generative perplexity:", generative_ppl)
    print("Sample entropy:", entropy)
    return metrics



# computes prompt tokens and prompt mask 
def mask_clean_samples(clean_sequence, prompt_type, batch_size, model_length, revealed_fraction=.1): 
    prompt_tokens = torch.zeros((batch_size, model_length), dtype=torch.long)
    prompt_mask = torch.zeros((batch_size, model_length), dtype=torch.float)

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
    """Main entry point for training."""
    L.seed_everything(config.seed)
    # _print_config(config, resolve=True, save_cfg=True)

    logger = utils.get_logger(__name__)
    tokenizer = dataloader.get_tokenizer(config)
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
    kwargs = {
        "diffusion_model": diffusion_model,
        "config": config,
        "tokenizer": tokenizer,
        "logger": logger,
    }
    if config.mode == "conditional_sweep":
        _sweep(**kwargs, is_conditional=True)

    else: 
        _sweep(**kwargs, is_conditional=False)

if __name__ == "__main__":
    main()

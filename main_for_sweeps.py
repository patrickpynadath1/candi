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
import algos
import dataloader as dataloader
from text_metrics import evaluate_text_metrics
import utils
import pdb
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


def _generate_samples(model, config, logger, tokenizer, compute_eval_metrics):
    logger.info("Starting Sample Eval.")
    if config.eval.disable_ema:
        logger.info("Disabling EMA.")
        model.ema = None
    all_samples = [] 

    print("Using single GPU for generation")

    # using compile 

    for _ in tqdm(range(config.sampling.num_sample_batches)):
        samples = model.restore_model_and_sample(num_steps=config.sampling.steps)

        text_samples = model.tokenizer.batch_decode(samples)
        all_samples.extend(list(text_samples))

        if compute_eval_metrics:
            model.metrics.record_entropy(samples)
            model.metrics.record_generative_perplexity(
                text_samples, config.model.length, model.device
            )

            generative_ppl = 0.0
            entropy = 0.0
            model.metrics.record_entropy(samples)
    
        
            device = torch.device('cuda:0')
            model.metrics.record_generative_perplexity(text_samples, config.model.length, device)

        all_samples.extend(list(text_samples))

    if compute_eval_metrics:
        generative_ppl = model.metrics.gen_ppl.compute().item()
        entropy = model.metrics.sample_entropy.compute().item()

        res = {
                    "generative_ppl": generative_ppl,
                    "entropy": entropy,
                    "generated_seqs": all_samples,
        }
    else: 
        res = {
            "generated_seqs": all_samples,
        }
    return res, samples.to("cpu")

def _sweep(diffusion_model, config, tokenizer, logger, is_conditional=False):

    temps_to_use = torch.linspace(.1, 2.0, steps=30).tolist()
    steps_to_use = [8, 16, 32, 64, 128]

    model = _load_from_checkpoint(
        diffusion_model=diffusion_model, config=config, tokenizer=tokenizer
    )
    # model.backbone = torch.compile(model.backbone)
    total_metadata = {}

    compute_eval_metrics = config.eval.get("compute_eval_metrics", True)
    for steps in steps_to_use:
        cur_step_info = {}
        config.sampling.steps = steps

        for temp in temps_to_use:
            model.temp = temp
            model.metrics.reset()
            res, samples = _generate_samples(model, config, logger, tokenizer, compute_eval_metrics=compute_eval_metrics)

            res['steps'] = steps
            res['temperature'] = temp
            samples_path = config.eval.generated_samples_path
            samples_path = samples_path.replace('.json', f'_steps{steps}_temp{temp:.3f}.json')
            with fsspec.open(samples_path, "w") as f:
                json.dump(res, f,
                    indent=4,
                )
            raw_tokens_path = samples_path.replace('.json', f'_steps{steps}_temp{temp:.3f}.pt')
            torch.save(samples, raw_tokens_path)
            print("Samples saved at:", samples_path)
            print("Raw token ids saved at:", raw_tokens_path)
            if compute_eval_metrics:
                if config.data.train == 'text8': 
                    train_ds, valid_ds = dataloader.get_dataloaders(config, tokenizer)
                    train_texts = []
                    # check if train_texts.txt exists
                    if os.path.exists("/home/patrick/duo/train_texts.txt"):
                        print("Loading training texts from train_texts.txt for text metrics...")
                        train_texts = open("/home/patrick/duo/train_texts.txt", "r").readlines()
                    else: 
                        print("Loading training texts for text metrics...")
                        for batch in tqdm(train_ds):
                            batch_texts = tokenizer.batch_decode(batch["input_ids"], skip_special_tokens=True)
                            train_texts.extend(batch_texts)

                        with open("/home/patrick/duo/train_texts.txt", "w") as f:
                            for text in train_texts:
                                f.write(f"{text}\n")
                    text_metrics = evaluate_text_metrics(res["generated_seqs"], train_texts)
                    cur_step_info[temp] = text_metrics
                    print(f"Steps: {steps}, Temp: {temp}")
                    print(text_metrics)
                else: 
                    cur_step_info[temp] = {
                        "perplexity": res["generative_ppl"],
                        "entropy": res["entropy"],
                    }
                    print(f"Steps: {steps}, Temp: {temp}")
                    print("perplexity: ", res["generative_ppl"])
                    print("entropy: ", res["entropy"])
            total_metadata[steps] = cur_step_info
    
    if compute_eval_metrics:
        metadata_path = config.eval.generated_samples_path.replace('.json', f'_sweep_metadata.json')
        with fsspec.open(metadata_path, "w") as f:
            json.dump(total_metadata, f,
                indent=4,
            )
    return


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(config):
    """Main entry point for training."""
    L.seed_everything(config.seed)
    # _print_config(config, resolve=True, save_cfg=True)

    logger = utils.get_logger(__name__)
    tokenizer = dataloader.get_tokenizer(config)

    if config.algo.name in algos.all_algos:
        diffusion_model = algos.all_algos[config.algo.name]
    else:
        raise ValueError(f"Invalid algorithm name: {config.algo.name}")
    kwargs = {
        "diffusion_model": diffusion_model,
        "config": config,
        "tokenizer": tokenizer,
        "logger": logger,
    }
    _sweep(**kwargs, is_conditional=False)

if __name__ == "__main__":
    main()
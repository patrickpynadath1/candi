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
from lm_eval.models.huggingface import HFLM
from lm_eval import evaluator
from hugging_face_wrappers import DiffusionEvalWrapper, CandiEvalWrapper


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
        print(f"First {k} tokens:", tokenizer.decode(first + tokenizer.eos_token_id))
        print("ids:", first)
        print(f"Last {k} tokens:", tokenizer.decode(last + tokenizer.eos_token_id))
        print("ids:", last)


def _generate_samples(diffusion_model, config, logger, tokenizer):
    logger.info("Starting Sample Eval.")

    if hasattr(config.data, "vocab_file"):
        vocab_offset = True 
        new_vocab = json.load(open(f'{config.data.cache_dir}/{config.data.vocab_file}', 'r'))
        tokenizer.add_tokens(new_vocab)
    model = _load_from_checkpoint(
        diffusion_model=diffusion_model, config=config, tokenizer=tokenizer
    )
    # model.metrics.gen_ppl.reset()
    # model.metrics.sample_entropy.reset()
    if config.eval.disable_ema:
        logger.info("Disabling EMA.")
        model.ema = None
    stride_length = config.sampling.stride_length
    num_strides = config.sampling.num_strides
    all_samples = [] 
    for _ in range(config.sampling.num_sample_batches):
        if config.sampling.semi_ar:
            _, intermediate_samples, _ = model.restore_model_and_semi_ar_sample(
                stride_length=stride_length,
                num_strides=num_strides,
                dt=1 / config.sampling.steps,
            )
            text_samples = intermediate_samples[-1]
            # Note: Samples generated using semi-ar method
            # need to to be processed before computing generative perplexity
            # since these samples contain numerous <|endoftext|> tokens
            # and diffusion.compute_generative_perplexity() discards
            # any text after the first EOS token.
        else:
            samples = model.restore_model_and_sample(num_steps=config.sampling.steps)

            model.metrics.record_entropy(samples)
            text_samples = model.tokenizer.batch_decode(samples)
            model.metrics.record_generative_perplexity(
                text_samples, config.model.length, model.device
            )
            all_samples.extend(list(text_samples))

    # pdb.set_trace()
    # with open('samples.txt', 'w') as f:
    #     for sample in text_samples:
    #         f.write(f"{sample}\n")

    # # Evaluate text metrics
    # train_texts = open("/home/patrick/duo/train_texts.txt", "r").readlines()
    # text_metrics = evaluate_text_metrics(all_samples, train_texts)
    # print(text_metrics)

    generative_ppl = 0.0
    entropy = 0.0
    if not config.sampling.semi_ar:
        generative_ppl = model.metrics.gen_ppl.compute().item()
        entropy = model.metrics.sample_entropy.compute().item()
        print("Generative perplexity:", generative_ppl)
        print("Sample entropy:", entropy)
    samples_path = config.eval.generated_samples_path
    with fsspec.open(samples_path, "w") as f:
        json.dump(
            {
                "generative_ppl": generative_ppl,
                "entropy": entropy,
                "generated_seqs": all_samples,
            },
            f,
            indent=4,
        )
    print("Samples saved at:", samples_path)


def _generate_conditional_samples(diffusion_model, config, logger, tokenizer):

    logger.info("Starting Conditional Eval.")
    # load dataset 
    # only use the validation ds 
    _, valid_ds = dataloader.get_dataloaders(config, tokenizer)

    # load diffusion model from checkpoint 
    model = _load_from_checkpoint(
        diffusion_model=diffusion_model, config=config, tokenizer=tokenizer
    )

    # sample first 1000 examples 
    num_samples = min(config.eval.num_conditional_samples, len(valid_ds))
    batch_size =64 
    model_length = config.model.length
    counter = 0
    model.backbone = torch.compile(model.backbone)

    samples_to_eval = []
    reference_samples = []
    with tqdm(total=num_samples, desc="Generating samples") as pbar:
        while counter < num_samples:
            batch = next(iter(valid_ds))
            clean_sequences = batch['input_ids'].to(model.device)

            eval_batch_size = clean_sequences.shape[0]
            # repeat over batch size 
            # dataloader will be set to batch size of 1 
            clean_sequences_repeated = clean_sequences
            # clean_sequences_repeated = clean_sequences.repeat_interleave(batch_size // eval_batch_size, dim=0)
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
            model.metrics.record_entropy(samples)
        
            gen_sequences = tokenizer.batch_decode(samples)
            model.metrics.record_generative_perplexity(gen_sequences, config.model.length, model.device)

            clean_samples = tokenizer.batch_decode(clean_sequences)

            counter += eval_batch_size

            samples_to_eval.extend(list(gen_sequences))
            reference_samples.extend(list(clean_samples))
            pbar.update(eval_batch_size)


        # either use prompting style input or masked style input

    # given generations, compute standard metrics -- perplexity, generative entropy, bleu, rouge, mauve, etc 
    generative_ppl = model.metrics.gen_ppl.compute().item()
    entropy = model.metrics.sample_entropy.compute().item()
    device = model.device
    # move model from gpu to cpu
    del model

    bertscore = evaluate.load("bertscore")
    bertscore_res = bertscore.compute(predictions=samples_to_eval, references=reference_samples, lang="en", model_type='roberta-large', batch_size=64, device=device)
    mauve_res = mauve.compute_mauve(p_text=samples_to_eval, q_text=reference_samples, device_id=0, num_buckets=50) 

    metrics = {
        'generative_ppl': generative_ppl,
        'entropy': entropy,
        'bertscore_precision': np.mean(bertscore_res['precision']),
        'bertscore_recall': np.mean(bertscore_res['recall']),
        'bertscore_f1': np.mean(bertscore_res['f1']),
        'mauve': mauve_res.mauve, 
        'gen_samples': samples_to_eval,
        'ref_samples': reference_samples
    }
    samples_path = config.eval.generated_samples_path
    with fsspec.open(samples_path, "w") as f:
        json.dump(metrics, f,
            indent=4,
        )
    print("Samples saved at:", samples_path)



    for k, v in metrics.items(): 
        if k not in ['gen_samples', 'ref_samples']:
            print(f"{k}: {v}")
    return 


def _automated_benchmarks(diffusion_model, config, logger, tokenizer):

    logger.info("Starting automated benchmarks.")

    # load diffusion model from checkpoint 
    model = _load_from_checkpoint(
        diffusion_model=diffusion_model, config=config, tokenizer=tokenizer
    )

    # first wrap it in a transformer class 

    if config.algo.name == "candi":
        lm = CandiEvalWrapper(model, tokenizer)
    else: 
        lm = DiffusionEvalWrapper(model, tokenizer)


    results = evaluator.simple_evaluate(
        model=lm, 
        tasks=['hellaswag'], 
        batch_size=32)
    
    print(results['results'])




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



def _eval_ppl(diffusion_model, config, logger, tokenizer):
    logger.info("Starting Perplexity Eval.")

    model = _load_from_checkpoint(
        diffusion_model=diffusion_model, config=config, tokenizer=tokenizer
    )
    if config.eval.disable_ema:
        logger.info("Disabling EMA.")
        model.ema = None

    wandb_logger = None
    if config.get("wandb", None) is not None:
        wandb_logger = L.pytorch.loggers.WandbLogger(
            config=omegaconf.OmegaConf.to_object(config), **config.wandb
        )
    callbacks = []
    if "callbacks" in config:
        for _, callback in config.callbacks.items():
            callbacks.append(hydra.utils.instantiate(callback))
    trainer = hydra.utils.instantiate(
        config.trainer,
        default_root_dir=os.getcwd(),
        callbacks=callbacks,
        strategy=hydra.utils.instantiate(config.strategy),
        logger=wandb_logger,
    )
    _, valid_ds = dataloader.get_dataloaders(
        config, tokenizer, skip_train=True, valid_seed=config.seed
    )
    trainer.validate(model, valid_ds)

def calc_mauve(generated_texts: List[str], reference_texts: List[str]) -> float:
    out = mauve.compute_mauve(p_text=generated_texts, q_text=reference_texts, device_id=0)
    return out.mauve


def _train(diffusion_model, config, logger, tokenizer):
    logger.info("Starting Training.")
    wandb_logger = None
    if config.get("wandb", None) is not None:
        wandb_logger = L.pytorch.loggers.WandbLogger(
            config=omegaconf.OmegaConf.to_object(config), **config.wandb
        )

    if (
        config.checkpointing.resume_from_ckpt
        and config.checkpointing.resume_ckpt_path is not None
        and utils.fsspec_exists(config.checkpointing.resume_ckpt_path)
    ):
        ckpt_path = config.checkpointing.resume_ckpt_path
    else:
        ckpt_path = None

    # Lightning callbacks
    callbacks = []
    if "callbacks" in config:
        for _, callback in config.callbacks.items():
            callbacks.append(hydra.utils.instantiate(callback))

    train_ds, valid_ds = dataloader.get_dataloaders(config, tokenizer)

    if hasattr(config.data, "vocab_file"): 
        vocab_offset = True 
        new_vocab = json.load(open(f'{config.data.cache_dir}/{config.data.vocab_file}', 'r'))
        tokenizer.add_tokens(new_vocab)
    else: 
        vocab_offset = False
    _print_batch(train_ds, valid_ds, tokenizer, vocab_offset=vocab_offset)

    if config.training.finetune_path != "":
        assert utils.fsspec_exists(config.training.finetune_path)
        model = diffusion_model.load_from_checkpoint(
            config.training.finetune_path, tokenizer=tokenizer, config=config
        )
    else:
        model = diffusion_model(config, tokenizer=tokenizer)

    trainer = hydra.utils.instantiate(
        config.trainer,
        default_root_dir=os.getcwd(),
        callbacks=callbacks,
        strategy=hydra.utils.instantiate(config.strategy),
        logger=wandb_logger,
    )
    trainer.fit(model, train_ds, valid_ds, ckpt_path=ckpt_path)


def _abductive_reasoning(diffusion_model, config, logger, tokenizer):
    logger.info("Starting Abductive Reasoning Eval.")

    model = _load_from_checkpoint(
        diffusion_model=diffusion_model, config=config, tokenizer=tokenizer
    )

    data_file = config.eval.abductive_data_file

    with open(f"{config.scratch_dir}/{data_file}", "r") as f:
        data = json.load(f)

    # do it in batches 

    # problem: how do deal with the fixed length issue? 

    



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
    if config.mode == "sample_eval":
        _generate_samples(**kwargs)
    elif config.mode == "ppl_eval":
        _eval_ppl(**kwargs)

    elif config.mode == "conditional_eval":
        _generate_conditional_samples(**kwargs)

    elif config.mode == "automated_benchmarks":
        _automated_benchmarks(**kwargs)
    else:
        _train(**kwargs)


if __name__ == "__main__":
    main()

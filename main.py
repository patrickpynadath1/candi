from tqdm import tqdm

from collections import Counter
import json
import math
import os
import time

from typing import List, Dict, Tuple
import fsspec
import hydra
import lightning as L
import omegaconf
import rich.syntax
import rich.tree
import torch
import dataloader as dataloader
import utils
import algos
from algos import algo_registry
from text_metrics import evaluate_text_metrics
from tasks.evaluation import evaluate_gsm8k, evaluate_sudoku


omegaconf.OmegaConf.register_new_resolver("cwd", os.getcwd)
omegaconf.OmegaConf.register_new_resolver("device_count", torch.cuda.device_count)
omegaconf.OmegaConf.register_new_resolver("eval", eval)
omegaconf.OmegaConf.register_new_resolver("div_up", lambda x, y: (x + y - 1) // y)

def _load_from_checkpoint(diffusion_model, config, tokenizer):
    if "hf" in config.algo.backbone:
        return diffusion_model(config, tokenizer=tokenizer).to("cuda")

    # Evaluation needs model buffers and (normally) EMA weights, not optimizer,
    # callback, or trainer-loop state. Lightning's load_from_checkpoint eagerly
    # restores the complete training archive and is pathologically slow for our
    # ~450 MB checkpoints. mmap keeps unused optimizer tensors lazy.
    started = time.monotonic()
    checkpoint = torch.load(
        config.eval.checkpoint_path,
        map_location="cpu",
        weights_only=False,
        mmap=True,
    )
    model = diffusion_model(config, tokenizer=tokenizer)
    state_dict = {
        key.replace("_orig_mod.", ""): value
        for key, value in checkpoint["state_dict"].items()
        if "teacher" not in key
    }
    model.load_state_dict(state_dict, strict=True)

    if not config.eval.disable_ema and model.ema is not None:
        shadows = checkpoint.get("ema", {}).get("shadow_params")
        parameters = list(model._get_parameters())
        if shadows is None:
            raise KeyError("checkpoint does not contain EMA shadow parameters")
        if len(shadows) != len(parameters):
            raise ValueError(
                "EMA parameter count mismatch: "
                f"checkpoint has {len(shadows)}, model has {len(parameters)}"
            )
        with torch.no_grad():
            for parameter, shadow in zip(parameters, shadows):
                parameter.copy_(shadow)
    # EMA is already materialized in the model. Disabling the helper prevents
    # _eval_mode() from cloning every parameter and copying it a second time.
    model.ema = None
    print(
        f"Lightweight checkpoint load completed in "
        f"{time.monotonic() - started:.2f}s",
        flush=True,
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

    # Evaluate text metrics
    # train_texts = open("/home/patrick/duo/train_texts.txt", "r").readlines()
    # text_metrics = evaluate_text_metrics(all_samples, train_texts)
    # print(text_metrics)
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
        text_metrics = evaluate_text_metrics(all_samples, train_texts)
        print(text_metrics)
        samples_path = config.eval.generated_samples_path
        with fsspec.open(samples_path, "w") as f:
            json.dump(
                {
                    "algorithm": config.algo.name,
                    "steps": config.sampling.steps,
                    "temperature": config.algo.get("temp", 1.0),
                    "num_samples": len(all_samples),
                    "metrics": text_metrics,
                    "generated_seqs": all_samples,
                },
                f,
                indent=2,
            )
        print("Samples and text8 metrics saved at:", samples_path)
    else: 
        generative_ppl = 0.0
        generative_bpc = 0.0
        entropy = 0.0
        if not config.sampling.semi_ar:
            generative_ppl = model.metrics.gen_ppl.compute().item()
            generative_bpc = model.metrics.gen_bpc.compute().item()
            entropy = model.metrics.sample_entropy.compute().item()
            print("Generative perplexity:", generative_ppl)
            print("Generative BPC:", generative_bpc)
            print("Sample entropy:", entropy)
        samples_path = config.eval.generated_samples_path
        with fsspec.open(samples_path, "w") as f:
            json.dump(
                {
                    "generative_ppl": generative_ppl,
                    "generative_bpc": generative_bpc,
                    "entropy": entropy,
                    "generated_seqs": all_samples,
                },
                f,
                indent=4,
            )
        print("Samples saved at:", samples_path)


def _decode_batch_texts(input_ids, attention_mask, tokenizer):
    texts = []
    input_ids = input_ids.detach().cpu()
    attention_mask = attention_mask.detach().cpu()
    for sample_ids, sample_mask in zip(input_ids, attention_mask):
        valid_ids = sample_ids[sample_mask.bool()].tolist()
        text = tokenizer.decode(
            valid_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        texts.append(text)
    return texts


@torch.no_grad()
def _compute_eval_metrics(model, valid_ds, tokenizer):
    model.metrics.reset()
    model._eval_mode()

    total_nll = torch.zeros((), dtype=torch.float64, device=model.device)
    total_tokens = torch.zeros((), dtype=torch.float64, device=model.device)
    total_utf8_chars = torch.zeros((), dtype=torch.float64, device=model.device)

    for batch in tqdm(valid_ds, desc="Evaluating"):
        input_ids = batch["input_ids"].to(model.device)
        attention_mask = batch["attention_mask"].to(model.device)

        input_tokens, output_tokens, valid_tokens = model._process_model_input(
            input_ids, attention_mask
        )
        token_nlls = model.nll(
            input_tokens,
            output_tokens,
            None,
            False,
        )
        if token_nlls.ndim != 2:
            raise ValueError(
                f"Expected per-token NLLs with shape [batch, seq], got {token_nlls.shape}."
            )

        sample_nlls = (token_nlls * valid_tokens).sum(dim=-1).to(torch.float64)
        sample_token_counts = valid_tokens.sum(dim=-1).to(torch.float64)
        sample_texts = _decode_batch_texts(input_ids, attention_mask, tokenizer)
        sample_utf8_chars = torch.tensor(
            [len(text.encode("utf-8")) for text in sample_texts],
            dtype=torch.float64,
            device=model.device,
        )

        total_nll += sample_nlls.sum()
        total_tokens += sample_token_counts.sum()
        total_utf8_chars += sample_utf8_chars.sum()

    model._train_mode()

    eval_nll = total_nll / total_tokens
    eval_ppl = torch.exp(eval_nll)
    eval_bpc = total_nll / (total_utf8_chars * math.log(2))

    return {
        "eval_nll": eval_nll.item(),
        "eval_ppl": eval_ppl.item(),
        "eval_bpc": eval_bpc.item(),
        "total_nll": total_nll.item(),
        "total_tokens": total_tokens.item(),
        "total_utf8_chars": total_utf8_chars.item(),
    }


def _eval_ppl(diffusion_model, config, logger, tokenizer):
    logger.info("Starting Perplexity Eval.")

    model = _load_from_checkpoint(
        diffusion_model=diffusion_model, config=config, tokenizer=tokenizer
    )
    if config.eval.disable_ema:
        logger.info("Disabling EMA.")
        model.ema = None

    _, valid_ds = dataloader.get_dataloaders(
        config, tokenizer, skip_train=True, valid_seed=config.seed
    )
    metrics = _compute_eval_metrics(model, valid_ds, tokenizer)

    print("Eval NLL:", metrics["eval_nll"])
    print("Eval PPL:", metrics["eval_ppl"])
    print("Eval BPC:", metrics["eval_bpc"])
    print("Total NLL:", metrics["total_nll"])
    print("Total tokens:", metrics["total_tokens"])
    print("Total UTF-8 chars:", metrics["total_utf8_chars"])


def _token_entropy(sample: torch.Tensor) -> float:
    _, counts = torch.unique(sample, return_counts=True, sorted=False)
    probs = counts.to(torch.float64) / counts.sum()
    return torch.special.entr(probs).sum().item()


def _sample_token_counts(sample: torch.Tensor) -> Counter:
    token_ids, counts = torch.unique(sample, return_counts=True, sorted=False)
    return Counter(
        {
            int(token_id): int(count)
            for token_id, count in zip(token_ids.tolist(), counts.tolist())
        }
    )


def _iter_entropy_samples(batch, use_attention_mask=True):
    input_ids = batch["input_ids"]
    attention_mask = batch.get("attention_mask")

    for idx, sample in enumerate(input_ids):
        if use_attention_mask and attention_mask is not None:
            sample = sample[attention_mask[idx].bool()]
        yield sample


def _compute_dataset_entropy(loader, max_batches=None, max_sequences=None, use_attention_mask=True):
    mean_entropy = 0.0
    m2 = 0.0
    num_sequences = 0
    token_counts = Counter()
    total_tokens = 0

    for batch_idx, batch in enumerate(tqdm(loader, desc="Computing entropy")):
        if max_batches is not None and batch_idx >= max_batches:
            break

        for sample in _iter_entropy_samples(
            batch, use_attention_mask=use_attention_mask
        ):
            entropy = _token_entropy(sample)
            sample_counts = _sample_token_counts(sample)
            num_sequences += 1
            delta = entropy - mean_entropy
            mean_entropy += delta / num_sequences
            delta2 = entropy - mean_entropy
            m2 += delta * delta2
            token_counts.update(sample_counts)
            total_tokens += int(sample.numel())
            if max_sequences is not None and num_sequences >= max_sequences:
                break

        if max_sequences is not None and num_sequences >= max_sequences:
            break

    if num_sequences == 0:
        raise ValueError("No samples were loaded from the selected split.")

    if num_sequences > 1:
        std_entropy = math.sqrt(m2 / (num_sequences - 1))
    else:
        std_entropy = 0.0

    empirical_frequencies = {
        str(token_id): count / total_tokens for token_id, count in token_counts.items()
    }

    return {
        "entropy_mean": mean_entropy,
        "entropy_std": std_entropy,
        "num_sequences": num_sequences,
        "total_tokens": total_tokens,
        "token_counts": dict(token_counts),
        "empirical_frequencies": empirical_frequencies,
    }


def _dataset_entropy(config, logger, tokenizer):
    logger.info("Starting Dataset Entropy Eval.")
    if hasattr(config.data, "vocab_file"):
        new_vocab = json.load(open(f"{config.data.cache_dir}/{config.data.vocab_file}", "r"))
        tokenizer.add_tokens(new_vocab)

    _, valid_ds = dataloader.get_dataloaders(
        config,
        tokenizer,
        skip_train=True,
        valid_seed=config.seed,
    )

    print("Dataset entropy config:")
    print("  data.valid:", config.data.valid)
    print("  model.length:", config.model.length)
    entropy_stats = _compute_dataset_entropy(
        valid_ds,
        use_attention_mask=True,
    )
    print("valid.num_sequences:", entropy_stats["num_sequences"])
    print("valid.total_tokens:", entropy_stats["total_tokens"])
    print("valid.entropy_mean:", entropy_stats["entropy_mean"])
    print("valid.entropy_std:", entropy_stats["entropy_std"])

    output_path = os.path.join(
        config.checkpointing.save_dir, "dataset_entropy_stats.json"
    )
    with fsspec.open(output_path, "w") as f:
        json.dump(entropy_stats, f, indent=4, sort_keys=True)
    print("Dataset entropy stats saved at:", output_path)


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

@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(config):
    """Main entry point for training."""
    L.seed_everything(config.seed)
    # _print_config(config, resolve=True, save_cfg=True)

    logger = utils.get_logger(__name__)
    tokenizer = dataloader.get_tokenizer(config)

    algo_cls = algo_registry.get(config.algo.name)
    if algo_cls is None:
        raise ValueError(f"Invalid algorithm name: {config.algo.name}")
    kwargs = {
        "diffusion_model": algo_cls,
        "config": config,
        "tokenizer": tokenizer,
        "logger": logger,
    }
    if config.mode == "sample_eval":
        _generate_samples(**kwargs)
    elif config.mode == "sudoku_eval":
        evaluate_sudoku(load_checkpoint=_load_from_checkpoint, **kwargs)
    elif config.mode == "gsm8k_eval":
        evaluate_gsm8k(load_checkpoint=_load_from_checkpoint, **kwargs)
    elif config.mode == "ppl_eval":
        _eval_ppl(**kwargs)
    elif config.mode == "dataset_entropy":
        _dataset_entropy(config=config, logger=logger, tokenizer=tokenizer)
    else:
        _train(**kwargs)


if __name__ == "__main__":
    main()

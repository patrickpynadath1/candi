"""Shared generation evaluation; task semantics are delegated to Task."""

from contextlib import nullcontext
from dataclasses import asdict
import time
import torch

from hydra.utils import instantiate


@torch.no_grad()
def evaluate(
    module,
    task,
    sampler_config,
    *,
    batch_size=64,
    max_examples=None,
    use_ema=True,
    precision="bf16-mixed",
    indices=None,
    evaluation_name=None,
):
    if batch_size <= 0 or (max_examples is not None and max_examples <= 0):
        raise ValueError("evaluation batch and example counts must be positive")
    evaluation_name = evaluation_name or task.primary_metric.dataset
    dataset = task.evaluation_dataset(evaluation_name)
    metrics = task.create_metric_accumulator(evaluation_name)
    indices = list(range(len(dataset))) if indices is None else list(indices)
    if max_examples is not None:
        indices = indices[:max_examples]
    if not indices:
        raise ValueError("cannot evaluate an empty split")
    sampler = instantiate(sampler_config)
    if hasattr(sampler, "configure_special_tokens"):
        sampler.configure_special_tokens(
            bos_token_id=getattr(
                getattr(task, "tokenizer", None), "bos_token_id", None
            ),
            eos_token_id=getattr(
                task,
                "generation_eos_token_id",
                getattr(getattr(task, "tokenizer", None), "eos_token_id", None),
            ),
            pad_token_id=getattr(
                getattr(task, "tokenizer", None), "pad_token_id", None
            ),
        )
    variable_length = getattr(module.algorithm, "supports_variable_length", False)
    if variable_length and not getattr(metrics, "supports_variable_length", False):
        raise TypeError("task metrics do not support variable-length generation")
    nfes = []
    sequence_lengths, truncations = [], []
    block_lengths = []
    started = time.monotonic()
    amp = (
        torch.autocast("cuda", dtype=torch.bfloat16)
        if module.device.type == "cuda" and precision == "bf16-mixed"
        else nullcontext()
    )
    with module.evaluation_weights(use_ema), amp:
        for ids in task.evaluation_batches(dataset, indices, batch_size):
            examples = [dataset[index] for index in ids]
            prompt, mask = task.prepare_generation(examples)
            samples, metadata = module.algorithm.generate(
                sampler, prompt.to(module.device), mask.to(module.device)
            )
            if variable_length:
                lengths = metadata.get("lengths")
                if (
                    lengths is None
                    or len(lengths) != len(samples)
                    or any(
                        not isinstance(n, int) or not 0 <= n <= samples.shape[1]
                        for n in lengths
                    )
                ):
                    raise ValueError(
                        "variable-length sampler must return valid per-sample lengths"
                    )
                blocks = metadata.get("block_lengths")
                if blocks is not None:
                    if len(blocks) != len(lengths) or any(
                        any(not isinstance(n, int) or n < 0 for n in row)
                        or sum(row) != length
                        for row, length in zip(blocks, lengths)
                    ):
                        raise ValueError("block lengths must sum to generated lengths")
                    block_lengths.extend(blocks)
                metrics.update(samples.cpu(), examples, ids, lengths=lengths)
                sequence_lengths.extend(lengths)
                truncations.extend(metadata.get("truncated", [False] * len(lengths)))
            else:
                metrics.update(samples.cpu(), examples, ids)
            nfes.extend(metadata["nfe_per_sample"])
    # Serialize typed task metrics at the reporting boundary. Execution
    # statistics are evaluator metadata, not puzzle correctness fields.
    result = asdict(metrics.compute())
    records = result.get("records", [])
    for record, nfe in zip(records, nfes):
        record["nfe"] = nfe
    if variable_length:
        result.update(
            generated_lengths=sequence_lengths, insertion_truncated=truncations
        )
    if block_lengths:
        if len(block_lengths) != len(sequence_lengths):
            raise ValueError("block lengths must be present for every evaluated sample")
        result["block_lengths"] = block_lengths
    result.update(
        mean_nfe=sum(nfes) / len(nfes),
        min_nfe=min(nfes),
        max_nfe=max(nfes),
        evaluation_seconds=time.monotonic() - started,
    )
    return result

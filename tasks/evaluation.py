"""Conditional generation loops for structured reasoning tasks."""

import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm, trange

from .gsm8k import (
    evaluate_codes,
    get_gsm8k_dataset,
    gold_answer,
    numbers_equal,
    score_response,
)
from .sudoku import get_sudoku_dataset, sudoku_metrics


def _prepare_model(diffusion_model, config, tokenizer, load_checkpoint):
    model = load_checkpoint(diffusion_model, config, tokenizer)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    if config.eval.disable_ema:
        model.ema = None
    model._eval_mode()
    return model


def _write_results(output_dir, results):
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    destination = path / "results.json"
    destination.write_text(json.dumps(results, indent=2))
    return destination


@torch.no_grad()
def evaluate_sudoku(diffusion_model, config, tokenizer, load_checkpoint, logger):
    model = _prepare_model(diffusion_model, config, tokenizer, load_checkpoint)
    dataset = get_sudoku_dataset(config, tokenizer)["validation"]
    max_examples = getattr(config.sudoku, "max_examples", None)
    if max_examples is not None:
        dataset = dataset.select(range(min(int(max_examples), len(dataset))))
    batch_size, prompt_length = config.sudoku.batch_size, tokenizer.prompt_len
    diagnostics = getattr(model, "rank_diagnostics", None)
    if callable(diagnostics):
        path = Path(config.sudoku.output_dir)
        path.mkdir(parents=True, exist_ok=True)
        (path / "rank_diagnostics.json").write_text(json.dumps(diagnostics(), indent=2))
    if model.device.type == "cuda":
        torch.cuda.synchronize(model.device)
    started = time.perf_counter()
    records = []
    sample_nfes = []
    for start in trange(0, len(dataset), batch_size, desc="Sudoku evaluation"):
        batch = torch.stack(
            [
                dataset[index]["input_ids"]
                for index in range(start, min(start + batch_size, len(dataset)))
            ]
        )
        count = batch.shape[0]
        prompt_tokens = torch.zeros(
            count, model.num_tokens, dtype=torch.long, device=model.device
        )
        prompt_mask = torch.zeros_like(prompt_tokens, dtype=torch.bool)
        prompt_tokens[:, :prompt_length] = batch[:, :prompt_length].to(model.device)
        prompt_mask[:, :prompt_length] = True
        sampling_result = model.generate_samples(
            count,
            num_steps=config.sampling.steps,
            prompt_tokens=prompt_tokens,
            prompt_mask=prompt_mask,
            return_result=True,
        )
        generated = sampling_result.samples.cpu()
        if "nfe_per_sample" in sampling_result.metadata:
            sample_nfes.extend(
                float(value)
                for value in sampling_result.metadata["nfe_per_sample"]
            )
        elif "nfe" in sampling_result.metadata:
            sample_nfes.extend(
                [float(sampling_result.metadata["nfe"])] * count
            )
        truth = batch[:, prompt_length:]
        predictions = generated[:, prompt_length : prompt_length + truth.shape[1]]
        for prediction, target, example in zip(predictions, truth, batch):
            metrics = sudoku_metrics(prediction, target, example[1:prompt_length-1])
            records.append(
                {
                    "generated": tokenizer.decode(prediction),
                    "ground_truth": tokenizer.decode(target),
                    **metrics,
                }
            )
    total = max(len(records), 1)
    if model.device.type == "cuda":
        torch.cuda.synchronize(model.device)
    elapsed = time.perf_counter()-started
    results = {
        "accuracy": sum(record["exact"] for record in records) / total,
        "cell_accuracy": sum(record["cell_accuracy"] for record in records) / total,
        "valid_grid_rate": sum(record["valid_grid"] for record in records) / total,
        "valid_row_rate": sum(record["valid_rows"] for record in records) / total,
        "valid_column_rate": sum(record["valid_columns"] for record in records) / total,
        "valid_box_rate": sum(record["valid_boxes"] for record in records) / total,
        "num_correct": sum(record["exact"] for record in records),
        "num_total": len(records),
        "digit_accuracy": sum(record["digit_accuracy"] for record in records) / total,
        "clue_consistency_rate": sum(record["clue_consistency"] for record in records) / total,
        "valid_solution_rate": sum(record["valid_solution"] for record in records) / total,
        "evaluation_seconds": elapsed,
        "seconds_per_puzzle": elapsed / total,
        "mean_nfe": (
            sum(sample_nfes) / len(sample_nfes) if sample_nfes else None
        ),
        "min_nfe": min(sample_nfes) if sample_nfes else None,
        "max_nfe": max(sample_nfes) if sample_nfes else None,
        "records": records,
    }
    destination = _write_results(config.sudoku.output_dir, results)
    logger.info(
        "Sudoku exact accuracy %.2f%%; results saved to %s",
        results["accuracy"] * 100,
        destination,
    )
    return results


def _decode_completion(tokenizer, sequence, prompt_length):
    completion = sequence[prompt_length:].tolist()
    eos = tokenizer.eos_token_id
    if eos in completion:
        completion = completion[: completion.index(eos)]
    return tokenizer.decode(completion, skip_special_tokens=True)


def exact_length_batches(dataset, indices, batch_size):
    """Group prefix prompts by exact length, including singleton batches."""
    buckets = defaultdict(list)
    for index in indices:
        buckets[len(dataset[index]["input_ids"])].append(index)
    for length in sorted(buckets):
        bucket = buckets[length]
        for start in range(0, len(bucket), batch_size):
            yield bucket[start : start + batch_size]


@torch.no_grad()
def generate_gsm8k_records(
    model, dataset, tokenizer, batch_size, num_steps, indices=None
):
    indices = list(range(len(dataset)) if indices is None else indices)
    records = []
    observed_nfes = []
    with tqdm(total=len(indices), desc="GSM8K generation", unit="example") as progress:
        for batch_indices in exact_length_batches(dataset, indices, batch_size):
            items = [dataset[index] for index in batch_indices]
            count = len(items)
            prompt_tokens = torch.zeros(
                count, model.num_tokens, dtype=torch.long, device=model.device
            )
            prompt_mask = torch.zeros_like(prompt_tokens, dtype=torch.bool)
            prompt_length = min(len(items[0]["input_ids"]), model.num_tokens)
            for row, item in enumerate(items):
                ids = item["input_ids"][: model.num_tokens]
                prompt_tokens[row, : len(ids)] = torch.as_tensor(ids, device=model.device)
                prompt_mask[row, : len(ids)] = True
            sampling_result = model.generate_samples(
                count,
                num_steps=num_steps,
                prompt_tokens=prompt_tokens,
                prompt_mask=prompt_mask,
                return_result=True,
            )
            generated = sampling_result.samples.cpu()
            metadata = sampling_result.metadata
            batch_nfes = metadata.get("nfe_per_sample")
            if batch_nfes is None and "nfe" in metadata:
                batch_nfes = [metadata["nfe"]] * count
            if batch_nfes is not None:
                batch_nfes = [float(value) for value in batch_nfes]
            argmax_changes = metadata.get("argmax_changes_per_sample")
            changed_positions = metadata.get(
                "argmax_changed_positions_per_sample"
            )
            changes_per_step = metadata.get("argmax_changes_per_step")
            for row, item in enumerate(items):
                completion = generated[row, prompt_length:].tolist()
                eos = tokenizer.eos_token_id
                hit_eos = eos in completion
                completion_length = completion.index(eos) if hit_eos else len(completion)
                record = {
                    "dataset_index": batch_indices[row],
                    "prompt": item["prompt"],
                    "response_ground_truth": item["response_ground_truth"],
                    "response": _decode_completion(
                        tokenizer, generated[row], prompt_length
                    ),
                    "prompt_tokens": prompt_length,
                    "completion_tokens": completion_length,
                    "hit_eos": hit_eos,
                }
                if batch_nfes is not None:
                    record["nfe"] = batch_nfes[row]
                    observed_nfes.append(batch_nfes[row])
                if argmax_changes is not None:
                    record["argmax_changes"] = int(argmax_changes[row])
                if changed_positions is not None:
                    record["argmax_changed_positions"] = int(
                        changed_positions[row]
                    )
                if changes_per_step is not None:
                    record["argmax_changes_per_step"] = [
                        int(step[row]) for step in changes_per_step
                    ]
                records.append(record)
            progress.update(count)
            if observed_nfes:
                progress.set_postfix(mean_nfe=f"{np.mean(observed_nfes):.1f}")
    return records


def score_gsm8k_records(records, method, timeout, bootstrap_size=1, seed=1):
    diagnostics = []
    executions = (
        evaluate_codes([record["response"] for record in records], timeout)
        if method == "code"
        else None
    )
    for index, record in enumerate(records):
        if executions is not None:
            execution = executions[index]
            diagnostics.append(
                {
                    "correct": numbers_equal(
                        execution.result,
                        gold_answer(record["response_ground_truth"]),
                    ),
                    "parsed": execution.parsed,
                    "executed": execution.executed,
                    "timed_out": execution.timed_out,
                    "sandbox_error": execution.sandbox_error,
                }
            )
        else:
            diagnostics.append(
                {
                    "correct": score_response(
                        record["response"],
                        record["response_ground_truth"],
                        method,
                        timeout,
                    )
                }
            )
    correct = np.asarray([item["correct"] for item in diagnostics], dtype=np.float64)
    accuracy = float(correct.mean()) if len(correct) else 0.0
    low = high = accuracy
    standard_error = 0.0
    if bootstrap_size > 1 and len(correct):
        rng = np.random.default_rng(seed)
        sampled = rng.integers(0, len(correct), size=(bootstrap_size, len(correct)))
        estimates = correct[sampled].mean(1)
        standard_error = float(estimates.std(ddof=1))
        low, high = (float(value) for value in np.percentile(estimates, [2.5, 97.5]))
    for record, diagnostic in zip(records, diagnostics):
        record.update(diagnostic)
    results = {
        "accuracy": accuracy,
        "accuracy_se": standard_error,
        "accuracy_ci_lo": low,
        "accuracy_ci_hi": high,
        "num_correct": int(correct.sum()),
        "num_total": len(records),
        "scoring": method,
        "records": records,
    }
    if method == "code" and diagnostics:
        for key in ("parsed", "executed", "timed_out", "sandbox_error"):
            results[f"{key}_rate"] = float(
                np.mean([bool(item[key]) for item in diagnostics])
            )
    if records and "hit_eos" in records[0]:
        results["eos_rate"] = float(np.mean([record["hit_eos"] for record in records]))
        results["mean_completion_tokens"] = float(
            np.mean([record["completion_tokens"] for record in records])
        )
        results["truncation_rate"] = 1.0 - results["eos_rate"]
    nfes = [float(record["nfe"]) for record in records if "nfe" in record]
    if nfes:
        results["mean_nfe"] = float(np.mean(nfes))
        results["min_nfe"] = float(np.min(nfes))
        results["max_nfe"] = float(np.max(nfes))
    argmax_changes = [
        int(record["argmax_changes"])
        for record in records
        if "argmax_changes" in record
    ]
    if argmax_changes:
        results["mean_argmax_changes"] = float(np.mean(argmax_changes))
        results["min_argmax_changes"] = int(np.min(argmax_changes))
        results["max_argmax_changes"] = int(np.max(argmax_changes))
        results["total_argmax_changes"] = int(np.sum(argmax_changes))
        results["mean_argmax_changed_positions"] = float(np.mean([
            record["argmax_changed_positions"]
            for record in records
            if "argmax_changed_positions" in record
        ]))
    return results


@torch.no_grad()
def evaluate_gsm8k(diffusion_model, config, tokenizer, load_checkpoint, logger):
    if config.gsm8k.input_file:
        loaded = json.loads(Path(config.gsm8k.input_file).read_text())
        records = loaded.get("records", loaded) if isinstance(loaded, dict) else loaded
    else:
        model = _prepare_model(diffusion_model, config, tokenizer, load_checkpoint)
        dataset = get_gsm8k_dataset(config, tokenizer)
        indices = None
        max_examples = getattr(config.gsm8k, "max_examples", None)
        if max_examples is not None:
            count = min(int(max_examples), len(dataset))
            rng = np.random.default_rng(getattr(config.gsm8k, "subset_seed", 42))
            indices = sorted(
                rng.choice(len(dataset), size=count, replace=False).tolist()
            )
        records = generate_gsm8k_records(
            model,
            dataset,
            tokenizer,
            config.loader.eval_batch_size,
            config.sampling.steps,
            indices=indices,
        )
    results = score_gsm8k_records(
        records,
        config.gsm8k.scoring,
        config.gsm8k.timeout,
        config.gsm8k.bootstrap_size,
        config.seed,
    )
    destination = _write_results(config.gsm8k.output_dir, results)
    if "mean_nfe" in results:
        logger.info(
            "GSM8K accuracy %.2f%%; NFE mean/min/max %.2f/%.0f/%.0f; "
            "results saved to %s",
            results["accuracy"] * 100,
            results["mean_nfe"],
            results["min_nfe"],
            results["max_nfe"],
            destination,
        )
    else:
        logger.info(
            "GSM8K accuracy %.2f%%; results saved to %s",
            results["accuracy"] * 100,
            destination,
        )
    if "mean_argmax_changes" in results:
        logger.info(
            "Argmax changes mean/min/max %.2f/%d/%d; total %d; "
            "mean positions changed %.2f",
            results["mean_argmax_changes"],
            results["min_argmax_changes"],
            results["max_argmax_changes"],
            results["total_argmax_changes"],
            results["mean_argmax_changed_positions"],
        )
    return results

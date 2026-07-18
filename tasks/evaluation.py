"""Conditional generation loops for structured reasoning tasks."""

import json
from pathlib import Path

import numpy as np
import torch
from tqdm import trange

from .gsm8k import get_gsm8k_dataset, score_response
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
    batch_size, prompt_length = config.sudoku.batch_size, tokenizer.prompt_len
    records = []
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
        generated = model.generate_samples(
            count,
            num_steps=config.sampling.steps,
            prompt_tokens=prompt_tokens,
            prompt_mask=prompt_mask,
        ).cpu()
        truth = batch[:, prompt_length:]
        predictions = generated[:, prompt_length : prompt_length + truth.shape[1]]
        for prediction, target in zip(predictions, truth):
            metrics = sudoku_metrics(prediction, target)
            records.append(
                {
                    "generated": tokenizer.decode(prediction),
                    "ground_truth": tokenizer.decode(target),
                    **metrics,
                }
            )
    total = max(len(records), 1)
    results = {
        "accuracy": sum(record["exact"] for record in records) / total,
        "cell_accuracy": sum(record["cell_accuracy"] for record in records) / total,
        "valid_grid_rate": sum(record["valid_grid"] for record in records) / total,
        "valid_row_rate": sum(record["valid_rows"] for record in records) / total,
        "valid_column_rate": sum(record["valid_columns"] for record in records) / total,
        "valid_box_rate": sum(record["valid_boxes"] for record in records) / total,
        "num_correct": sum(record["exact"] for record in records),
        "num_total": len(records),
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


@torch.no_grad()
def evaluate_gsm8k(diffusion_model, config, tokenizer, load_checkpoint, logger):
    if config.gsm8k.input_file:
        loaded = json.loads(Path(config.gsm8k.input_file).read_text())
        records = loaded.get("records", loaded) if isinstance(loaded, dict) else loaded
    else:
        model = _prepare_model(diffusion_model, config, tokenizer, load_checkpoint)
        dataset = get_gsm8k_dataset(config, tokenizer)
        records = []
        batch_size = config.loader.eval_batch_size
        for start in trange(0, len(dataset), batch_size, desc="GSM8K generation"):
            items = [
                dataset[index]
                for index in range(start, min(start + batch_size, len(dataset)))
            ]
            count = len(items)
            prompt_tokens = torch.zeros(
                count, model.num_tokens, dtype=torch.long, device=model.device
            )
            prompt_mask = torch.zeros_like(prompt_tokens, dtype=torch.bool)
            lengths = []
            for row, item in enumerate(items):
                ids = item["input_ids"][: model.num_tokens]
                length = len(ids)
                lengths.append(length)
                prompt_tokens[row, :length] = torch.as_tensor(ids, device=model.device)
                prompt_mask[row, :length] = True
            generated = model.generate_samples(
                count,
                num_steps=config.sampling.steps,
                prompt_tokens=prompt_tokens,
                prompt_mask=prompt_mask,
            ).cpu()
            for row, item in enumerate(items):
                records.append(
                    {
                        "prompt": item["prompt"],
                        "response_ground_truth": item["response_ground_truth"],
                        "response": _decode_completion(
                            tokenizer, generated[row], lengths[row]
                        ),
                    }
                )
    correct = np.asarray(
        [
            score_response(
                record["response"],
                record["response_ground_truth"],
                config.gsm8k.scoring,
                config.gsm8k.timeout,
            )
            for record in records
        ],
        dtype=np.float64,
    )
    accuracy = float(correct.mean()) if len(correct) else 0.0
    low = high = accuracy
    standard_error = 0.0
    if config.gsm8k.bootstrap_size > 1 and len(correct):
        rng = np.random.default_rng(config.seed)
        indices = rng.integers(
            0, len(correct), size=(config.gsm8k.bootstrap_size, len(correct))
        )
        estimates = correct[indices].mean(1)
        standard_error = float(estimates.std(ddof=1))
        low, high = (float(value) for value in np.percentile(estimates, [2.5, 97.5]))
    results = {
        "accuracy": accuracy,
        "accuracy_se": standard_error,
        "accuracy_ci_lo": low,
        "accuracy_ci_hi": high,
        "num_correct": int(correct.sum()),
        "num_total": len(records),
        "scoring": config.gsm8k.scoring,
        "records": records,
    }
    destination = _write_results(config.gsm8k.output_dir, results)
    logger.info(
        "GSM8K accuracy %.2f%%; results saved to %s", accuracy * 100, destination
    )
    return results

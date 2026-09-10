"""Shared structured-task encoding, prepared data loading, and scoring."""

import json
from pathlib import Path
import datasets
import numpy as np
import torch
from .sudoku import SudokuTokenizer, encode_example, encode_grid, sudoku_metrics
from .zebra import SYMBOLS, parse_clues, zebra_metrics
from .maze import maze_metrics

TASK_NAMES = frozenset({"sudoku_extreme", "sudoku_shah", "maze_unique", "zebra"})


class SymbolTokenizer:
    def __init__(self, symbols):
        self.symbols = tuple(symbols)
        self.ids = {s: i for i, s in enumerate(symbols)}
        self.vocab_size = len(symbols)
        self.pad_token_id = 0
        self.bos_token_id = 1
        self.eos_token_id = 2
        self.pad_token, self.bos_token, self.eos_token = self.symbols[:3]
        self.mask_token = None
        self.mask_token_id = None

    def __len__(self):
        return self.vocab_size

    def decode(self, ids, skip_special_tokens=False):
        return " ".join(
            self.symbols[int(i)] if 0 <= int(i) < len(self) else "?"
            for i in ids
            if not skip_special_tokens or int(i) > 2
        )

    def batch_decode(self, rows, skip_special_tokens=False):
        return [self.decode(row, skip_special_tokens) for row in rows]


def get_reasoning_tokenizer(name):
    if name in ("sudoku_extreme", "sudoku_shah"):
        return SudokuTokenizer()
    if name == "zebra":
        return SymbolTokenizer(SYMBOLS)
    if name == "maze_unique":
        return SymbolTokenizer(("<PAD>", "<BOS>", "<ANSWER>", "#", " ", "S", "G", "o"))
    raise ValueError(name)


def encode_record(record, tokenizer, prompt_length, length):
    task = record["task"]
    puzzle = record["puzzle"]
    answer = record["answer"]
    if task.startswith("sudoku_"):
        ids, loss = encode_example(puzzle, answer, tokenizer)
        prompt = [int(not x) for x in loss]
        padding = [0] * len(ids)
    else:
        if task == "maze_unique":
            # Map source IDs 1..5 to token IDs 3..7; preserve PAD=0 separately.
            question = [1] + (np.asarray(puzzle).reshape(-1) + 2).tolist() + [2]
            target = (np.asarray(answer).reshape(-1) + 2).tolist()
        else:
            clues, tokens = parse_clues(puzzle)
            del clues
            a, h = np.asarray(answer).shape
            question = [
                1,
                tokenizer.ids["<HOUSES>"],
                tokenizer.ids[str(h)],
                tokenizer.ids["<ATTRS>"],
                tokenizer.ids[str(a)],
            ]
            question += [tokenizer.ids[t] for t in tokens] + [2]
            target = [tokenizer.ids[str(v)] for v in np.asarray(answer).reshape(-1)]
        if len(question) > prompt_length or len(question) + len(target) > length:
            raise ValueError(
                f"example {record['source_id']} exceeds configured layout; never truncate clues"
            )
        if task == "zebra":
            # Pack the target directly after the variable-length prompt, as in
            # the cited masked-diffusion preprocessing. Padding is trailing only.
            ids = question + target + [0] * (length - len(question) - len(target))
            loss = (
                [0] * len(question)
                + [1] * len(target)
                + [0] * (length - len(question) - len(target))
            )
        else:
            ids = (
                question
                + [0] * (prompt_length - len(question))
                + target
                + [0] * (length - prompt_length - len(target))
            )
            loss = (
                [0] * prompt_length
                + [1] * len(target)
                + [0] * (length - prompt_length - len(target))
            )
        # Every non-target position, including trailing padding, stays fixed.
        prompt = [int(not x) for x in loss]
        padding = [int(i == 0) for i in ids]
    if len(ids) != length:
        raise ValueError(f"expected sequence length {length}, got {len(ids)}")
    return dict(
        input_ids=ids,
        attention_mask=loss,
        prompt_mask=prompt,
        padding_mask=padding,
        source_id=record["source_id"],
        group_id=record["group_id"],
        metadata=json.dumps(record),
    )


def load_prepared(config, tokenizer, split):
    root = Path(config.data.prepared_dir)
    manifest_path = root / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Prepare data first: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    if manifest["task"] == "zebra" and manifest.get("encoding_version", 1) != 2:
        raise ValueError(
            "prepared Zebra data uses the obsolete fixed-gap encoding; "
            "prepare a packed v2 dataset"
        )
    if manifest["task"] != config.data.train or manifest["vocab_size"] != len(
        tokenizer
    ):
        raise ValueError("prepared task/vocabulary does not match configuration")
    if (
        manifest["length"] != config.model.length
        or manifest["prompt_length"] != config.data.prompt_length
    ):
        raise ValueError("prepared layout does not match model/data configuration")
    if split == "train" and manifest["full_training"] != config.data.get(
        "full_training", False
    ):
        raise ValueError(
            "full-training dataset requires explicit data.full_training=true and fixed-budget training"
        )
    dataset = datasets.load_from_disk(str(root / "dataset"))[split]
    limit = config.data.get("limit_" + split)
    if limit is not None:
        dataset = dataset.select(range(min(int(limit), len(dataset))))
    return dataset.with_format(
        "torch",
        columns=["input_ids", "attention_mask", "prompt_mask", "padding_mask"],
        output_all_columns=True,
    )


def score_record(prediction, record, tokenizer):
    task = record["task"]
    target = record["answer"]
    puzzle = record["puzzle"]
    if task.startswith("sudoku_"):
        return sudoku_metrics(prediction, encode_grid(target), encode_grid(puzzle))
    if task == "maze_unique":
        return maze_metrics(np.asarray(prediction) - 2, target, puzzle)
    values = []
    for token in prediction:
        symbol = (
            tokenizer.symbols[int(token)] if 0 <= int(token) < len(tokenizer) else ""
        )
        values.append(int(symbol) if symbol.isdigit() else -1)
    return zebra_metrics(values, target, parse_clues(puzzle)[0])


@torch.no_grad()
def evaluate_reasoning(diffusion_model, config, tokenizer, load_checkpoint, logger):
    from .evaluation import _prepare_model, _write_results

    model = _prepare_model(diffusion_model, config, tokenizer, load_checkpoint)
    dataset = load_prepared(config, tokenizer, config.reasoning.split)
    if config.reasoning.max_examples is not None:
        dataset = dataset.select(
            range(min(len(dataset), config.reasoning.max_examples))
        )
    output = Path(config.reasoning.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(
        (Path(config.data.prepared_dir) / "manifest.json").read_text()
    )
    (output / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2))
    diagnostics = getattr(model, "rank_diagnostics", None)
    if callable(diagnostics):
        (output / "rank_diagnostics.json").write_text(json.dumps(diagnostics()))
    metrics = generate_reasoning_metrics(
        model,
        dataset,
        tokenizer,
        config.reasoning.batch_size,
        config.sampling.steps,
        manifest["task"],
        config.reasoning.split,
    )
    _write_results(output, metrics)
    logger.info("Reasoning evaluation accuracy: %.4f", metrics["accuracy"])
    return metrics


@torch.no_grad()
def generate_reasoning_metrics(
    model, dataset, tokenizer, batch_size, num_steps, task, split
):
    """Shared scoring path for periodic monitoring and checkpoint evaluation."""
    import time

    rows = []
    nfes = []
    if model.device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    for offset in range(0, len(dataset), batch_size):
        items = [
            dataset[i] for i in range(offset, min(offset + batch_size, len(dataset)))
        ]
        ids = torch.stack([x["input_ids"] for x in items]).to(model.device)
        masks = torch.stack([x["prompt_mask"] for x in items]).bool().to(model.device)
        # Do not pass ground-truth answer tokens in a request, even under a false mask.
        prompts = torch.where(masks, ids, torch.zeros_like(ids))
        result = model.generate_samples(
            len(items),
            num_steps=num_steps,
            prompt_tokens=prompts,
            prompt_mask=masks,
            return_result=True,
        )
        nfes.extend(
            result.metadata.get(
                "nfe_per_sample",
                [result.metadata.get("nfe", num_steps)] * len(items),
            )
        )
        for item, generated in zip(items, result.samples.cpu()):
            record = json.loads(item["metadata"])
            mask = item["attention_mask"].bool()
            metrics = score_record(generated[mask].tolist(), record, tokenizer)
            rows.append(
                dict(
                    source_id=record["source_id"],
                    prediction=generated[mask].tolist(),
                    **metrics,
                    **(
                        {"layout": f"{record['attributes']}x{record['houses']}"}
                        if record["task"] == "zebra"
                        else {}
                    ),
                )
            )
    if model.device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    if not rows:
        raise ValueError("empty evaluation split")
    metrics = {
        k: sum(float(r[k]) for r in rows) / len(rows)
        for k in rows[0]
        if k not in ("source_id", "prediction", "layout")
    }
    metrics.update(
        accuracy=metrics["exact"],
        num_total=len(rows),
        mean_nfe=float(np.mean(nfes)),
        evaluation_seconds=elapsed,
        seconds_per_puzzle=elapsed / len(rows),
        task=task,
        split=split,
        records=rows,
    )
    if task == "zebra":
        metrics["by_layout"] = {
            layout: dict(
                num_total=sum(r["layout"] == layout for r in rows),
                exact=sum(r["exact"] for r in rows if r["layout"] == layout)
                / sum(r["layout"] == layout for r in rows),
                constraints_satisfied=sum(
                    r["constraints_satisfied"] for r in rows if r["layout"] == layout
                )
                / sum(r["layout"] == layout for r in rows),
            )
            for layout in sorted({r["layout"] for r in rows})
        }
    return metrics

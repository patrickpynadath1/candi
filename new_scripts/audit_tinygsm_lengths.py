#!/usr/bin/env python3
"""Audit TinyGSM/GSM8K lengths with the exact training tokenizer and format."""

import argparse
import json
import datasets
import numpy as np
from transformers import AutoTokenizer


def ids(tokenizer, text):
    return tokenizer(text, add_special_tokens=False)["input_ids"]


def summarize(values):
    array = np.asarray(values, dtype=np.int32)
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "percentiles": {
            str(q): float(np.percentile(array, q))
            for q in (50, 75, 90, 95, 97.5, 99, 99.5, 99.9, 100)
        },
    }


def context_report(lengths, contexts):
    count = len(lengths)
    return {
        str(context): {
            "fit_count": sum(length <= context for length in lengths),
            "fit_fraction": sum(length <= context for length in lengths) / count,
            "mean_utilization_if_filtered": (
                np.mean([length / context for length in lengths if length <= context])
            ),
        }
        for context in contexts
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-train-examples", type=int, default=100_000)
    parser.add_argument("--tokenizer", default="HuggingFaceTB/SmolLM-135M")
    parser.add_argument("--contexts", type=int, nargs="+", default=[256, 384, 512])
    parser.add_argument("--output", default="new_scripts/tinygsm_length_audit.json")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    separator_length = len(ids(tokenizer, "\n"))
    train = datasets.load_dataset("TinyGSM/TinyGSM", split="train", streaming=True)

    full_lengths, prompt_lengths, answer_lengths = [], [], []
    for index, example in enumerate(train):
        if index >= args.max_train_examples:
            break
        question_length = len(ids(tokenizer, example["question"].strip()))
        answer_length = len(ids(tokenizer, example["code"].strip()))
        prompt_length = 1 + question_length + separator_length
        prompt_lengths.append(prompt_length)
        answer_lengths.append(answer_length + 1)  # include EOS
        full_lengths.append(prompt_length + answer_length + 1)

    gsm8k = datasets.load_dataset("gsm8k", "main", split="test")
    eval_prompt_lengths = [
        1 + len(ids(tokenizer, row["question"].strip())) + separator_length
        for row in gsm8k
    ]

    report = {
        "tokenizer": args.tokenizer,
        "format": "BOS + question + newline + Python code + EOS",
        "sample_is_stream_prefix": True,
        "train": {
            "full_length": summarize(full_lengths),
            "prompt_length": summarize(prompt_lengths),
            "answer_length_including_eos": summarize(answer_lengths),
            "contexts": context_report(full_lengths, args.contexts),
        },
        "gsm8k_test": {
            "prompt_length": summarize(eval_prompt_lengths),
            "generation_budget": {
                str(context): summarize(
                    [max(0, context - length) for length in eval_prompt_lengths]
                )
                for context in args.contexts
            },
        },
    }
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

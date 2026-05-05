"""
Compute generative perplexity, generative BPC, and entropy for pre-generated samples stored in JSON files.
Uses Metrics class directly from metrics.py for exact equivalence with the generation script.

Usage:
    python compute_metrics.py \
        --samples_dir /path/to/json/dir \
        [--eval_model gpt2-large] \
        [--eval_ppl_batch_size 1024] \
        [--max_length 128] \
        [--output_path results.json] \
        [--device cuda]
"""

import argparse
import json
from pathlib import Path
import pdb

import torch

from metrics import Metrics


def parse_args():
    p = argparse.ArgumentParser(description="Compute gen-PPL, gen-BPC, and entropy for saved samples.")
    p.add_argument("--samples_dir",         required=True,  help="Directory containing *.json and *.pt sample files")
    p.add_argument("--eval_model",          default="gpt2-large")
    p.add_argument("--eval_model_source",   default="hf", choices=["hf", "local"])
    p.add_argument("--local_eval_checkpoint_path", default=None)
    p.add_argument("--local_eval_device",   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--local_eval_context_size", type=int, default=None)
    p.add_argument("--output_path",         default="metrics_results.json")
    p.add_argument("--eval_ppl_batch_size", type=int, default=16)
    p.add_argument("--max_length",          type=int, default=1024)
    p.add_argument("--device",              default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main():
    args = parse_args()

    # ------------------------------------------------------------------ #
    # 1. Discover all JSON files and find their paired .pt files
    # ------------------------------------------------------------------ #
    json_paths = sorted(Path(args.samples_dir).glob("*.json"))
    if not json_paths:
        raise FileNotFoundError(f"No .json files found in {args.samples_dir}")
    print(f"Found {len(json_paths)} JSON file(s) in {args.samples_dir}")

    # ------------------------------------------------------------------ #
    # 2. Instantiate Metrics once — same as the generation script
    # ------------------------------------------------------------------ #
    metrics = Metrics(
        gen_ppl_eval_model_name_or_path=args.eval_model,
        eval_ppl_batch_size=args.eval_ppl_batch_size,
        eval_model_source=args.eval_model_source,
        local_eval_checkpoint_path=args.local_eval_checkpoint_path,
        local_eval_device=args.local_eval_device,
        local_eval_context_size=args.local_eval_context_size,
    )
    metrics.to(args.device)

    # ------------------------------------------------------------------ #
    # 3. Compute metrics per file
    # ------------------------------------------------------------------ #
    all_results = []
    tokenizer = metrics.tokenizer  # Use the same tokenizer as Metrics for exact equivalence

    for json_path in json_paths:
        # pt_path = json_path.with_suffix(".pt")
        # if not pt_path.exists():
        #     print(f"Skipping {json_path.name} — no matching .pt file found.")
        #     continue

        with open(json_path) as f:
            obj = json.load(f)

        text_samples = obj.get("generated_seqs", [])
        steps = obj.get("steps", None)
        temp  = obj.get("temperature", None)

        if not text_samples:
            print(f"Skipping {json_path.name} — no 'generated_seqs' found.")
            continue

        print(f"\n--- {json_path.name} | steps={steps} temp={temp} | n={len(text_samples)} ---")

        # Reset exactly as the generation script does before each (steps, temp) combo
        metrics.reset()

        # Load raw token ids for entropy — exact same tensors as record_entropy received
        # token_ids = torch.load(pt_path)  # shape: (N, seq_len)
        token_ids = tokenizer.batch_encode_plus(
            text_samples,
            padding=True,
            truncation=True,
            max_length=args.max_length,
            return_tensors="pt"
        )["input_ids"].to(args.device)

        # Entropy — identical call to generation script
        # for sample in range(token_ids.shape[0]): 
        metrics.record_entropy(token_ids)
        entropy = metrics.sample_entropy.compute().item()
        print(f"  entropy        = {entropy:.6f}")

        # Generative perplexity — identical call to generation script
        metrics.record_generative_perplexity(
            text_samples, args.max_length, device=args.device
        )
        gen_ppl = metrics.gen_ppl.compute().item()
        gen_bpc = metrics.gen_bpc.compute().item()
        print(f"  generative_ppl = {gen_ppl:.4f}")
        print(f"  generative_bpc = {gen_bpc:.4f}")

        all_results.append({
            "source_file"   : str(json_path),
            "steps"         : steps,
            "temperature"   : temp,
            "num_samples"   : len(text_samples),
            "generative_ppl": gen_ppl,
            "generative_bpc": gen_bpc,
            "entropy"       : entropy,
        })

    # ------------------------------------------------------------------ #
    # 4. Sort by steps, then temperature
    # ------------------------------------------------------------------ #
    all_results.sort(key=lambda x: (x["steps"], x["temperature"]))

    # ------------------------------------------------------------------ #
    # 5. Save results
    # ------------------------------------------------------------------ #
    output = {
        "eval_model" : args.eval_model,
        "samples_dir": args.samples_dir,
        "results"    : all_results,
    }
    with open(args.output_path, "w") as f:
        json.dump(output, f, indent=4)
    print(f"\nResults saved to {args.output_path}")


if __name__ == "__main__":
    main()

"""
Compute generative perplexity and entropy for pre-generated samples stored in JSON files.

Usage:
    python compute_metrics.py \
        --samples_dir /path/to/json/dir \
        --eval_model gpt2 \
        --tokenizer gpt2 \
        --output_path results.json \
        [--eval_ppl_batch_size 8] \
        [--max_length 1024] \
        [--device cuda]
"""

import argparse
import json
import os
from pathlib import Path

import torch
import torch.nn.functional as F
import torchmetrics
import transformers
from tqdm import tqdm
import math

LOG2 = math.log(2)


# ---------------------------------------------------------------------------
# Metric primitives (mirrors metrics.py exactly)
# ---------------------------------------------------------------------------

class NLL(torchmetrics.aggregation.MeanMetric):
    def update(self, value, weight=1.0):
        if not isinstance(value, torch.Tensor):
            value = torch.as_tensor(value, dtype=self.dtype, device=self.device)
        if weight is not None and not isinstance(weight, torch.Tensor):
            weight = torch.as_tensor(weight, dtype=self.dtype, device=self.device)
        weight = torch.broadcast_to(weight, value.shape)
        value, weight = self._cast_and_nan_check_input(value, weight)
        if value.numel() == 0:
            return
        self.mean_value += value.sum()
        self.weight += weight.sum()


class BPD(NLL):
    def compute(self):
        return self.mean_value / self.weight / LOG2


class Perplexity(NLL):
    def compute(self):
        return torch.exp(self.mean_value / self.weight)


# ---------------------------------------------------------------------------
# Metric computation functions (mirrors Metrics class methods exactly)
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_entropy(token_ids_list: list[torch.Tensor]) -> float:
    """
    Mirrors Metrics.record_entropy.
    Accepts a list of 1-D token-id tensors (one per sample).
    Returns the mean per-sample entropy.
    """
    metric = torchmetrics.aggregation.MeanMetric()
    for sample in token_ids_list:
        _, counts = torch.unique(sample, return_counts=True, sorted=False)
        entropy = torch.special.entr(counts.float() / counts.sum()).sum().item()
        metric.update(entropy)
    return metric.compute().item()


@torch.no_grad()
def compute_generative_perplexity(
    text_samples: list[str],
    eval_model_name_or_path: str,
    max_length: int,
    eval_ppl_batch_size: int = 8,
    device: str = "cuda",
) -> float:
    """
    Mirrors Metrics.record_generative_perplexity + Perplexity.compute.
    Returns scalar perplexity.
    """
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # ---- tokenizer ----
    tokenizer = transformers.AutoTokenizer.from_pretrained(eval_model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # ---- eval model ----
    is_llama2 = "llama2" in eval_model_name_or_path.lower()
    eval_model = (
        transformers.AutoModelForCausalLM.from_pretrained(eval_model_name_or_path)
        .eval()
    )
    if not is_llama2:
        eval_model = eval_model.to(device)

    # ---- retokenize (mirrors _eval_retokenize) ----
    if is_llama2:
        tokenizer_kwargs = dict(
            text=text_samples,          # positional kwarg name for llama path
            return_tensors="pt",
            return_token_type_ids=False,
            return_attention_mask=True,
            truncation=True,
            padding=True,
            max_length=max_length,
        )
        eval_context_size = 4096
    else:
        tokenizer_kwargs = dict(
            return_tensors="pt",
            return_token_type_ids=False,
            return_attention_mask=True,
            truncation=True,
            padding=True,
            max_length=max_length,
        )
        eval_context_size = 1024

    if is_llama2:
        encoded = tokenizer(**tokenizer_kwargs)
    else:
        encoded = tokenizer(text_samples, **tokenizer_kwargs)

    samples   = encoded["input_ids"]
    attn_mask = encoded["attention_mask"]

    if not is_llama2:
        samples   = samples.to(device)
        attn_mask = attn_mask.to(device)

    # ---- batched NLL accumulation (mirrors the loop in record_generative_perplexity) ----
    gen_ppl = Perplexity()
    gen_ppl = gen_ppl.to(device)
    gen_ppl.set_dtype(torch.float64)

    batch_size  = min(eval_ppl_batch_size, samples.shape[0])
    num_batches = samples.shape[0] // batch_size

    for i in tqdm(range(num_batches), desc="Evaluating perplexity"):
        _samples   = torch.split(samples  [i * batch_size:(i + 1) * batch_size], eval_context_size, dim=-1)
        _attn_mask = torch.split(attn_mask[i * batch_size:(i + 1) * batch_size], eval_context_size, dim=-1)

        for sample_chunk, attn_mask_chunk in zip(_samples, _attn_mask):
            logits = eval_model(sample_chunk, attention_mask=attn_mask_chunk)
            logits = logits[0].transpose(-1, -2)          # (B, V, T)
            nlls   = F.cross_entropy(
                logits[..., :-1], sample_chunk[..., 1:], reduction="none"
            )
            first_eos    = (sample_chunk == tokenizer.eos_token_id).cumsum(-1) == 1
            token_mask   = sample_chunk != tokenizer.eos_token_id
            valid_tokens = (first_eos[..., 1:] + token_mask[..., 1:]).to(device)
            nlls         = nlls.to(device)
            gen_ppl.update(nlls * valid_tokens, valid_tokens)

    return gen_ppl.compute().item()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Compute gen-PPL and entropy for saved samples.")
    p.add_argument("--samples_dir",         required=True,  help="Directory containing *.json and *.pt sample files")
    p.add_argument("--eval_model",          default="gpt2-large", help="HF model name/path used for perplexity eval (default: gpt2-large)")
    p.add_argument("--output_path",         default="metrics_results.json", help="Where to write the output JSON")
    p.add_argument("--eval_ppl_batch_size", type=int, default=1024)
    p.add_argument("--max_length",          type=int, default=128)
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
    # 2. Compute metrics per file
    # ------------------------------------------------------------------ #
    all_results = []

    for json_path in json_paths:
        pt_path = json_path.with_suffix(".pt")
        if not pt_path.exists():
            print(f"Skipping {json_path.name} — no matching .pt file found.")
            continue

        # Load metadata + text samples from JSON
        with open(json_path) as f:
            obj = json.load(f)
        text_samples = obj.get("generated_seqs", [])
        steps = obj.get("steps", None)
        temp  = obj.get("temperature", None)

        if not text_samples:
            print(f"Skipping {json_path.name} — no 'generated_seqs' found.")
            continue

        print(f"\n--- {json_path.name} | steps={steps} temp={temp} | n={len(text_samples)} ---")

        # Load raw token ids from .pt for entropy (exact match to original)
        token_ids = torch.load(pt_path)  # shape: (N, seq_len)

        # Entropy — pass each row as a 1-D tensor, exactly as record_entropy does
        entropy = compute_entropy([token_ids[i] for i in range(token_ids.shape[0])])
        print(f"  entropy        = {entropy:.6f}")

        # Generative perplexity — uses decoded text + eval model
        gen_ppl = compute_generative_perplexity(
            text_samples            = text_samples,
            eval_model_name_or_path = args.eval_model,
            max_length              = args.max_length,
            eval_ppl_batch_size     = args.eval_ppl_batch_size,
            device                  = args.device,
        )
        print(f"  generative_ppl = {gen_ppl:.4f}")

        all_results.append({
            "source_file"   : str(json_path),
            "steps"         : steps,
            "temperature"   : temp,
            "num_samples"   : len(text_samples),
            "generative_ppl": gen_ppl,
            "entropy"       : entropy,
        })

    # ------------------------------------------------------------------ #
    # 3. Sort by steps, then temperature
    # ------------------------------------------------------------------ #
    all_results.sort(key=lambda x: (x["steps"], x["temperature"]))

    # ------------------------------------------------------------------ #
    # 4. Save all results
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
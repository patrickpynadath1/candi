"""Evaluate the block checkpoint with speculative or fixed rolling Euler sampling."""

import argparse
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import lightning as L
import numpy as np
from omegaconf import OmegaConf
import torch

from algos import algo_registry
from main import _load_from_checkpoint
from tasks.evaluation import _decode_completion, exact_length_batches, generate_gsm8k_records, score_gsm8k_records
from tasks.gsm8k import get_gsm8k_dataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--max-examples', type=int, default=64)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--warmup-steps', type=int, default=0)
    parser.add_argument('--sampler', choices=['block_diffusion', 'block_diffusion_euler'], default='block_diffusion')
    parser.add_argument('--tokens-per-step', type=int, default=4)
    parser.add_argument('--preview-first', action='store_true')
    args = parser.parse_args()
    checkpoint = Path('/home/patrick/.cache/discrete_diffusion/openwebtext-train/candi_block_64_warmup/checkpoints/last.ckpt')
    label = 'speculative' if args.sampler == 'block_diffusion' else f'euler_tokens{args.tokens_per_step}'
    output = ROOT / 'runs/candi_block_64_warmup' / f'{label}_temp{str(args.temperature).replace(".", "p")}_warmup{args.warmup_steps}_n{args.max_examples}'
    if args.preview_first:
        output = output / 'preview_first'
    output.mkdir(parents=True, exist_ok=True)
    L.seed_everything(1)
    archive = torch.load(checkpoint, map_location='cpu', mmap=True, weights_only=False)
    config = archive['hyper_parameters']['config']
    tokenizer = archive['hyper_parameters']['tokenizer']
    config.eval.checkpoint_path = str(checkpoint)
    config.sampling.predictor = args.sampler
    config.algo.temp = args.temperature
    config.algo.rolling_warmup_steps = args.warmup_steps
    config.algo.euler_tokens_per_step = args.tokens_per_step
    config.algo.euler_nfe_budget = None
    config.data = OmegaConf.load(ROOT / 'configs/data/gsm8k-test.yaml')
    config.data.cache_dir = str(ROOT / 'data_cache')
    config.gsm8k.max_examples = args.max_examples
    config.gsm8k.subset_seed = 42
    config.gsm8k.output_dir = str(output)
    config.loader.eval_batch_size = 1
    OmegaConf.save(config, output / 'config.yaml')
    model = _load_from_checkpoint(algo_registry.get(config.algo.name), config, tokenizer).cuda()
    model._eval_mode()
    dataset = get_gsm8k_dataset(config, tokenizer)
    indices = sorted(np.random.default_rng(42).choice(len(dataset), size=min(args.max_examples, len(dataset)), replace=False).tolist())
    if args.preview_first:
        indices = next(exact_length_batches(dataset, indices, 1))
    calls = []
    live_samples = []
    ordered_indices = [batch[0] for batch in exact_length_batches(dataset, indices, 1)]
    original_generate = model.generate_samples

    def generate_with_metadata(*positional, **kwargs):
        torch.cuda.synchronize()
        start = time.perf_counter()
        result = original_generate(*positional, **kwargs)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        dataset_index = ordered_indices[len(calls)]
        prompt_length = int(kwargs['prompt_mask'][0].sum().item())
        live_samples.append({
            'dataset_index': dataset_index,
            'prompt': dataset[dataset_index]['prompt'],
            'response': _decode_completion(tokenizer, result.samples[0].cpu(), prompt_length),
            'nfe': result.metadata['nfe'],
        })
        (output / 'live_samples.json').write_text(json.dumps(live_samples, indent=2))
        calls.append({**result.metadata, 'seconds': elapsed})
        (output / 'sampler_metadata.json').write_text(json.dumps(calls, indent=2))
        return result

    model.generate_samples = generate_with_metadata
    records = generate_gsm8k_records(model, dataset, tokenizer, 1, config.sampling.steps, indices)
    assert len(records) == len(calls)
    for record, metadata in zip(records, calls):
        record['sampler_metadata'] = metadata
        if args.sampler == 'block_diffusion_euler' and not config.algo.rolling_refinement_steps:
            record['nfe_through_first_eos'] = (
                args.warmup_steps + math.ceil((record['completion_tokens'] + 1) / args.tokens_per_step)
                if record['hit_eos'] else None
            )
    (output / 'generations.json').write_text(json.dumps(records, indent=2))
    results = score_gsm8k_records(records, 'code', config.gsm8k.timeout, config.gsm8k.bootstrap_size, 1)
    accepted = sum(m['accepted'] for m in calls)
    rejected = sum(m['rejected'] for m in calls)
    windows = sum(m['verification_sequences'] for m in calls)
    results['speculative'] = {
        'acceptance_scope': 'full generated capacity, including positions after EOS',
        'accepted': accepted, 'rejected': rejected,
        'acceptance_rate': accepted / (accepted + rejected) if accepted + rejected else None,
        'mean_accepted_prefix': sum(m['accepted_prefix_total'] for m in calls) / windows if windows else None,
        'mean_verified_advance': sum(m['verified_advance_total'] for m in calls) / windows if windows else None,
        'generation_seconds': sum(m['seconds'] for m in calls),
        'batch_size': 1, 'checkpoint_step': archive['global_step'],
    }
    results['answer_efficiency'] = {
        'completion_tokens_before_eos': sum(r['completion_tokens'] for r in records),
        'total_nfe_including_post_eos_work': sum(r['nfe'] for r in records),
        'answer_tokens_per_actual_nfe': sum(r['completion_tokens'] for r in records) / sum(r['nfe'] for r in records),
        'nfe_to_first_eos': None,
        'pre_eos_acceptance_rate': None,
        'note': 'The sampler does not record the EOS round; full-sequence acceptance is not answer-only acceptance.',
    }
    if args.sampler == 'block_diffusion_euler':
        results['rolling'] = results.pop('speculative')
        results['rolling']['sampler'] = args.sampler
        results['rolling']['tokens_per_step'] = args.tokens_per_step
        eos_nfes = [r['nfe_through_first_eos'] for r in records if r.get('nfe_through_first_eos') is not None]
        results['answer_efficiency']['mean_nfe_through_first_eos'] = float(np.mean(eos_nfes)) if eos_nfes else None
        results['answer_efficiency']['num_examples_with_eos_nfe'] = len(eos_nfes)
        results['answer_efficiency']['note'] = 'NFE through first EOS is derived from the fixed token advance, includes its committing pass and warmup, and excludes subsequent passes.'
    (output / 'results.json').write_text(json.dumps(results, indent=2))
    print(json.dumps({k: v for k, v in results.items() if k != 'records'}, indent=2), flush=True)
    print(f'Results: {output / "results.json"}', flush=True)


if __name__ == '__main__':
    main()

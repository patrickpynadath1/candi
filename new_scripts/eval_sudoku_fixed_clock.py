#!/usr/bin/env python3
"""Evaluate frozen training code with its learned time warp bypassed after loading."""

import argparse
import json
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--source', type=Path, required=True)
    args, hydra_args = parser.parse_known_args()
    sys.path.insert(0, str(args.source.resolve()))
    sys.argv = [str(args.source / 'main.py'), '--config-path',
                str(args.source.resolve() / 'configs'), *hydra_args]
    import main as entry
    import torch
    from algos.noise_schedules import AdaptiveSchedule

    original_loader = entry._load_from_checkpoint

    def load_with_fixed_clock(diffusion_model, config, tokenizer):
        model = original_loader(diffusion_model, config, tokenizer)
        if not isinstance(model.noise, AdaptiveSchedule):
            raise ValueError('Expected an adaptive-trained checkpoint')
        clock = torch.linspace(0, 1, 101, dtype=torch.float64)
        learned_alpha = model.noise.alpha_t(clock).tolist()
        model.noise = model.noise.base
        fixed_alpha = model.noise.alpha_t(clock)
        expected = fixed_alpha[0] + clock * (fixed_alpha[-1] - fixed_alpha[0])
        if not torch.allclose(fixed_alpha, expected, atol=1e-6):
            raise ValueError('Base schedule is not linear')
        output = Path(config.sudoku.output_dir)
        output.mkdir(parents=True, exist_ok=True)
        (output / 'inference_schedule.json').write_text(json.dumps(dict(
            override='fixed_linear_clock', checkpoint=config.eval.checkpoint_path,
            clock=clock.tolist(), learned_alpha=learned_alpha,
            inference_alpha=fixed_alpha.tolist(),
            corruption_path_unchanged=True,
        ), indent=2))
        print('Using fixed linear inference clock; learned weights and corruption path unchanged', flush=True)
        return model

    entry._load_from_checkpoint = load_with_fixed_clock
    entry.main()


if __name__ == '__main__':
    main()

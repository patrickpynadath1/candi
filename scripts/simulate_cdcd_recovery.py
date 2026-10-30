"""CPU-only stress test of adaptive coverage, not a neural training experiment.

Use a real legacy checkpoint to start from the observed concentrated CDF. Then
feed both fitters a known loss curve whose low-noise region worsens halfway
through. No model weights, checkpoint files, or running jobs are modified.
"""
import argparse
import json
from pathlib import Path

import torch
from core.time_sampling.cdcd import CDCDTimeSampler
from core.time_sampling.loss_derivative import LossDerivativeTimeSampler, interpolate


def run(checkpoint, steps=4000, batch=512, seeds=(0, 1, 2), worsen=True):
    torch.set_num_threads(1)
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False, mmap=True)
    prefix = 'algorithm.time_sampler.'
    state = {k[len(prefix):]: v.clone() for k, v in saved['state_dict'].items() if k.startswith(prefix)}
    epsilon = saved['hyper_parameters']['config']['time_sampler']['epsilon']
    del saved
    grid, profile = state['grid'], state['profile']
    results = []
    for seed in seeds:
        for kind in ['legacy', 'cdcd_concentrated', 'cdcd_fresh']:
            torch.manual_seed(seed)
            if kind == 'legacy':
                s = LossDerivativeTimeSampler(adaptive=True, epsilon=epsilon)
                s.load_state_dict(state)
            else:
                s = CDCDTimeSampler(adaptive=True, epsilon=epsilon)
                if kind == 'cdcd_concentrated':
                    # Approximate the real proposal on fixed input bins and add
                    # CDCD's required bin safeguards. This is a deliberate stress
                    # initialization, NOT the normal fresh-sampler migration.
                    edges = torch.linspace(0, 1, s.bins + 1)
                    masses = interpolate(edges, grid, state['cdf']).diff().clamp_min(1e-12)
                    logits = (masses * profile[-1].clamp_min(.01)).log()
                    s.logits[1].copy_(logits); s.ema_logits.copy_(s.logits)
                    s.warmup = 0; s.ready.fill_(True)
            windows = []
            hits = 0; total = 0; first_hit = None
            for step in range(steps):
                t = s.sample(batch, 'cpu')
                worsening = worsen and step >= steps//2
                # Fixed profile from the real checkpoint plus an explicitly
                # synthetic low-noise deterioration. The target is observed only
                # at sampled times, just as during actual training.
                y = interpolate(t, grid, profile)
                if worsening: y = y + .5 * (1-t).square()
                s.observe(t, y[:, None], torch.ones(batch, 1, dtype=torch.bool), 250000+step)
                if kind != 'legacy': s.post_optimizer_step()
                low = int((t < .5).sum()); hits += low; total += batch
                if worsening and low and first_hit is None: first_hit = step - steps//2
                if (step+1) % 500 == 0:
                    windows.append({'step': step+1, 'low_noise_draws': hits,
                                    'total_draws': total, 'fraction_below_half': hits/total})
                    hits = total = 0
            row = {'worsen':worsen, 'seed':seed,'sampler':kind,'first_low_noise_batch_after_change':first_hit,'windows':windows}
            results.append(row); print(json.dumps(row),flush=True)
    return results


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('checkpoint',type=Path);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--steps',type=int,default=4000)
    p.add_argument('--no-worsening',action='store_true')
    p.add_argument('--seeds',type=int,nargs='+',default=[0,1,2])
    args=p.parse_args()
    data=run(args.checkpoint,args.steps,seeds=args.seeds,worsen=not args.no_worsening)
    args.output.write_text(json.dumps(data,indent=2))

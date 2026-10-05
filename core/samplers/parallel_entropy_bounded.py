"""Entropy-bounded commitment for posterior-driven parallel noisy views.

Frozen acceptance is cumulative. Revisable acceptance is step-local, with
simultaneous selection of all generated positions required for early stopping.
Released continuous commitments are forward-corrupted at destination time;
positions already noisy follow their configured reverse update.
"""
import torch
from core.modifiers.hybrid.views import prior, clamp
from core.modifiers.self_conditioning.session import prediction_session
from core.processes.absorbing import AbsorbingProcess
from core.processes.uniform import UniformProcess
from .entropy_bounded import select_prefix
from .hybrid_entropy_bounded import MaskedHybridEntropyBoundedSampler


class ParallelEntropyBoundedSampler(MaskedHybridEntropyBoundedSampler):
    def __init__(self, *, freeze=True, **kwargs):
        super().__init__(**kwargs)
        self.freeze = bool(freeze)

    @torch.no_grad()
    def sample(self, algo, prompt_tokens, prompt_mask):
        algo = prediction_session(algo)
        absorbing = [isinstance(v.process, AbsorbingProcess) for v in algo.views]
        if not self.freeze and any(absorbing):
            raise ValueError('Revisable EB cannot bypass absorbing posterior constraints')
        batch = len(prompt_tokens)
        generated = ~prompt_mask.bool()
        times = torch.linspace(1, self.epsilon, self.steps + 1, device=prompt_tokens.device)
        points = algo.points(times[0].expand(batch), corruption_mask=generated)
        states = tuple(prior(v, prompt_tokens, p, prompt_mask) for v, p in zip(algo.views, points))
        committed = prompt_mask.bool().clone()
        complete = ~generated.any(-1)
        output = prompt_tokens.clone()
        per_sample = torch.zeros(batch, dtype=torch.long, device=output.device)
        counts = []
        for index in range(self.steps):
            active = (generated & ~committed if self.freeze else generated) & ~complete[:, None]
            if not active.any():
                break
            time, next_time = times[index].expand(batch), times[index + 1].expand(batch)
            source = algo.points(time, corruption_mask=generated)
            destination = algo.points(next_time, corruption_mask=generated)
            probabilities = algo.sampling_prediction(states, source, time).float().exp()
            entropy = -torch.special.xlogy(probabilities, probabilities).sum(-1)
            selected = select_prefix(entropy, active, self.bound)
            clean = (probabilities.argmax(-1) if self.greedy_clean_estimate
                     else torch.distributions.Categorical(probabilities).sample())
            next_output = torch.where(selected, clean, output)
            released = committed & generated & ~selected & ~complete[:, None]
            next_committed = (committed | selected) if self.freeze else (prompt_mask.bool() | selected)
            updated = []
            for is_absorbing, view, state, pt, ps in zip(absorbing, algo.views, states, source, destination):
                value = state if is_absorbing else view.update(
                    view.process, view.representation, state, probabilities, pt, ps,
                    noise=torch.zeros_like(state),
                )
                if not self.freeze and not isinstance(view.process, UniformProcess) and released.any():
                    noisy = view.process.corrupt_state(view.representation.encode(output), ps.coefficients, released)
                    mask = released
                    while mask.ndim < value.ndim:
                        mask = mask.unsqueeze(-1)
                    value = torch.where(mask, noisy, value)
                value = clamp(view, value, next_output, next_committed)
                done = complete
                while done.ndim < value.ndim:
                    done = done.unsqueeze(-1)
                updated.append(torch.where(done, state, value))
            states = tuple(updated)
            output = torch.where(complete[:, None], output, next_output)
            committed = next_committed
            counts.append(selected.sum(-1).cpu().tolist())
            newly_complete = ~complete & committed.all(-1)
            per_sample[newly_complete] = len(counts)
            complete |= newly_complete
        if not complete.all():
            # Read out unfinished sequences at the endpoint; do not present
            # this fallback as successful entropy-bounded acceptance.
            final_time = times[-1].expand(batch)
            points = algo.points(final_time, corruption_mask=generated)
            prediction = algo.sampling_prediction(states, points, final_time)
            fallback = (prediction.argmax(-1) if self.greedy_clean_estimate
                        else torch.distributions.Categorical(logits=prediction).sample())
            output = torch.where(complete[:, None] | committed, output, fallback)
            per_sample[~complete] = len(counts) + 1
        output = torch.where(prompt_mask.bool(), prompt_tokens, output)
        return output, {
            'nfe': len(counts) + int(not complete.all()),
            'nfe_per_sample': per_sample.cpu().tolist(),
            'accepted_per_step': counts,
            'completed_per_sample': complete.cpu().tolist(),
            'entropy_bound': self.bound,
            'freeze': self.freeze,
        }

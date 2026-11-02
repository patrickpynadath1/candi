"""Blockwise latent ODE generation with a separate terminal token decoder."""

import torch
from .loop import FixedGridLoop, LinearTimeGrid


class LatentAutoencoderSampler:
    def __init__(self, steps=15, epsilon=1e-5, temperature=1.0, greedy=True):
        if (
            type(steps) is not int
            or steps < 1
            or not 0 < epsilon < 1
            or temperature <= 0
        ):
            raise ValueError("invalid latent sampler settings")
        self.steps, self.epsilon, self.temperature, self.greedy = (
            steps,
            epsilon,
            temperature,
            greedy,
        )

    def steps_for_nfe(self, nfe):
        # A terminal denoiser call plus token decoding are additional evaluations.
        if nfe < 3:
            raise ValueError("latent sampling needs at least three NFE per block")
        return nfe - 2

    @torch.no_grad()
    def sample(self, algo, prompt_tokens, prompt_mask):
        model = algo.model
        result = prompt_tokens.clone()
        counts = []
        for row in range(len(result)):
            mask = prompt_mask[row].bool()
            prefix = int(mask.sum())
            if not torch.equal(
                mask, torch.arange(len(mask), device=mask.device) < prefix
            ):
                raise ValueError("latent sampling requires a contiguous prompt prefix")
            if prefix != len(mask) and prefix % model.block_size:
                raise ValueError("prompt must end at a block boundary")
            encoder_cache, denoiser_cache, decoder_cache = {}, {}, {}
            nfe = 0
            for start in range(prefix, result.shape[1], model.block_size):
                end = min(start + model.block_size, result.shape[1])
                context = model.encode(
                    result[row : row + 1, :start], cache=encoder_cache
                )
                state = torch.randn(
                    1, end - start, model.hidden_size, device=result.device
                )
                times = LinearTimeGrid(end=self.epsilon, steps=self.steps)(
                    result.device
                )

                def update(state, time, next_time, _):
                    time = time.expand(1)
                    estimate = model.denoise(
                        state, time, context, offset=start, cache=denoiser_cache
                    )
                    path = algo.schedule(time)
                    velocity = algo.process.reverse_vector_field(
                        state,
                        estimate,
                        path.alpha,
                        path.sigma,
                        path.dalpha_dt,
                        path.dsigma_dt,
                    )
                    return state + (next_time - time)[:, None, None] * velocity

                state = FixedGridLoop()(state, times, update)
                estimate = model.denoise(
                    state,
                    times[-1].expand(1),
                    context,
                    offset=start,
                    cache=denoiser_cache,
                )
                logits = model.decode(
                    estimate, context, offset=start, cache=decoder_cache
                )
                probs = (logits.float() / self.temperature).softmax(-1)
                tokens = (
                    probs.argmax(-1)
                    if self.greedy
                    else torch.multinomial(probs.flatten(0, 1), 1).reshape(1, -1)
                )
                result[row : row + 1, start:end] = tokens
                nfe += self.steps + 2
            counts.append(nfe)
        return result, {"nfe": max(counts, default=0), "nfe_per_sample": counts}

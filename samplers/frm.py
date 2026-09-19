"""Recurrent probability-flow samplers for Flow Reasoning Models."""

import torch

from .base import Sampler, SamplerSpec, SamplingResult
from .registry import sampler_registry


def symmetric_kl(left, right, eps=1e-8):
    left = left.clamp_min(eps)
    right = right.clamp_min(eps)
    return 0.5 * (
        (left * (left.log() - right.log())).sum(-1)
        + (right * (right.log() - left.log())).sum(-1)
    )


class FRMSampler(Sampler):
    spec = SamplerSpec(
        "frm_euler",
        frozenset({"frm"}),
        frozenset({"frm_denoise", "frm_sudoku_layout"}),
    )

    def __init__(self, held=False, flow_steps=None):
        self.held = held
        self.flow_steps = flow_steps

    def sample(self, model, request):
        if request.prompt_tokens is None:
            raise ValueError("FRM sampling requires a Sudoku puzzle prompt")
        clues = model.clues_from_prompt(request.prompt_tokens)
        clue_mask = clues.ne(0)
        nfe = request.num_steps or model.config.sampling.steps
        if nfe < 2:
            raise ValueError("FRM sampling requires at least two function evaluations")
        flow_nfe = nfe
        if self.held:
            configured = self.flow_steps
            flow_nfe = max(2, min(nfe, configured if configured is not None else 16))

        state = model.clamp_state(model.prior_sample(request.num_samples), clues)
        carry = torch.zeros_like(state)
        previous = None
        residual = torch.full(
            (request.num_samples,), float("nan"), device=model.device
        )
        # Count the endpoint readout in NFE. The last flow call therefore does
        # not perform another Euler transition.
        times = torch.linspace(0, 1 - request.eps, flow_nfe, device=model.device)
        for index, time_scalar in enumerate(times):
            time = time_scalar.expand(request.num_samples)
            prediction = model.predict(state, time, clues, carry)
            if previous is not None:
                residual = symmetric_kl(prediction, previous).mean(-1)
            if index + 1 < flow_nfe:
                next_time = times[index + 1]
                delta = next_time - time_scalar
                state = state + delta * (prediction - state) / (1 - time_scalar).clamp_min(1e-5)
                state = model.clamp_state(state, clues)
            previous, carry = prediction, prediction

        # Held-time reasoning refines only the clean prediction/carry; the flow
        # state and its terminal time remain fixed.
        for _ in range(nfe - flow_nfe):
            prediction = model.predict(state, times[-1].expand(request.num_samples), clues, carry)
            residual = symmetric_kl(prediction, carry).mean(-1)
            previous, carry = prediction, prediction

        solution = carry.argmax(-1)
        solution = torch.where(clue_mask, clues, solution)
        samples = model.serialize_solution(solution, request.prompt_tokens)
        return SamplingResult(
            samples,
            {
                "nfe": nfe,
                "nfe_per_sample": [nfe] * request.num_samples,
                "flow_nfe": flow_nfe,
                "held_nfe": nfe - flow_nfe,
                "residual_per_sample": residual.detach().cpu().tolist(),
            },
        )


class FRMHeldSampler(FRMSampler):
    spec = SamplerSpec(
        "frm_held",
        frozenset({"frm"}),
        frozenset({"frm_denoise", "frm_sudoku_layout"}),
    )


@sampler_registry.register("frm_euler")
def _create_euler(config):
    del config
    return FRMSampler()


@sampler_registry.register("frm_held")
def _create_held(config):
    return FRMHeldSampler(
        held=True,
        flow_steps=getattr(config.sampling, "frm_flow_steps", None),
    )

"""Generated training examples and teacher/auxiliary targets.

Source states, forward noising and reverse bridges are independently configured.
Each student proposes a clean distribution from a sampled noisy source state.
"""

from contextlib import nullcontext
import torch
from torch import nn
from core.processes.absorbing import AbsorbingProcess
from core.processes.gaussian import GaussianProcess


class ForwardTokens(nn.Module):
    def forward(self, view, proposal, source_state, source, destination, mask):
        tokens = torch.distributions.Categorical(probs=proposal.detach()).sample()
        return view.process.corrupt_state(
            view.representation.encode(tokens), destination.coefficients, mask
        )


class ReverseBridge(nn.Module):
    def forward(self, view, proposal, source_state, source, destination, mask):
        if not isinstance(view.process, AbsorbingProcess):
            raise ValueError("initial bridge preset supports absorbing diffusion only")
        return view.process.reverse(
            source_state,
            proposal.detach(),
            source.coefficients[0],
            destination.coefficients[0],
        )


class GaussianForward(nn.Module):
    def forward(self, view, proposal, source_state, source, destination, mask):
        if not isinstance(view.process, GaussianProcess):
            raise ValueError("pathwise Gaussian noising requires GaussianProcess")
        return view.process.corrupt_state(
            view.representation.posterior_mean(proposal), destination.coefficients, mask
        )


class GeneratedPair(nn.Module):
    def __init__(
        self,
        noising,
        source,
        detach_proposal=False,
        minimum_time=0.05,
        maximum_time=0.95,
    ):
        super().__init__()
        self.noising, self.detach_proposal = noising, detach_proposal
        self.source = source
        if not 0 < minimum_time < maximum_time < 1:
            raise ValueError("matching times must be strictly inside (0,1)")
        self.minimum_time, self.maximum_time = minimum_time, maximum_time

    def forward(self, models, batch, step):
        student, teacher = models["student"], models["teacher"]
        if len(student.views) != 1 or len(teacher.views) != 1:
            raise ValueError("generated one-step presets require a single view")
        tokens, mask = batch["input_ids"], batch["loss_mask"].bool()
        corrupt = batch.get("corruption_mask", mask).bool()
        origin = self.source(models, batch, step)
        source_time = origin["time"]
        source = origin["points"][0]
        view = teacher.views[0]  # fixed teacher coordinates for soft embeddings
        initial = origin["states"][0]
        with torch.no_grad() if self.detach_proposal else nullcontext():
            logp = student.training_prediction((initial,), (source,), source_time)
            proposal = logp.exp()
        time = source_time * (
            self.minimum_time
            + (self.maximum_time - self.minimum_time) * torch.rand_like(source_time)
        )
        points = teacher.points(time)
        noisy = self.noising(view, proposal, initial, source, points[0], corrupt)
        # Reapply the observed prompt after categorical proposals/noising.
        from core.modifiers.hybrid.views import clamp

        noisy = clamp(view, noisy, tokens, ~corrupt)
        active = (
            noisy.eq(view.process.mask_index)
            if isinstance(view.process, AbsorbingProcess)
            else mask
        )
        return dict(
            states=(noisy,),
            points=points,
            time=time,
            mask=mask,
            tokens=tokens,
            active=active,
            proposal=proposal,
            clean=(
                view.representation.posterior_mean(proposal)
                if isinstance(view.process, GaussianProcess)
                else proposal
            ),
        )


class FieldPrediction(nn.Module):
    def __init__(self, field="proposal"):
        super().__init__()
        self.field = field

    def forward(self, models, state):
        return state[self.field]


class RolePosterior(nn.Module):
    def __init__(self, role="auxiliary", mean=False):
        super().__init__()
        self.role, self.mean = role, mean

    def forward(self, models, state):
        logp = models[self.role].training_prediction(
            state["states"], state["points"], state["time"]
        )
        return (
            models["teacher"].views[0].representation.posterior_mean(logp.exp())
            if self.mean
            else logp
        )


class PosteriorDifference(nn.Module):
    def __init__(self, weighting=None):
        super().__init__()
        self.weighting = weighting

    @torch.no_grad()
    def forward(self, models, state):
        args = state["states"], state["points"], state["time"]
        teacher = models["teacher"].sampling_prediction(*args)
        auxiliary = models["auxiliary"].sampling_prediction(*args)
        result = auxiliary.float() - teacher.float()
        result = torch.where(torch.isfinite(result), result, 0)
        return (
            result
            if self.weighting is None
            else result * self.weighting(state)[..., None]
        )


class ProposalTargets(nn.Module):
    def __init__(self, teacher_weight=0.0, mean=False, weighting=None):
        super().__init__()
        self.teacher_weight, self.mean = teacher_weight, mean
        self.weighting = weighting

    @torch.no_grad()
    def forward(self, models, state):
        value = state["clean"] if self.mean else state["proposal"]
        if self.teacher_weight:
            teacher = (
                models["teacher"]
                .sampling_prediction(state["states"], state["points"], state["time"])
                .exp()
            )
            value = value + self.teacher_weight * teacher
        if self.weighting is not None:
            value = value * self.weighting(state)[..., None]
        return value.detach()


class MaskedDenoisingWeight(nn.Module):
    def forward(self, state):
        keep, derivative = state["points"][0].coefficients
        return state["active"] * (-derivative / (1 - keep).clamp_min(1e-8))[:, None]


class GaussianScoreDifference(nn.Module):
    @torch.no_grad()
    def forward(self, models, state):
        teacher = models["teacher"]
        if not isinstance(teacher.views[0].process, GaussianProcess):
            raise ValueError("Gaussian score estimator requires GaussianProcess")
        args = state["states"], state["points"], state["time"]
        representation = teacher.views[0].representation
        real = representation.posterior_mean(teacher.sampling_prediction(*args).exp())
        fake = representation.posterior_mean(
            models["auxiliary"].sampling_prediction(*args).exp()
        )
        c = state["points"][0].coefficients
        # d KL(q_t||p_t)/d clean = alpha*(score_q-score_p).
        return ((c.alpha / c.sigma).square()[:, None, None] * (fake - real)).detach()

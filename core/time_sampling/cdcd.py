"""CDCD Appendix A.1--A.4 learned piecewise-linear loss CDF.

https://arxiv.org/abs/2211.15089
Both axes have learned bin widths. The unnormalised output heights fit sequence
cross entropy with inverse proposal-density weighted MSE. EMA logits drive the
proposal. Positive bin pseudocounts are numerical safeguards, NOT the optional
uniform mixture from A.6 (which is deliberately not implemented).

Fitting state is buffers, separate from denoiser parameters/EMA/AdamW. Gradients
are accumulated across microbatches and ranks, then a separate Adam update is
committed by TrainingModule's post_optimizer_step hook. Checkpoints include all
fitting and pending-gradient state. Legacy loss_derivative checkpoints must not
be loaded as this sampler without explicitly resetting the sampler state.
"""

import math
import torch
from torch import nn
from .uniform import UniformTimeSampler
from .simplex_grid import SimplexInferenceGrid


class CDCDTimeSampler(nn.Module):
    def __init__(self, adaptive=False, bins=100, min_bin_size=1e-4,
                 learning_rate=1e-3, momentum=0.99, warmup=1000,
                 refit_every=1, epsilon=1e-5, antithetic=True):
        super().__init__()
        if (not isinstance(bins, int) or bins < 2 or
                not math.isfinite(min_bin_size) or not 0 < min_bin_size < 1 / bins or
                not math.isfinite(learning_rate) or learning_rate <= 0 or
                not 0 <= momentum < 1 or warmup < 0 or refit_every < 1):
            raise ValueError('invalid CDCD time sampler settings')
        self.adaptive, self.bins = adaptive, bins
        self.min_bin_size, self.learning_rate = min_bin_size, learning_rate
        self.momentum, self.warmup, self.refit_every = momentum, warmup, refit_every
        self.uniform = UniformTimeSampler(epsilon, antithetic)
        initial = torch.full((2, bins), -math.log(bins))
        for name, value in {
            'logits': initial, 'ema_logits': initial.clone(),
            'adam_first': torch.zeros_like(initial),
            'adam_second': torch.zeros_like(initial),
            'pending_gradient': torch.zeros_like(initial),
            'pending_count': torch.tensor(0., dtype=torch.float32),
            'count': torch.tensor(0, dtype=torch.long),
            'updates': torch.tensor(0, dtype=torch.long),
            'last_step': torch.tensor(-1, dtype=torch.long),
            'ready': torch.tensor(False),
        }.items():
            self.register_buffer(name, value)

    def _widths(self, logits):
        # Adding the same pseudocount to BOTH axes does not mix F(t) with t.
        widths = logits.float().softmax(-1) + self.min_bin_size
        return widths / widths.sum(-1, keepdim=True)

    @staticmethod
    def _piecewise(x, widths_in, widths_out):
        left_in = torch.cat((widths_in.new_zeros(1), widths_in.cumsum(0)[:-1]))
        left_out = torch.cat((widths_out.new_zeros(1), widths_out.cumsum(0)[:-1]))
        index = torch.searchsorted(left_in.detach().contiguous(), x.contiguous(), right=True)
        index = (index - 1).clamp(0, len(widths_in) - 1)
        fraction = ((x - left_in[index]) / widths_in[index]).clamp(0, 1)
        value = left_out[index] + fraction * widths_out[index]
        slope = widths_out[index] / widths_in[index]
        return value, slope

    def _unit_time(self, times):
        return ((times.float() - self.uniform.epsilon) / (1 - self.uniform.epsilon)).clamp(0, 1)

    def cdf(self, times):
        widths = self._widths(self.ema_logits)
        return self._piecewise(self._unit_time(times), widths[0], widths[1])[0]

    def density(self, times):
        widths = self._widths(self.ema_logits)
        return self._piecewise(self._unit_time(times), widths[0], widths[1])[1] / (1 - self.uniform.epsilon)

    @torch.no_grad()
    def inverse_cdf(self, probabilities):
        widths = self._widths(self.ema_logits)
        x = self._piecewise(probabilities.float().clamp(0, 1), widths[1], widths[0])[0]
        return self.uniform.epsilon + (1 - self.uniform.epsilon) * x

    @torch.no_grad()
    def forward(self, uniform_times):
        if not self.adaptive or not bool(self.ready):
            return uniform_times
        # Epsilon bounds physical time, not CDF probability: clipping u would
        # once again remove low-mass regions from the sampling support.
        return self.inverse_cdf(self._unit_time(uniform_times))

    def sample(self, count, device, **context):
        return self(self.uniform.sample(count, device, **context))

    def fitting_loss(self, logits, times, losses, proposal_density):
        """Per-observation weighted MSE; no gradients to observations/proposal."""
        widths = self._widths(logits)
        fitted, _ = self._piecewise(self._unit_time(times.detach()), widths[0], widths[1])
        fitted = fitted * logits[1].float().exp().sum()
        return (fitted - losses.detach()).square() / proposal_density.detach()

    @torch.no_grad()
    def observe(self, times, token_loss, mask, step):
        if not self.adaptive:
            return
        weights = torch.ones_like(token_loss) if mask is None else mask.to(token_loss)
        lengths = weights.sum(-1)
        losses = (torch.where(weights > 0, token_loss.detach(), 0) * weights).sum(-1) / lengths.clamp_min(1)
        valid = ((lengths > 0) & torch.isfinite(losses) & torch.isfinite(times)
                 & (times >= self.uniform.epsilon) & (times <= 1))
        times, losses = times.detach()[valid].float(), losses[valid].float()
        # Sampling logits stay fixed until post_optimizer_step, so this is the
        # actual proposal used for every microbatch in the full global batch.
        proposal = (self.density(times) if bool(self.ready) else
                    torch.full_like(times, 1 / (1 - self.uniform.epsilon)))
        gradient = torch.zeros_like(self.logits)
        if len(times):
            with torch.enable_grad(), torch.autocast(times.device.type, enabled=False):
                logits = self.logits.detach().clone().requires_grad_(True)
                objective = self.fitting_loss(logits, times, losses, proposal).sum()
                gradient = torch.autograd.grad(objective, logits)[0].detach()
        packed = torch.cat((gradient.flatten(), gradient.new_tensor([len(times)])))
        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(packed)
        if not torch.isfinite(packed).all():
            raise FloatingPointError('nonfinite CDCD CDF fitting gradient')
        self.pending_gradient.add_(packed[:-1].reshape_as(gradient))
        self.pending_count.add_(packed[-1])
        self.count.add_(packed[-1].long())
        self.last_step.fill_(step)

    @torch.no_grad()
    def post_optimizer_step(self):
        if (not self.adaptive or self.pending_count <= 0 or
                (int(self.last_step) + 1) % self.refit_every):
            return
        gradient = self.pending_gradient / self.pending_count
        # A dedicated Adam, persisted as buffers so the denoiser optimizer and
        # model EMA never update these logits or change their parameter ordering.
        self.updates.add_(1)
        n = int(self.updates)
        self.adam_first.lerp_(gradient, 0.1)
        self.adam_second.lerp_(gradient.square(), 0.001)
        update = (self.adam_first / (1 - 0.9 ** n)) / (
            (self.adam_second / (1 - 0.999 ** n)).sqrt() + 1e-8)
        self.logits.add_(update, alpha=-self.learning_rate)
        self.ema_logits.lerp_(self.logits, 1 - self.momentum)
        self.pending_gradient.zero_()
        self.pending_count.zero_()
        # Warmup belongs to the fitter, not the denoiser global step: an
        # explicitly reset sampler on a resumed run needs its own bootstrap.
        self.ready.fill_(n >= self.warmup)

    def sampling_grid(self, steps, schedule, device):
        return SimplexInferenceGrid(schedule, self).build(steps, device)

"""Full hybrid ELBO; see docs/replaid_hybrid_derivation.tex."""
import math

import torch
import torch.nn.functional as F

from .registry import algo_registry
from .replaid import RePlaid
from models.ema import ExponentialMovingAverage


class HybridClock(torch.nn.Module):
    """Ordered Gaussian endpoints and a learned piecewise-uniform time density.

    Its inverse CDF is the shared clock: r(t)=u(t), gamma(t)=lo+width*u(t).
    Density parameters receive only an unbiased second-moment score gradient.
    """

    def __init__(self, lower, upper, bins=64, uniform_mix=0.01):
        super().__init__()
        if upper <= lower or bins < 1 or not 0 < uniform_mix <= 1:
            raise ValueError("invalid hybrid endpoints, bins, or uniform mixture")
        self.gamma_min = torch.nn.Parameter(torch.tensor(float(lower)))
        self.width_raw = torch.nn.Parameter(torch.tensor(math.log(math.expm1(upper-lower))))
        self.clock_logits = torch.nn.Parameter(torch.zeros(bins))
        self.uniform_mix = uniform_mix

    def bounds(self):
        return self.gamma_min, self.gamma_min + F.softplus(self.width_raw)

    def endpoint_parameters(self):
        return [self.gamma_min, self.width_raw]

    def probabilities(self):
        return (1-self.uniform_mix)*self.clock_logits.softmax(0) + self.uniform_mix/self.clock_logits.numel()

    def inverse_cdf(self, t):
        p = self.probabilities()
        cdf = p.cumsum(0)
        index = torch.searchsorted(cdf.detach(), t.to(p).contiguous()).clamp_max(p.numel()-1)
        left = torch.cat((p.new_zeros(1), cdf[:-1]))[index]
        u = (index + (t.to(p)-left)/p[index])/p.numel()
        return u.clamp(0, 1), 1/(p.numel()*p[index])

    def gamma_at(self, u):
        lower, upper = self.bounds()
        return lower + (upper-lower)*u

    def forward(self, t):
        u, derivative = self.inverse_cdf(t)
        lower, upper = self.bounds()
        return self.gamma_at(u), (upper-lower)*derivative

    def draw(self, count):
        p = self.probabilities()
        # One random point in each equal-probability stratum. Permute so
        # every data row has the correct marginal time distribution.
        t = (torch.arange(count, device=p.device, dtype=p.dtype)
             + torch.rand(count, device=p.device, dtype=p.dtype))/count
        t = t[torch.randperm(count, device=p.device)]
        with torch.no_grad():
            u, _ = self.inverse_cdf(t)
        index = (u*p.numel()).long().clamp_max(p.numel()-1)
        return u, p.numel()*p[index]

    @staticmethod
    def variance_surrogate(estimate, density):
        # d int E[A^2]/p du = -E_p[(A/p)^2 d log p]. No
        # pathwise derivative through categorical masks or sampled u is needed.
        score = -estimate.detach().square()*density.log()
        return score-score.detach()


@algo_registry.register("replaid_hybrid")
class RePlaidHybrid(RePlaid):
    default_sampler = "replaid_hybrid_ddpm"
    supported_samplers = frozenset({"replaid_hybrid_ddpm", "replaid_hybrid_entropy_bounded"})
    sampling_family = "replaid_hybrid"
    sampling_capabilities = frozenset({"replaid_prior", "hybrid_denoise", "hybrid_clock", "embedding_projection"})

    def __init__(self, config, tokenizer):
        super().__init__(config, tokenizer)
        if self.self_condition_rate != 0 or config.algo.T != 0:
            raise ValueError("hybrid ELBO requires self_condition_rate=0 and T=0")
        self.gamma_schedule = HybridClock(config.algo.gamma_min, config.algo.gamma_max,
                                          config.algo.clock_bins, config.algo.clock_uniform_mix).double()
        if self.ema is not None:
            self.ema = ExponentialMovingAverage(self._get_parameters(), decay=config.training.ema)

    def nll(self, x0, output_tokens, current_accumulation_step=None, train_mode=False, valid_tokens=None):
        del output_tokens, current_accumulation_step
        batch, length = x0.shape
        valid = torch.ones_like(x0, dtype=torch.bool) if valid_tokens is None else valid_tokens.bool()
        if valid_tokens is not None and not torch.all((valid_tokens == 0) | (valid_tokens == 1)):
            raise ValueError("hybrid conditional estimator requires binary valid_tokens")
        counts = valid.sum(-1)
        # Uniformly choose a loss position, force it masked, and analytically
        # cancel P(M_i=1)=u. Other positions retain their true Bernoulli law.
        selection_weights = valid.float() + (counts == 0)[:, None].float()
        target = torch.multinomial(selection_weights, 1).squeeze(-1)
        rows = torch.arange(batch, device=x0.device)
        u, density = self.gamma_schedule.draw(batch)
        gamma = self.gamma_schedule.gamma_at(u)
        clean = self.backbone.embed_tokens(x0)
        masked = (torch.rand(batch, length, device=x0.device) < u[:, None]) & valid
        masked[rows, target] = counts > 0
        context = torch.where(masked[..., None], torch.zeros_like(clean), clean)
        z = torch.where(masked[..., None], self.q_sample(clean, gamma), clean)
        log_probs = self.denoise(z, gamma.float(), context)
        predicted = self.backbone.embed_tokens(log_probs.exp())
        ce = -log_probs.gather(-1, x0[..., None]).squeeze(-1).double()
        mse = (predicted-clean).double().square().sum(-1)
        lower, upper = self.gamma_schedule.bounds()
        continuous = (0.5*u*(upper-lower)*torch.exp(-gamma))[:, None]*mse
        # Given the observed mask, the forced target is uniform among its K
        # masked positions. Average over that target instead of retaining one
        # sample (Rao-Blackwellization). Proposal/original mask density=K/(N*u).
        masked_count = masked.sum(-1).clamp_min(1)
        reveal_estimate = (ce*masked).sum(-1)/masked_count/density.detach()
        continuous_estimate = (continuous*masked).sum(-1)/masked_count/density.detach()
        estimate = reveal_estimate+continuous_estimate
        interior = ((ce+continuous)*masked
                    * (counts/masked_count/density.detach())[:, None])
        if train_mode:
            surrogate = self.gamma_schedule.variance_surrogate(estimate, density)
            interior = interior+surrogate[:, None]*valid
        alpha_sq, sigma_sq = torch.sigmoid(-upper), torch.sigmoid(upper)
        prior = 0.5*(alpha_sq*clean.double().square().sum(-1)
                     + self.config.algo.embedding_dim*(sigma_sq-1-F.logsigmoid(upper)))
        losses = (interior+prior)*valid
        if train_mode:
            active = counts > 0
            denominator = active.sum().clamp_min(1)
            def average(value):
                return (value.detach()*active).sum()/denominator
            self._last_hybrid_metrics = {
                "hybrid/reveal": average(reveal_estimate),
                "hybrid/continuous": average(continuous_estimate),
                "hybrid/prior": (prior.detach()*valid).sum()/counts.sum().clamp_min(1),
                "hybrid/gamma_min": lower.detach(), "hybrid/gamma_max": upper.detach(),
                "hybrid/clock_min_probability": self.gamma_schedule.probabilities().detach().min(),
            }
        self._record_time_loss(u.float(), losses, valid, train_mode)
        return losses

    def training_step(self, batch, batch_idx):
        loss = super().training_step(batch, batch_idx)
        self.log_dict(self._last_hybrid_metrics, on_step=True, on_epoch=False)
        return loss

    def generate_samples(self, num_samples, num_steps=None, eps=None, **kwargs):
        # The generic API defaults to a nonzero epsilon; this process must
        # reach its all-visible endpoint, including in the Sudoku callback.
        return super().generate_samples(num_samples, num_steps, 0.0 if eps is None else eps, **kwargs)

    def sampling_mask_probability(self, t):
        return self.gamma_schedule.inverse_cdf(t)[0]

    @torch.no_grad()
    def sample_hybrid_denoise(self, z, t, tokens, kept):
        gamma = self.sampling_gamma(t)
        clean = self.sampling_embeddings(tokens)
        context = torch.where(kept[..., None], clean, torch.zeros_like(clean))
        log_probs = self.denoise(z, gamma.float(), context)
        return log_probs, self.sampling_embeddings(log_probs.exp()), gamma

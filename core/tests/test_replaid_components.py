"""Independent properties of the shared components the REPLAiD presets compose."""

import pytest
import torch
from hydra.utils import instantiate
from torch.nn import functional as F

from core.modifiers.hybrid.views import DiffusionView
from core.noise_schedules.gaussian_path import GaussianPath, path_from_gamma
from core.noise_schedules.learned_log_snr import LinearLogSNRSchedule, PlaidGammaSchedule
from core.objectives.gaussian_elbo import gaussian_diffusion_weight, gaussian_prior_kl
from core.processes.absorbing import AbsorbingProcess
from core.processes.forced_mask import ForcedTargetMask
from core.time_sampling.adaptive_allocation import TimeAllocation
from core.time_sampling.importance import PiecewiseImportanceTimeSampler
from core.training.legacy_checkpoints import rename_replaid_key


def test_importance_density_is_normalized_and_quantile_map_is_detached():
    sampler = PiecewiseImportanceTimeSampler(bins=5, uniform_mix=0.2)
    with torch.no_grad():
        sampler.clock_logits.copy_(torch.tensor([2.0, -1.0, 0.0, 1.0, -3.0]))
    grid = (torch.arange(50_000, dtype=torch.float64) + 0.5) / 50_000
    torch.testing.assert_close(sampler.proposal_density(grid).mean(), torch.tensor(1.0, dtype=torch.float64))
    mapped = sampler(torch.rand(16, dtype=torch.float64))
    assert not mapped.requires_grad
    # The map is monotone and pushes uniform draws to the proposal density.
    t = torch.linspace(0, 1, 101, dtype=torch.float64)
    assert torch.all(sampler(t).diff() >= 0)
    value, derivative = sampler.inverse_cdf(t[1:-1])
    torch.testing.assert_close(derivative, 1 / sampler.proposal_density(value))
    clone = PiecewiseImportanceTimeSampler(bins=5, uniform_mix=0.2)
    clone.load_state_dict(sampler.state_dict())
    assert torch.equal(clone(t), sampler(t))


def test_stratified_uniform_draws_one_time_per_stratum():
    time = PiecewiseImportanceTimeSampler().uniform.sample(32, "cpu")
    assert time.dtype == torch.float64
    assert torch.equal((time * 32).floor().sort().values, torch.arange(32.0, dtype=torch.float64))


def test_linear_log_snr_schedule_endpoints_derivative_and_vp_identity():
    schedule = LinearLogSNRSchedule(-7.0, 3.0)
    time = torch.linspace(0, 1, 9, dtype=torch.float64, requires_grad=True)
    gamma, derivative, _ = schedule.gamma(time)
    torch.testing.assert_close(gamma[[0, -1]], torch.tensor([-7.0, 3.0], dtype=torch.float64))
    torch.testing.assert_close(torch.autograd.grad(gamma.sum(), time)[0], derivative)
    path = schedule(time.detach())
    torch.testing.assert_close(path.alpha.square() + path.sigma.square(), torch.ones(9, dtype=torch.float64))
    assert {name for name, _ in schedule.named_parameters()} == {"gamma_min", "gamma_delta_raw"}


def test_diffusion_weight_is_minus_half_snr_derivative():
    # VP path from gamma: -1/2 dSNR/dt = 1/2 gamma' exp(-gamma).
    gamma = torch.tensor([-5.0, 0.0, 4.0], dtype=torch.float64)
    speed = torch.tensor([2.0, 3.0, 0.5], dtype=torch.float64)
    torch.testing.assert_close(
        gaussian_diffusion_weight(path_from_gamma(gamma, speed)), 0.5 * speed * (-gamma).exp()
    )
    # A non-VP path: compare with autograd of alpha^2 / sigma^2.
    time = torch.tensor([0.2, 0.5, 0.8], dtype=torch.float64, requires_grad=True)
    alpha, sigma = 1 - 0.9 * time, 0.1 + time.square()
    dalpha, dsigma = torch.full_like(time, -0.9), 2 * time
    snr_derivative = torch.autograd.grad((alpha.square() / sigma.square()).sum(), time)[0]
    path = GaussianPath(alpha.detach(), sigma.detach(), dalpha.detach(), dsigma.detach())
    torch.testing.assert_close(gaussian_diffusion_weight(path), -0.5 * snr_derivative)


def test_prior_kl_matches_gaussian_kl():
    schedule = LinearLogSNRSchedule(-2.0, 1.5)
    clean = F.normalize(torch.randn(2, 3, 4, dtype=torch.float64), dim=-1)
    terminal = schedule(torch.ones(1, dtype=torch.float64))
    q = torch.distributions.Normal(terminal.alpha * clean, terminal.sigma.expand_as(clean))
    p = torch.distributions.Normal(torch.zeros_like(clean), torch.ones_like(clean))
    torch.testing.assert_close(
        gaussian_prior_kl(schedule, clean), torch.distributions.kl_divergence(q, p).sum(-1)
    )


def test_forced_target_proposal_marginals_and_support():
    process = AbsorbingProcess(9, proposal=ForcedTargetMask())
    clean = torch.zeros(40_000, 4, dtype=torch.long)
    eligible = torch.tensor([False, True, True, True]).expand_as(clean)
    torch.manual_seed(0)
    state = process.corrupt(clean, torch.full((len(clean),), 0.7), eligible)
    masked = state == 9
    assert not masked[:, 0].any()
    assert masked[:, 1:].any(-1).all()
    # P(position masked) = rho + (1 - rho) / n with rho = 1 - keep and n eligible.
    rho, n = 0.3, 3
    torch.testing.assert_close(
        masked[:, 1:].float().mean(0), torch.full((3,), rho + (1 - rho) / n), atol=0.01, rtol=0
    )
    empty = process.corrupt(clean[:2], torch.full((2,), 0.7), torch.zeros_like(clean[:2], dtype=torch.bool))
    assert torch.equal(empty, clean[:2])


def test_variance_adaptation_weights_follow_observed_term():
    schedule = PlaidGammaSchedule(-13.3, 5.0, 8, adaptation_term="diffusion")
    baseline = PlaidGammaSchedule(-13.3, 5.0, 8)
    baseline.load_state_dict(schedule.state_dict())
    time = torch.tensor([0.0, 0.4, 0.9], dtype=torch.float64)
    path = schedule.training_coefficients(time)
    reference = baseline(time)
    term = torch.tensor([[1.0, 3.0], [2.0, 2.0], [0.5, 1.5]], dtype=torch.float64)
    mask = torch.tensor([[1, 1], [1, 0], [1, 1]])
    schedule.observe_terms({"diffusion": term}, draw=TimeAllocation(time, 1), loss_mask=mask)
    path.sigma.sum().backward()
    reference.sigma.sum().backward()
    # Shape gradients scale by 2 * masked row mean: rows (0 recon, 4, 2).
    weights = torch.tensor([0.0, 4.0, 2.0], dtype=torch.float64)
    expected = PlaidGammaSchedule(-13.3, 5.0, 8)
    expected.load_state_dict(schedule.state_dict())
    gamma, derivative, _ = expected.gamma(time)
    torch.autograd.backward(path_from_gamma(gamma, derivative).sigma, weights)
    shape = {"input_weight", "bias", "output_weight"}
    for name, parameter in schedule.named_parameters():
        if name in shape:
            reweighted = dict(expected.named_parameters())[name].grad
            torch.testing.assert_close(parameter.grad, reweighted)
        else:  # endpoints keep their ordinary gradient
            torch.testing.assert_close(parameter.grad, dict(baseline.named_parameters())[name].grad)
    with pytest.raises(ValueError, match="objective term"):
        schedule.training_coefficients(time)
        schedule.observe_terms({}, draw=None)


def test_revealed_context_reads_only_revealed_clean_tokens():
    from functools import partial

    from core.models.modular import LinearInput
    from core.models.revealed_context import RevealedContextInput
    from core.processes.gaussian import GaussianProcess
    from core.representations.embedding import EmbeddingRepresentation
    from core.representations.tokens import TokenRepresentation
    from core.noise_schedules.log_linear import LogLinear
    from core.samplers.hybrid_updates import DiscretePosteriorUpdate, GaussianPosteriorUpdate

    views = {
        "content": DiffusionView(GaussianProcess(), LinearLogSNRSchedule(),
                                 EmbeddingRepresentation(6, 4), GaussianPosteriorUpdate()),
        "revealed": DiffusionView(AbsorbingProcess(6), LogLinear(0.0), TokenRepresentation(6),
                                  DiscretePosteriorUpdate()),
    }
    slot = RevealedContextInput(views, partial(LinearInput, bias=False), hidden_size=8, context_scale="unit_embedding")
    assert slot.context_scale == 2.0
    with pytest.raises(ValueError, match="state_dim"):
        RevealedContextInput(views, partial(LinearInput), hidden_size=8, state_dim=5)
    from core.modifiers.hybrid.views import point

    t = torch.full((1,), 0.5, dtype=torch.float64)
    points = (point(views["content"], t), point(views["revealed"], t))
    state = torch.randn(1, 3, 4)
    tokens = torch.tensor([[1, 6, 2]])  # position 1 is masked
    base = slot((state, tokens), points)
    continuous = slot.continuous(state, points[0])
    # Masked positions see only the noisy state; revealed ones add their token.
    torch.testing.assert_close(base[:, 1], continuous[:, 1])
    clean = views["content"].representation.encode(tokens[:, [0, 2]]) * 2.0
    torch.testing.assert_close(base[:, [0, 2]], continuous[:, [0, 2]] + slot.context_projection(clean))


def test_denoiser_only_rows_do_not_train_schedule_shape():
    from core.sweeps.configuration import compose
    from core.tests.networks import TINY32

    cfg = compose("replaid_sudoku_hard", [*TINY32, "self_conditioning.probability=1.0"])
    algorithm = instantiate(cfg.algorithm)
    tokens = torch.randint(0, 12, (6, 10))
    mask = torch.ones_like(tokens, dtype=torch.bool)
    algorithm.loss(tokens, mask, training=True)["loss"].backward()
    schedule = dict(algorithm.schedule.named_parameters())
    for name in ("input_weight", "bias", "output_weight"):
        assert schedule[name].grad is None or schedule[name].grad.abs().sum() == 0, name
    # The terminal prior still trains the endpoints; the network trains.
    assert schedule["gamma_max"].grad.abs() > 0
    assert algorithm.model.output_layer.linear.weight.grad.abs().sum() > 0


def test_clean_fixed_feedback_in_training_bootstrap():
    from core.sweeps.configuration import compose
    from core.tests.networks import TINY32

    cfg = compose("replaid_sudoku_hard", [*TINY32, "self_conditioning.probability=1.0"])
    algorithm = instantiate(cfg.algorithm)
    tokens = torch.randint(0, 12, (4, 10))
    mask = torch.ones_like(tokens, dtype=torch.bool)
    mask[:, :3] = False
    seen = []
    handle = algorithm.model.register_forward_hook(
        lambda model, args, kwargs, output: seen.append(kwargs.get("self_condition")),
        with_kwargs=True,
    )
    algorithm.loss(tokens, mask, training=True)
    handle.remove()
    clean = algorithm.representation.encode(tokens)
    torch.testing.assert_close(seen[-1][:, :3], clean[:, :3])


def test_legacy_hybrid_key_names():
    old = "algorithm.model.input.views.1.schedule.clock.width_raw"
    assert rename_replaid_key(old) == "algorithm.model.input.views.0.schedule.gamma_delta_raw"
    assert rename_replaid_key("model.input.views.0.schedule.clock.clock_logits") == "time_sampler.clock_logits"
    assert rename_replaid_key("own_views.0.schedule.bias") == "own_views.0.schedule.bias"

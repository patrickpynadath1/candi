"""Reference equations, gradient ownership and core lifecycle for RePlaid."""

from pathlib import Path

from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
import pytest
import torch
from torch.nn import functional as F

from core.noise_schedules.learned_log_snr import LearnedLogSNRSchedule
from core.objectives.replaid import RePlaidObjective
from core.parameterizations.gaussian_prior import GaussianPriorPosterior
from core.processes.gaussian import GaussianProcess
from core.modifiers.self_conditioning.gradient_routing import detach_selected, embedding_mean
from core.modifiers.self_conditioning.prediction import PredictionCoordinator
from core.samplers.gaussian_ddpm import GaussianDDPMSampler
from core.time_sampling.adaptive_allocation import AdaptiveTimeAllocation
from core.training.module import TrainingModule

ROOT = Path(__file__).resolve().parents[2]


def config(*overrides):
    with initialize_config_dir(
        version_base=None, config_dir=str(ROOT / "configs")
    ):
        return compose(
            config_name="config",
            overrides=[
                "+experiment=replaid_sudoku_hard",
                "model.hidden_size=32",
                "model.cond_dim=16",
                "model.n_blocks=1",
                "model.n_heads=4",
                "model.dropout=0",
                "training.warmup_steps=0",
                "output_dir=/tmp/replaid_test",
                "data.cache_dir=/tmp/replaid_test_data",
                *overrides,
            ],
        )


def test_variance_hook_changes_only_shape_gradients():
    schedule, baseline = LearnedLogSNRSchedule(), LearnedLogSNRSchedule()
    time = torch.tensor([0.0, 0.3, 0.8])
    gamma, derivative, adaptation = schedule.gamma(time, 1)
    adaptation.weight = torch.tensor([0.0, 3.0, 3.0], dtype=torch.float64)
    reference_gamma, reference_derivative, _ = baseline.gamma(time)
    (gamma[1:].sum() + derivative[1:].sum()).backward()
    (reference_gamma[1:].sum() + reference_derivative[1:].sum()).backward()
    for name, parameter in schedule.named_parameters():
        multiplier = 3 if name in {"slope_raw", "centers"} else 1
        torch.testing.assert_close(
            parameter.grad, dict(baseline.named_parameters())[name].grad * multiplier
        )


def test_objective_hand_computed_masked_reduction():
    tokens = torch.tensor([[0, 1], [1, 0]])
    probabilities = torch.tensor([[[0.8, 0.2], [0.3, 0.7]], [[0.4, 0.6], [0.9, 0.1]]])
    clean = F.one_hot(tokens, 2).float()
    gamma = torch.tensor([-2.0, 1.0], dtype=torch.float64)
    derivative = torch.tensor([3.0, 4.0], dtype=torch.float64)
    terminal = torch.tensor([5.0], dtype=torch.float64)
    terms = RePlaidObjective()(
        probabilities.log(),
        tokens,
        clean,
        probabilities,
        gamma,
        derivative,
        terminal,
        clean,
        1,
    )
    ce = -probabilities.log().gather(-1, tokens[..., None]).squeeze(-1).double()
    diffusion = (
        0.5
        * derivative[:, None]
        * (-gamma).exp()[:, None]
        * (probabilities - clean).double().square().sum(-1)
    )
    alpha_sq = torch.sigmoid(-terminal[0])
    sigma_sq = 1 - alpha_sq
    prior = 0.5 * (alpha_sq + 2 * (sigma_sq - 1 - sigma_sq.log()))
    expected = prior + torch.stack([2 * ce[0], 2 * diffusion[1]])
    torch.testing.assert_close(terms.total, expected)
    mask = torch.tensor([[False, True], [True, False]])
    torch.testing.assert_close(
        (terms.total * mask).sum() / mask.sum(), (expected[0, 1] + expected[1, 0]) / 2
    )
    assert terms.total.dtype == torch.float64


def test_selected_rows_detach_embedding_and_schedule_but_not_denoiser():
    selected = torch.tensor([True, True])
    schedule = LearnedLogSNRSchedule()
    table = F.normalize(torch.randn(4, 3), dim=-1).requires_grad_()
    tokens = torch.tensor([[0, 1], [2, 3]])
    gamma, derivative, _ = schedule.gamma(torch.tensor([0.3, 0.7]))
    effective_gamma, effective_derivative = (
        detach_selected(gamma, selected),
        detach_selected(derivative, selected),
    )
    clean = detach_selected(F.embedding(tokens, table), selected)
    path = schedule.coefficients(effective_gamma, effective_derivative)
    state = path.alpha.float()[:, None, None] * clean + path.sigma.float()[
        :, None, None
    ] * torch.randn_like(clean)
    logits = torch.randn(2, 2, 4, requires_grad=True)
    predictor = GaussianPriorPosterior()
    log_probs = predictor.corrected_logits(
        logits, state, path.alpha, path.sigma, table, selected
    ).log_softmax(-1)
    predicted = embedding_mean(log_probs.exp(), table, selected)
    loss = (predicted - clean).double().square().sum() + log_probs.square().sum()
    loss.backward()
    assert logits.grad.abs().sum() > 0
    assert table.grad.abs().sum() == 0
    assert all(p.grad.abs().sum() == 0 for p in schedule.parameters())
    # Terminal prior is intentionally a separate ordinary-gradient path.
    schedule.zero_grad()
    table.grad = None
    terminal, _, _ = schedule.gamma(torch.ones(1))
    (0.5 * torch.sigmoid(-terminal[0]) * table.double().square().sum()).backward()
    assert table.grad.abs().sum() > 0
    assert schedule.gamma_min.grad.abs() > 0


def test_prior_logits_match_explicit_gaussian_density():
    predictor = GaussianPriorPosterior()
    table = F.normalize(torch.randn(5, 3), dim=-1)
    state = torch.randn(2, 4, 3)
    alpha, sigma = torch.tensor([0.3, 0.8]), torch.tensor([0.9, 0.4])
    logits = torch.randn(2, 4, 5)
    corrected = predictor.corrected_logits(logits, state, alpha, sigma, table)
    density = -(state[:, :, None] - alpha[:, None, None, None] * table).square().sum(
        -1
    ) / (2 * sigma[:, None, None].square())
    torch.testing.assert_close(
        corrected.log_softmax(-1),
        (logits + density).log_softmax(-1),
        atol=2e-6,
        rtol=2e-6,
    )


def test_reverse_posterior_reference_and_zero_gap():
    state, clean, noise = (torch.randn(3, 4, 2) for _ in range(3))
    schedule = LearnedLogSNRSchedule()
    time, previous = torch.tensor([0.3, 0.6, 0.9]), torch.tensor([0.2, 0.4, 0.8])
    gamma_t, _, _ = schedule.gamma(time)
    gamma_s, _, _ = schedule.gamma(previous)
    alpha_t = torch.sigmoid(-gamma_t).sqrt().to(state)[:, None, None]
    alpha_s = torch.sigmoid(-gamma_s).sqrt().to(state)[:, None, None]
    c = (1 - torch.exp(gamma_s - gamma_t)).to(state)[:, None, None]
    expected = (
        alpha_s / alpha_t * (1 - c) * state
        + c * alpha_s * clean
        + (c * (1 - alpha_s.square())).clamp_min(0).sqrt() * noise
    )
    process = GaussianProcess()
    actual = process.reverse_posterior(
        state, clean, schedule(time), schedule(previous), noise
    )
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=1e-5)
    torch.testing.assert_close(
        process.reverse_posterior(state, clean, schedule(time), schedule(time), noise),
        state,
    )


def test_fixed_count_and_allocation_state():
    policy = PredictionCoordinator(
        enabled=True, probability=0.25, selection="fixed_count"
    )
    assert policy.select(7, "cpu", True).sum() == 2
    assert policy.select(7, "cpu", False).all()
    allocation = AdaptiveTimeAllocation(momentum=0.5)
    sample = allocation.sample(8, "cpu")
    assert sample.reconstruction_count == 4
    assert sample.time[:4].eq(0).all()
    assert ((sample.time[4:] >= 0) & (sample.time[4:] < 1)).all()
    allocation.observe(torch.arange(8.0)[:, None], 2 * torch.arange(8.0)[:, None], 4)
    clone = AdaptiveTimeAllocation(momentum=0.5)
    clone.load_state_dict(allocation.state_dict())
    assert clone.recon_std_ema == allocation.recon_std_ema
    assert clone.sample(1, "cpu").reconstruction_count == 0


@pytest.mark.parametrize("source", ["posterior_mean", "hidden"])
def test_complete_loss_bootstrap_sampler_and_optimizer(source):
    module = TrainingModule(config(f"self_conditioning.source={source}"))
    algorithm = module.algorithm
    with torch.no_grad():
        algorithm.model.output_layer.linear.weight.normal_(std=0.1)
    tokens = torch.randint(0, 12, (8, 10))
    mask = torch.ones_like(tokens, dtype=torch.bool)
    mask[:, :3] = False
    seen = []
    handle = algorithm.model.register_forward_hook(
        lambda model, args, kwargs, output: seen.append(
            (len(args[0]), torch.is_grad_enabled(), kwargs.get("self_condition"))
        ),
        with_kwargs=True,
    )
    result = algorithm.loss(tokens, mask, training=True)
    result["loss"].backward()
    assert [item[:2] for item in seen] == [(2, False), (8, True)]
    assert seen[-1][2].requires_grad is False
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in algorithm.parameters()
    )
    optimizer = module.configure_optimizers()["optimizer"]
    assert [group["lr"] for group in optimizer.param_groups] == [0.0003, 0.01, 0.01]
    optimizer.step()
    algorithm.representation.post_optimizer_step()
    torch.testing.assert_close(
        algorithm.representation.table.norm(dim=-1), torch.ones(12)
    )
    seen.clear()
    algorithm.eval()
    generated, stats = algorithm.generate(GaussianDDPMSampler(3), tokens, ~mask)
    assert len(seen) == stats["nfe"] == 4
    assert generated[~mask].equal(tokens[~mask])
    assert seen[0][2] is None
    seen.clear()
    algorithm.generate(GaussianDDPMSampler(1), tokens, ~mask)
    assert seen[0][2] is None
    handle.remove()


def test_resume_model_allocation_ema_and_optimizer(tmp_path):
    config_ = config()
    module = TrainingModule(config_)
    tokens = torch.randint(0, 12, (8, 10))
    mask = torch.ones_like(tokens, dtype=torch.bool)
    optimizer = module.configure_optimizers()["optimizer"]
    module.algorithm.loss(tokens, mask, training=True)["loss"].backward()
    optimizer.step()
    module.algorithm.representation.post_optimizer_step()
    module.ema.update(module.algorithm.parameters())
    checkpoint = {
        "state_dict": module.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    module.on_save_checkpoint(checkpoint)
    path = tmp_path / "checkpoint.pt"
    torch.save(checkpoint, path)
    loaded = torch.load(path, weights_only=False)
    resumed = TrainingModule(config_)
    resumed.load_state_dict(loaded["state_dict"])
    resumed.on_load_checkpoint(loaded)
    resumed_optimizer = resumed.configure_optimizers()["optimizer"]
    resumed_optimizer.load_state_dict(loaded["optimizer"])
    torch.manual_seed(27)
    left = module.algorithm.loss(tokens, mask, training=True)["loss"]
    torch.manual_seed(27)
    right = resumed.algorithm.loss(tokens, mask, training=True)["loss"]
    torch.testing.assert_close(left, right, rtol=0, atol=0)
    for a, b in zip(module.ema.shadow_params, resumed.ema.shadow_params):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_bf16_loss_and_generation():
    algorithm = instantiate(config().algorithm).cuda()
    tokens = torch.randint(0, 12, (8, 10), device="cuda")
    mask = torch.ones_like(tokens, dtype=torch.bool)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        result = algorithm.loss(tokens, mask, training=True)
        result["loss"].backward()
        generated, _ = algorithm.generate(GaussianDDPMSampler(3), tokens, ~mask)
    assert torch.isfinite(result["loss"])
    assert all(
        p.grad is None or torch.isfinite(p.grad).all() for p in algorithm.parameters()
    )
    assert generated.shape == tokens.shape


def test_optimizer_lifecycle_ema_precedes_representation_projection(monkeypatch):
    import lightning as L

    module = TrainingModule(config())
    table = module.algorithm.representation.table
    expected = table.detach().clone() * 2
    monkeypatch.setattr(
        L.LightningModule,
        "optimizer_step",
        lambda *args, **kwargs: table.data.copy_(expected),
    )
    module.optimizer_step()
    torch.testing.assert_close(table.norm(dim=-1), torch.ones(len(table)))
    index = next(
        index
        for index, parameter in enumerate(module.algorithm.parameters())
        if parameter is table
    )
    # First EMA update uses min(.9999, 2/11), before the table is projected.
    original = expected / 2
    torch.testing.assert_close(
        module.ema.shadow_params[index], (2 / 11) * original + (9 / 11) * expected
    )


def test_schedule_derivative_endpoints_and_vp_identity():
    schedule = LearnedLogSNRSchedule()
    time = torch.linspace(0, 1, 13, dtype=torch.float64, requires_grad=True)
    gamma, derivative, _ = schedule.gamma(time)
    torch.testing.assert_close(
        gamma[[0, -1]], torch.tensor([-13.3, 5.0], dtype=torch.float64)
    )
    torch.testing.assert_close(torch.autograd.grad(gamma.sum(), time)[0], derivative)
    path = schedule(time)
    torch.testing.assert_close(
        path.alpha.square() + path.sigma.square(), torch.ones_like(time)
    )
    assert (derivative > 0).all()


def test_public_training_prediction_subsets_prior_context():
    algorithm = instantiate(config().algorithm)
    state = torch.randn(8, 5, 16)
    time = torch.linspace(0.1, 0.9, 8)
    prediction = algorithm.training_prediction(state, time, algorithm.schedule(time))
    assert prediction.shape == (8, 5, 12)
    assert torch.isfinite(prediction).all()
    torch.testing.assert_close(prediction.exp().sum(-1), torch.ones(8, 5))


def test_corrected_tanh_schedule_derivative_and_gradient_routing():
    import copy
    from core.noise_schedules.learned_log_snr import PlaidGammaSchedule

    schedule = PlaidGammaSchedule(-13.3, 5.0, 16)
    baseline = copy.deepcopy(schedule)
    time = torch.linspace(0, 1, 7, dtype=torch.float64, requires_grad=True)
    gamma, derivative, _ = baseline.gamma(time)
    torch.testing.assert_close(
        torch.autograd.grad(gamma.sum(), time, retain_graph=True)[0], derivative
    )
    torch.testing.assert_close(gamma[[0, -1]], torch.stack(baseline.bounds()))
    assert (derivative > 0).all()
    adapted, speed, feedback = schedule.gamma(time.detach(), 1)
    feedback.weight = torch.full_like(adapted, 3)
    (adapted[1:].sum() + speed[1:].sum()).backward()
    (gamma[1:].sum() + derivative[1:].sum()).backward()
    for name, parameter in schedule.named_parameters():
        scale = 1 if name.startswith("gamma_") else 3
        torch.testing.assert_close(
            parameter.grad, dict(baseline.named_parameters())[name].grad * scale
        )


def test_singleton_allocation_moments_and_masked_adaptation():
    from core.noise_schedules.learned_log_snr import ScheduleGradient

    allocation = AdaptiveTimeAllocation(momentum=0.5)
    mask = torch.tensor([[0, 1], [0, 1]])
    for value in [1.0, 5.0, 1.0, 5.0]:
        recon = torch.tensor([[1e6, value], [1e6, 0.0]])
        diff = torch.tensor([[1e6, 0.0], [1e6, 2.0]])
        allocation.observe(recon, diff, 1, mask)
    assert allocation.recon_std_ema > 1
    assert allocation.recon_mean_ema < 5  # Masked prefix cannot contaminate moments.
    gradient = ScheduleGradient(torch.ones(2))
    gradient.observe(diff, 1, torch.zeros(2, dtype=torch.bool), mask)
    torch.testing.assert_close(
        gradient.weight, torch.tensor([0.0, 4.0], dtype=torch.float64)
    )


def test_corrected_sampler_bootstraps_same_state_clamps_prompts_and_noises_final_step():
    cfg = config("sampler.steps=2")
    algo = instantiate(cfg.algorithm).eval()
    sampler = instantiate(cfg.sampler)
    tokens = torch.randint(0, 12, (3, 7))
    prompt = torch.zeros_like(tokens, dtype=torch.bool)
    prompt[:, :2] = True
    clean = algo.representation.encode(tokens)
    calls, noises = [], []

    def record(model, args, kwargs, output):
        calls.append((args[0].clone(), kwargs.get("self_condition")))

    handle = algo.model.register_forward_hook(record, with_kwargs=True)
    original = algo.process.reverse_posterior

    def transition(state, mean, current, previous, noise):
        noises.append(noise.clone())
        return original(state, mean, current, previous, noise)

    algo.process.reverse_posterior = transition
    generated, metadata = algo.generate(sampler, tokens, prompt)
    handle.remove()
    assert metadata["nfe"] == len(calls) == 6
    assert noises[-1].abs().sum() > 0
    assert generated[prompt].equal(tokens[prompt])
    for index in range(0, len(calls), 2):
        assert calls[index][1] is None
        torch.testing.assert_close(calls[index][0], calls[index + 1][0])
        torch.testing.assert_close(calls[index][0][prompt], clean[prompt])
        torch.testing.assert_close(calls[index + 1][1][prompt], clean[prompt])


def test_singleton_loss_contains_both_estimators():
    algo = instantiate(config().algorithm)
    tokens = torch.ones(1, 7, dtype=torch.long)
    mask = torch.ones_like(tokens, dtype=torch.bool)
    result = algo.loss(tokens, mask, training=True)
    result["loss"].backward()
    assert torch.isfinite(result["loss"])
    assert result["num_tokens"] == 7
    assert algo.model.output_layer.linear.weight.grad is not None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_bootstrap_autocast_does_not_freeze_transformer():
    algo = instantiate(config("self_conditioning.probability=1").algorithm).cuda()
    optimizer = torch.optim.Adam(algo.parameters(), lr=1e-3)
    head = algo.model.output_layer.linear.weight
    attention = algo.model.blocks[0].attn_qkv.weight
    initial_head, initial_attention = head.detach().clone(), attention.detach().clone()
    tokens = torch.randint(0, 12, (8, 7), device="cuda")
    for _ in range(3):
        optimizer.zero_grad()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = algo.loss(tokens, torch.ones_like(tokens), training=True)["loss"]
        loss.backward()
        assert head.grad is not None and head.grad.abs().sum() > 0
        optimizer.step()
    assert not torch.equal(head, initial_head)
    assert not torch.equal(attention, initial_attention)

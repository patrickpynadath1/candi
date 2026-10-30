import copy
import pytest
import torch
from core.time_sampling.cdcd import CDCDTimeSampler


def sampler(**kw):
    return CDCDTimeSampler(adaptive=True, bins=8, warmup=0, **kw)


def observe(s, t, y, step=0):
    s.observe(t, y[:, None], torch.ones(len(t), 1, dtype=torch.bool), step)


def test_identity_endpoints_and_positive_bins_under_extreme_logits():
    s = sampler(epsilon=0.01)
    u = torch.linspace(0, 1, 1001)
    torch.testing.assert_close(s.inverse_cdf(u), 0.01 + 0.99 * u)
    with torch.no_grad():
        s.ema_logits[0].copy_(torch.linspace(-1000, 1000, 8))
        s.ema_logits[1].copy_(torch.linspace(1000, -1000, 8))
    t = s.inverse_cdf(u)
    assert torch.isfinite(t).all() and (t.diff() > 0).all()
    assert s.density(t).min() > 0
    torch.testing.assert_close(s.cdf(t), u, atol=1e-3, rtol=1e-3)
    assert t[0] == pytest.approx(0.01) and t[-1] == pytest.approx(1.0)
    # Both axes have a floor; there is deliberately no uniform mixture parameter.
    assert (
        s._widths(s.ema_logits) >= s.min_bin_size / (1 + 8 * s.min_bin_size) - 1e-7
    ).all()


def test_sampling_epsilon_bounds_time_not_probability():
    s = sampler(epsilon=0.01)
    s.ready.fill_(True)
    s.ema_logits[1, -1] = 6
    u = torch.tensor([0.0, 1e-5, 0.5, 1.0])
    torch.testing.assert_close(s(0.01 + 0.99 * u), s.inverse_cdf(u))
    assert s(0.01 + 0.99 * u)[0] == pytest.approx(0.01)
    assert not s(torch.tensor([0.2], requires_grad=True)).requires_grad


def test_importance_weights_recover_uniform_fit_objective():
    s = sampler()
    s.ema_logits[1].copy_(torch.linspace(-1, 1, 8))
    u = (torch.arange(20000) + 0.5) / 20000
    t = s.inverse_cdf(u)
    losses = 3 * s._unit_time(t).square()
    logits = s.logits.clone().requires_grad_()
    warped = s.fitting_loss(logits, t, losses, s.density(t)).mean()
    uniform_t = s.uniform.epsilon + (1 - s.uniform.epsilon) * u
    uniform = s.fitting_loss(
        logits,
        uniform_t,
        3 * u.square(),
        torch.full_like(u, 1 / (1 - s.uniform.epsilon)),
    ).mean()
    torch.testing.assert_close(warped, uniform, atol=2e-4, rtol=2e-4)
    g1 = torch.autograd.grad(warped, logits)[0]
    g2 = torch.autograd.grad(uniform, logits)[0]
    torch.testing.assert_close(g1, g2, atol=5e-4, rtol=5e-3)


def test_fit_learns_nonlinear_loss_with_no_holes():
    s = sampler(learning_rate=0.03, momentum=0.9)
    u = (torch.arange(256) + 0.5) / 256
    for step in range(600):
        t = s.inverse_cdf(u)
        observe(s, t, 3 * s._unit_time(t).square(), step)
        s.post_optimizer_step()
    x = torch.linspace(0.01, 0.99, 101)
    assert (s.cdf(x) - s._unit_time(x).square()).abs().mean() < 0.04
    assert s.density(torch.linspace(s.uniform.epsilon, 1, 1001)).min() > 0
    assert not torch.equal(s.logits[0], torch.full_like(s.logits[0], s.logits[0, 0]))
    assert list(s.parameters()) == []


def test_accumulation_checkpoint_and_gradient_isolation():
    full, micro = sampler(), sampler()
    t = torch.linspace(0.01, 0.99, 12, requires_grad=True)
    y = (2 * t.square()).detach().requires_grad_()
    observe(full, t, y)
    full.post_optimizer_step()
    for a, b in zip(t.chunk(3), y.chunk(3)):
        observe(micro, a, b)
    assert micro.updates == 0
    resumed = sampler()
    resumed.load_state_dict(copy.deepcopy(micro.state_dict()))
    micro.post_optimizer_step()
    resumed.post_optimizer_step()
    for k, v in full.state_dict().items():
        torch.testing.assert_close(micro.state_dict()[k], v)
        torch.testing.assert_close(resumed.state_dict()[k], v)
    assert t.grad is None and y.grad is None
    assert full.count == 12
    full.post_optimizer_step()
    assert full.updates == 1


def test_masking_empty_observations_and_warmup():
    s = CDCDTimeSampler(adaptive=True, bins=8, warmup=2)
    t = torch.tensor([0.2, 0.5, 0.8])
    loss = torch.tensor([[1.0, float("nan")], [2.0, 3.0], [float("nan"), 0.0]])
    mask = torch.tensor([[1, 0], [0, 0], [1, 0]], dtype=torch.bool)
    s.observe(t, loss, mask, 0)
    s.post_optimizer_step()
    assert s.count == 1 and not s.ready
    torch.testing.assert_close(s(t), t)
    observe(s, t, torch.ones(3), 1)
    s.post_optimizer_step()
    assert s.ready
    old = copy.deepcopy(s.state_dict())
    s.observe(t, torch.zeros(3, 2), torch.zeros(3, 2, dtype=torch.bool), 2)
    s.post_optimizer_step()
    for k, v in old.items():
        if k != "last_step":
            torch.testing.assert_close(s.state_dict()[k], v)


def test_legacy_state_rejected_without_explicit_migration():
    from core.time_sampling.loss_derivative import LossDerivativeTimeSampler

    with pytest.raises(RuntimeError):
        sampler().load_state_dict(LossDerivativeTimeSampler(adaptive=True).state_dict())


def test_tinygsm_configs_use_cdcd():
    from core.sweeps.configuration import compose
    from hydra.utils import instantiate

    for name in [
        "tinygsm_blurring_sc_adaptive",
        "tinygsm_gaussian_matched_blurring_sc_adaptive",
        "tinygsm_masked_gaussian_sc_adaptive",
        "tinygsm_uniform_gaussian_sc_adaptive",
    ]:
        assert isinstance(instantiate(compose(name).time_sampler), CDCDTimeSampler)


def test_reset_sampler_gets_warmup_even_on_late_model_checkpoint():
    s = CDCDTimeSampler(adaptive=True, bins=8, warmup=2)
    t = torch.tensor([0.1, 0.5, 0.9])
    observe(s, t, t.square(), step=250000)
    s.post_optimizer_step()
    assert not s.ready
    observe(s, t, t.square(), step=250001)
    s.post_optimizer_step()
    assert s.ready


def _distributed_worker(rank, path, output):
    import torch.distributed as dist

    dist.init_process_group(
        "gloo", init_method="file://" + path, rank=rank, world_size=2
    )
    s = sampler()
    t = torch.linspace(0.01, 0.99, 12)
    y = 2 * t.square()
    # Unequal valid counts across ranks, including an empty local microbatch.
    for indices in (
        [slice(0, 2), slice(2, 5)] if rank == 0 else [slice(5, 5), slice(5, 12)]
    ):
        observe(s, t[indices], y[indices])
    s.post_optimizer_step()
    torch.save(s.state_dict(), output + str(rank))
    dist.destroy_process_group()


def test_distributed_fit_matches_single_global_batch(tmp_path):
    import torch.multiprocessing as mp

    mp.spawn(
        _distributed_worker,
        args=(str(tmp_path / "rendezvous"), str(tmp_path / "state")),
        nprocs=2,
        join=True,
    )
    s = sampler()
    t = torch.linspace(0.01, 0.99, 12)
    observe(s, t, 2 * t.square())
    s.post_optimizer_step()
    for rank in range(2):
        state = torch.load(str(tmp_path / "state") + str(rank), weights_only=True)
        for k, v in s.state_dict().items():
            torch.testing.assert_close(state[k], v)


def test_explicit_migration_preserves_model_optimizer_and_ema():
    from core.training.cdcd_migration import migrate_checkpoint
    from core.training.module import TrainingModule
    from core.sweeps.configuration import compose
    from omegaconf import OmegaConf

    cfg = compose("tinygsm_blurring_sc_adaptive")
    cfg.task.vocab_size = 11
    cfg.task.length = 8
    cfg.model.hidden_size = 16
    cfg.model.cond_dim = 8
    cfg.model.n_blocks = 1
    cfg.model.n_heads = 2
    cfg.model.output_dim = 8
    cfg.representation.dimension = 8
    cfg.time_sampler = OmegaConf.load("configs/time_sampler/loss_derivative.yaml")
    cfg.time_sampler.adaptive = True
    old = TrainingModule(cfg)
    original = {
        "state_dict": old.state_dict(),
        "hyper_parameters": {"config": OmegaConf.to_container(cfg, resolve=True)},
        "global_step": 190000,
        "optimizer_states": [{"sentinel": torch.ones(1)}],
        "ema": old.ema.state_dict(),
    }
    settings = OmegaConf.load("configs/time_sampler/cdcd.yaml")
    settings.adaptive = True
    migrated = migrate_checkpoint(original, settings)
    assert migrated["optimizer_states"] is original["optimizer_states"]
    assert migrated["ema"] is original["ema"]
    assert "algorithm.time_sampler.times" in original["state_dict"]
    fresh = TrainingModule(OmegaConf.create(migrated["hyper_parameters"]["config"]))
    fresh.load_state_dict(migrated["state_dict"], strict=True)
    for a, b in zip(old.algorithm.parameters(), fresh.algorithm.parameters()):
        torch.testing.assert_close(a, b)
    assert (
        fresh.algorithm.time_sampler.updates == 0
        and not fresh.algorithm.time_sampler.ready
    )


def test_collapsed_legacy_proposal_still_gets_low_noise_observations_with_cdcd():
    """Regression for the observed flat-CDF + probability-epsilon dead zone."""
    from core.time_sampling.loss_derivative import LossDerivativeTimeSampler

    old = LossDerivativeTimeSampler(adaptive=True, epsilon=0.001)
    old.cdf.copy_(((old.grid - 0.7) / 0.3).clamp(0, 1))
    new = CDCDTimeSampler(adaptive=True, bins=100, epsilon=0.001, warmup=0)
    edges = torch.linspace(0, 1, 101)
    mass = ((edges - 0.7) / 0.3).clamp(0, 1).diff().clamp_min(1e-12)
    new.logits[1].copy_(mass.log())
    new.ema_logits.copy_(new.logits)
    new.ready.fill_(True)
    torch.manual_seed(17)
    old_times = old.sample(8192, "cpu")
    torch.manual_seed(17)
    new_times = new.sample(8192, "cpu")
    assert (old_times < 0.5).sum() == 0
    assert (new_times < 0.5).sum() >= 30
    # Coverage comes from positive learned bins, with no uniform mixture.
    assert new_times.min() >= 0.001 and new_times.max() <= 1

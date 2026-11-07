"""Every corruption family x self-conditioning cell trains and samples at 128 NFE."""

from hydra.utils import instantiate
from omegaconf import OmegaConf
import pytest
import torch

from core.tests.factorial_configs import cells, samplers

CELLS = list(cells())


def tiny(source):
    cfg = OmegaConf.create(OmegaConf.to_container(source, resolve=False))
    cfg.model.backbone.hidden_size = 16
    cfg.model.backbone.cond_dim = 8
    cfg.model.backbone.n_blocks = 1
    cfg.model.backbone.n_heads = 2
    cfg.model.backbone.dropout = 0
    return cfg


def test_factorial_cells_are_balanced_and_use_the_intended_schedules():
    assert len(CELLS) == 12
    for family, enabled, cfg in CELLS:
        assert cfg.self_conditioning.enabled == enabled
        assert cfg.self_conditioning.source == "hidden"
        assert cfg.model.time_conditioning == "identity"
        assert cfg.objective._target_.endswith("CrossEntropy")
        assert cfg.time_sampler._target_.endswith("UniformTimeSampler")
        assert not cfg.block_diffusion.enabled and not cfg.token_feedback.enabled
        assert len(cfg.model.views) == 1
        view = cfg.model.views.content
        if family == "spherical":
            assert view.schedule.auto_truncation is False
        if family == "gaussian":
            assert view.schedule._target_.endswith("CDCDGaussianSchedule")
            assert view.schedule.base.coupling == "linear"
            assert view.representation.state_representation._target_.endswith(
                "OneHotRepresentation"
            )


@pytest.mark.parametrize("index", range(len(CELLS)))
def test_every_cell_backward_and_real_128_nfe_samplers(index):
    _, _, cfg = CELLS[index]
    cfg = tiny(cfg)
    algo = instantiate(cfg.algorithm)
    assert algo.prediction.enabled == cfg.self_conditioning.enabled
    torch.manual_seed(31)
    tokens = torch.randint(0, 12, (2, 6))
    mask = torch.ones_like(tokens, dtype=torch.bool)
    mask[:, :2] = False
    result = algo.loss(tokens, mask, training=True)
    assert torch.isfinite(result["loss"])
    result["loss"].backward()
    assert all(
        torch.isfinite(p.grad).all() for p in algo.parameters() if p.grad is not None
    )
    for group, sampler_config in samplers():
        sampler = instantiate(sampler_config)
        calls = []
        # Observe the actual posterior passed to each native transition.
        observed = []
        original_update = algo.views[0].update

        def capture(*args, **kwargs):
            observed.append(args[3].detach().clone())
            return original_update(*args, **kwargs)

        algo.views[0].update = capture
        handle = algo.model.register_forward_hook(lambda *args: calls.append(1))
        try:
            output, metadata = algo.generate(sampler, tokens, ~mask)
        finally:
            handle.remove()
            algo.views[0].update = original_update
        assert torch.equal(output[:, :2], tokens[:, :2])
        assert output.min() >= 0 and output.max() < 12
        assert metadata["nfe"] == len(calls)
        if group == "hybrid":
            assert len(observed) == 127
            if sampler.greedy_clean_estimate:
                assert all(torch.all((p == 0) | (p == 1)) for p in observed)
                assert all(torch.all(p.sum(-1) == 1) for p in observed)
            assert len(calls) == 128
            assert metadata["nfe_per_sample"] == [128, 128]
        else:
            assert 0 < len(calls) <= 128
            assert all(0 < n <= len(calls) for n in metadata["nfe_per_sample"])


def test_adaptive_gaussian_feedback_and_checkpoint():
    cfg = tiny(next(c for family, _, c in CELLS if family == "gaussian"))
    schedule = cfg.model.views.content.schedule
    schedule.buffer_size = 32
    schedule.warmup_steps = 0
    schedule.refit_every = 1
    algo = instantiate(cfg.algorithm)
    tokens = torch.randint(0, 12, (32, 6))
    algo.loss(tokens, torch.ones_like(tokens, dtype=torch.bool), step=1, training=True)
    clock = algo.views[0].schedule.clock
    assert clock.num_observed.item() == 32
    assert clock.last_refit_step.item() == 1
    assert clock.has_schedule.item()
    restored = instantiate(cfg.algorithm)
    restored.load_state_dict(algo.state_dict())
    times = torch.linspace(0.01, 0.99, 20)
    torch.testing.assert_close(clock(times), restored.views[0].schedule.clock(times))
    _, sampler_config = next(samplers())
    generated, metadata = restored.generate(
        instantiate(sampler_config),
        tokens[:2],
        torch.zeros_like(tokens[:2], dtype=torch.bool),
    )
    assert generated.min() >= 0 and generated.max() < 12
    assert metadata["nfe"] == 128

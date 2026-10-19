"""Loss-adaptive clocks apply to every continuous hybrid view and survive restore."""

import copy

import pytest
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from core.modifiers.hybrid.views import point
from core.sweeps.configuration import ROOT
from core.training.module import TrainingModule
from examples.corruption_self_conditioning_sweep import base_config, parser, variants


@pytest.mark.parametrize("family", ["gaussian", "simplex", "spherical", "blurring"])
def test_adaptive_clock_training_sampling_and_checkpoint(family):
    args = parser().parse_args(["--families", family])
    config = base_config(next(variants(args)), args)
    config.model.backbone.hidden_size = 16
    config.model.backbone.n_blocks = 1
    config.model.backbone.n_heads = 2
    config.model.backbone.cond_dim = 8
    config.training.ema = 0
    view = config.model.views.content
    if family == "gaussian":
        view.schedule = view.schedule.base
    view.time_map = OmegaConf.load(ROOT / "configs/time_map/cdcd.yaml")
    view.time_map.clock.buffer_size = 64
    view.time_map.clock.warmup_steps = 0
    view.time_map.clock.refit_every = 1
    view.time_map.clock.bins = 8
    view.time_map.clock.n_grid = 64
    config.self_conditioning.enabled = True
    algorithm = TrainingModule(config).algorithm
    actual = algorithm.views[0]
    clock = actual.time_map.clock
    time = torch.linspace(0.01, 0.99, 64)
    actual.time_map.observe(time, time.square(), 0, training=True)
    assert clock.refit_count.item() == 1
    assert not torch.allclose(actual.time_map(time), time, atol=0.01)
    assert torch.all(actual.time_map(time).diff() >= 0)
    torch.testing.assert_close(
        actual.time_map(torch.tensor([0.0, 1.0])), torch.tensor([0.0, 1.0])
    )

    tokens = torch.randint(0, 12, (2, 4))
    mask = torch.ones_like(tokens, dtype=torch.bool)
    mask[:, 0] = False
    observed = []
    handle = clock.register_forward_hook(
        lambda module, inputs, output: observed.append(inputs[0].clone())
    )
    result = algorithm.loss(tokens, mask, mask, step=1, training=True)
    handle.remove()
    assert observed and clock.refit_count.item() == 2
    assert torch.isfinite(result["loss"])
    result["loss"].backward()
    # The view clock is applied before every kind of coefficient construction,
    # including blurring's sequence-layout-specific path.
    query = torch.tensor([0.25, 0.75])
    at = point(actual, query, corruption_mask=mask)
    torch.testing.assert_close(at.time, actual.time_map(query))

    restored = TrainingModule(config).algorithm
    restored.load_state_dict(copy.deepcopy(algorithm.state_dict()))
    torch.testing.assert_close(restored.views[0].time_map(time), actual.time_map(time))
    before = clock.refit_count.clone()
    sampler = instantiate(config.sampler)
    sampler.steps = 3
    samples, info = algorithm.generate(sampler, tokens, ~mask)
    assert torch.equal(samples[~mask], tokens[~mask])
    assert torch.equal(clock.refit_count, before)

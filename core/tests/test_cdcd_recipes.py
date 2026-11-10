"""The worked examples' CDCD recipes train, warp time once and sample."""

import pytest
import torch
from hydra.utils import instantiate

from core.modifiers.hybrid.time_maps import IdentityTimeMap
from core.sweeps.configuration import compose
from core.noise_schedules.learned_time_maps import CDCDTimeMap
from core.time_sampling.uniform import UniformTimeSampler


def test_gaussian_cdcd_training_and_sampling():
    cfg = compose(overrides=["+sweep=sc_vs_depth"])
    cfg.model.hidden_size = 32
    cfg.model.cond_dim = 16
    cfg.model.n_heads = 4
    cfg.model.n_blocks = 1
    cfg.model.dropout = 0
    cfg.time_map.warmup = 1
    algo = instantiate(cfg.algorithm)
    # The learned map is on the information path; the sampler only draws.
    assert isinstance(algo.time_map, CDCDTimeMap) and algo.time_map.adaptive
    assert isinstance(algo.time_sampler, UniformTimeSampler)
    assert algo.adaptive_time_views == (0,)
    assert all(isinstance(v.time_map, IdentityTimeMap) for v in algo.views)
    tokens = torch.randint(0, cfg.task.vocab_size, (2, 8))
    mask = torch.ones_like(tokens, dtype=torch.bool)
    loss = algo.loss(tokens, mask, step=0, training=True)["loss"]
    assert torch.isfinite(loss)
    loss.backward()
    assert algo.time_map.pending_count == 2
    algo.time_map.post_optimizer_step()
    assert algo.time_map.ready
    # Deliberately nonidentity fitted clock to detect ignored/double warping.
    with torch.no_grad():
        algo.time_map.ema_logits[1].copy_(
            torch.linspace(-3, 0, algo.time_map.bins)
        )
    t = torch.tensor([0.2, 0.7])
    expected = algo.time_map(t)
    assert not torch.allclose(expected, t)
    for training in (False, True):
        points = algo.points(t, training=training)
        torch.testing.assert_close(points[0].time, expected)
    torch.testing.assert_close(
        algo._adaptive_time(torch.tensor([0.0, 1.0])), torch.tensor([0.0, 1.0])
    )
    cfg.sampler.steps = 2
    prompt = torch.zeros_like(mask)
    prompt[:, :2] = True
    result, _ = algo.eval().generate(instantiate(cfg.sampler), tokens, prompt)
    assert torch.equal(result[prompt], tokens[prompt])
    assert result.shape == tokens.shape


@pytest.mark.parametrize("path", ["simplex", "spherical", "spherical_vmf"])
@pytest.mark.parametrize("feedback", ["none", "expected_embedding"])
def test_geometry_cdcd_training_and_sampling(path, feedback):
    cfg = compose(
        overrides=[
            "+sweep=continuous_schedules",
            f"ablation_path={path}",
            f"modifiers/feedback={feedback}",
        ]
    )
    cfg.model.hidden_size = 32
    cfg.model.cond_dim = 16
    cfg.model.n_heads = 4
    cfg.model.n_blocks = 1
    cfg.model.dropout = 0
    cfg.time_map.warmup = 1
    algo = instantiate(cfg.algorithm)
    assert isinstance(algo.time_map, CDCDTimeMap)
    assert isinstance(algo.time_sampler, UniformTimeSampler)
    assert algo.adaptive_time_views == (0,)
    assert isinstance(algo.views[0].time_map, IdentityTimeMap)
    if path != "simplex":
        assert cfg.model.views.content.representation.state_representation.trainable
    tokens = torch.randint(0, cfg.task.vocab_size, (2, 8))
    mask = torch.ones_like(tokens, dtype=torch.bool)
    loss = algo.loss(tokens, mask, step=0, training=True)["loss"]
    assert torch.isfinite(loss)
    loss.backward()
    assert all(
        torch.isfinite(p.grad).all() for p in algo.parameters() if p.grad is not None
    )
    algo.time_map.post_optimizer_step()
    assert algo.time_map.ready
    with torch.no_grad():
        algo.time_map.ema_logits[1].copy_(
            torch.linspace(-3, 0, algo.time_map.bins)
        )
    times = torch.tensor([0.2, 0.7])
    torch.testing.assert_close(algo.points(times)[0].time, algo.time_map(times))
    prompt = torch.zeros_like(mask)
    prompt[:, :2] = True
    for sampler in [
        instantiate(cfg.sampler, steps=2),
        instantiate(cfg.sampler, steps=2, greedy_clean_estimate=True),
        instantiate(
            {
                "_target_": "core.samplers.parallel_entropy_bounded.ParallelEntropyBoundedSampler",
                "steps": 2,
                "freeze": True,
                "greedy_clean_estimate": True,
                "entropy_bound": 0.1,
            }
        ),
    ]:
        result, _ = algo.eval().generate(sampler, tokens, prompt)
        assert torch.equal(result[prompt], tokens[prompt])
        assert result.shape == tokens.shape

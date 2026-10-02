from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate

from core.processes.spherical import exp_map, log_map, sphere_normalize, slerp
from core.noise_schedules.spherical import SphericalSchedule
from core.samplers.spherical import posterior_velocity


def config(experiment="sudoku_spherical_truncated_adaptive"):
    with initialize_config_dir(
        config_dir=str(Path("configs").resolve()), version_base=None
    ):
        return compose(
            config_name="config",
            overrides=[
                f"+experiment={experiment}",
                "model.hidden_size=32",
                "model.n_blocks=1",
                "model.n_heads=4",
                "model.dropout=0",
                "representation.dimension=32",
                "sampler.steps=4",
            ],
        )


def test_geometry_endpoints_and_gradient():
    torch.manual_seed(7)
    x = sphere_normalize(torch.randn(2, 3, 7, dtype=torch.float64)).requires_grad_()
    y = sphere_normalize(torch.randn_like(x))
    torch.testing.assert_close(exp_map(x, log_map(x, y)), y)
    torch.testing.assert_close(slerp(x, y, torch.zeros(2)), x)
    torch.testing.assert_close(slerp(x, y, torch.ones(2)), y)
    for target in (x, -x):
        torch.testing.assert_close(exp_map(x, log_map(x, target)), target)
    slerp(x, y, torch.full((2,), 0.3)).square().sum().backward()
    assert torch.isfinite(x.grad).all()


def test_velocity_reference_and_tangency():
    torch.manual_seed(4)
    x = sphere_normalize(torch.randn(2, 3, 8, dtype=torch.float64))
    embeddings = sphere_normalize(torch.randn(5, 8, dtype=torch.float64))
    p = torch.randn(2, 3, 5, dtype=torch.float64).softmax(-1)
    reference = sum(
        p[..., i, None] * log_map(x, row.expand_as(x))
        for i, row in enumerate(embeddings)
    )
    actual = posterior_velocity(x, embeddings, p)
    torch.testing.assert_close(actual, reference)
    torch.testing.assert_close(
        (actual * x).sum(-1), torch.zeros(2, 3, dtype=x.dtype), atol=1e-12, rtol=0
    )


@pytest.mark.parametrize(
    "name",
    [
        "spherical_flow_matching",
        "sudoku_spherical",
        "sudoku_spherical_adaptive",
        "sudoku_spherical_truncated_adaptive",
    ],
)
def test_presets_training_sampling_and_restore(name):
    cfg = config(name)
    algo = instantiate(cfg.algorithm)
    tokens = torch.randint(0, 12, (3, 10))
    mask = torch.ones_like(tokens, dtype=torch.bool)
    mask[:, :3] = False
    optimizer = torch.optim.Adam(algo.parameters(), lr=1e-3)
    result = algo.loss(tokens, mask, training=True)
    result["loss"].backward()
    assert torch.isfinite(result["loss"])
    optimizer.step()
    optimizer.zero_grad()
    second = algo.loss(tokens, mask, training=True)
    second["loss"].backward()
    assert algo.representation.table.grad is not None
    assert torch.isfinite(algo.representation.table.grad).all()
    replica = instantiate(cfg.algorithm)
    replica.load_state_dict(algo.state_dict())
    torch.manual_seed(1)
    generated, metadata = algo.generate(instantiate(cfg.sampler), tokens, ~mask)
    torch.manual_seed(1)
    restored, _ = replica.generate(instantiate(cfg.sampler), tokens, ~mask)
    assert torch.equal(generated, restored)
    assert torch.equal(generated[:, :3], tokens[:, :3])
    assert metadata["nfe"] == 4
    if "adaptive" in name:
        assert algo.schedule.coordinate == "physical"


def test_truncation_and_adaptive_coordinate():
    schedule = SphericalSchedule(auto_truncation=True)
    assert 0 < schedule.alpha_max < 0.2
    keep, _ = schedule(torch.tensor([0.0, 1.0]))
    torch.testing.assert_close(keep, torch.tensor([schedule.alpha_max, 0.001]))
    cfg = config()
    cfg.noise_schedule.buffer_size = 16
    cfg.noise_schedule.warmup_steps = 0
    cfg.noise_schedule.refit_every = 1
    cfg.noise_schedule.n_knots = 4
    algo = instantiate(cfg.algorithm)
    t = torch.linspace(0.01, 0.99, 16)
    algo.schedule.observe(t, t.square(), 0, training=True)
    assert algo.schedule.refit_count == 1
    before = algo.schedule(torch.tensor([0.0, 1.0]))[0]
    replica = instantiate(cfg.algorithm)
    replica.load_state_dict(algo.state_dict())
    torch.testing.assert_close(before, replica.schedule(torch.tensor([0.0, 1.0]))[0])
    torch.testing.assert_close(
        before, torch.tensor([algo.schedule.base.base.alpha_max, 0.001])
    )


def test_default_adaptation_uses_sampled_time():
    with initialize_config_dir(
        config_dir=str(Path("configs").resolve()), version_base=None
    ):
        cfg = compose(
            config_name="config",
            overrides=[
                "+experiment=spherical_flow_matching",
                "noise_schedule=spherical_adaptive",
            ],
        )
    assert instantiate(cfg.noise_schedule).coordinate == "legacy_time"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_cuda_bfloat16_training_prediction():
    algo = instantiate(config().algorithm).cuda()
    tokens = torch.randint(0, 12, (2, 10), device="cuda")
    mask = torch.ones_like(tokens, dtype=torch.bool)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        prediction = algo.training_prediction(
            algo.representation.encode(tokens), torch.full((2,), 0.1, device="cuda")
        )
        assert prediction.dtype == torch.float32
        result = algo.loss(tokens, mask)
    result["loss"].backward()
    assert torch.isfinite(result["loss"])

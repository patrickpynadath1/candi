"""Composed Gaussian choices retain training, adaptive feedback, and resume contracts."""

from pathlib import Path

from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
import pytest
import torch

from core.noise_schedules.adaptive.gaussian import CDCDGaussianSchedule
from core.noise_schedules.rescaled_gaussian import RescaledGaussianSchedule
from core.samplers.euler import EulerSampler
from core.training.module import TrainingModule


def configuration():
    with initialize_config_dir(
        version_base=None, config_dir=str(Path(__file__).parents[2] / "configs")
    ):
        return compose(
            config_name="config",
            overrides=[
                "+experiment=gaussian_diffusion_embedding_sudoku",
                "noise_schedule=linear_interpolation",
                "model.hidden_size=16",
                "model.n_heads=2",
                "model.n_blocks=1",
                "model.cond_dim=8",
                "model.dropout=0",
                "model.input_preconditioning=none",
                "model.time_conditioning=log_snr",
                "representation.dimension=8",
                "representation.trainable=true",
                "parameterization.softcap=2",
                "objective=cross_entropy_learned",
                "objective.label_smoothing=0.1",
                "objective.weighting.cond_dim=8",
                "objective.weighting.hidden_dim=8",
                "training.warmup_steps=0",
                "data.cache_dir=/tmp/core_component_data",
                "output_dir=/tmp/core_component_integration",
            ],
        )


def batch():
    tokens = torch.tensor([[0, 1, 2, 3], [3, 2, 1, 0]])
    mask = torch.tensor([[0, 0, 1, 1], [0, 1, 1, 1]])
    return tokens, mask


def test_composed_components_optimizer_ema_and_checkpoint(tmp_path):
    cfg = configuration()
    module = TrainingModule(cfg)
    algorithm = module.algorithm
    assert algorithm.model.input_projection is not None
    parameters = dict(algorithm.named_parameters())
    learned = {
        name: p for name, p in parameters.items() if name.startswith("objective.")
    }
    assert learned
    optimizer = module.configure_optimizers()["optimizer"]
    optimized = {id(p) for group in optimizer.param_groups for p in group["params"]}
    assert all(id(p) in optimized for p in learned.values())
    before = {name: p.detach().clone() for name, p in learned.items()}
    tokens, mask = batch()
    loss = algorithm.loss(tokens, mask, training=True)["loss"]
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in learned.values())
    assert algorithm.model.input_projection.weight.grad.abs().sum() > 0
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in algorithm.representation.parameters()
    )
    optimizer.step()
    module.ema.update(algorithm.parameters())
    assert any(not torch.equal(before[name], p) for name, p in learned.items())
    for parameter, shadow in zip(
        (p for p in algorithm.parameters() if p.requires_grad), module.ema.shadow_params
    ):
        if id(parameter) in {id(p) for p in learned.values()}:
            assert torch.isfinite(shadow).all()
    checkpoint = {
        "state_dict": module.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    module.on_save_checkpoint(checkpoint)
    torch.save(checkpoint, tmp_path / "state.pt")
    saved = torch.load(tmp_path / "state.pt", weights_only=False)
    restored = TrainingModule(cfg)
    restored.load_state_dict(saved["state_dict"])
    restored.on_load_checkpoint(saved)
    restored_optimizer = restored.configure_optimizers()["optimizer"]
    restored_optimizer.load_state_dict(saved["optimizer"])
    assert len(restored_optimizer.state) == len(optimizer.state)
    with module.evaluation_weights(), restored.evaluation_weights():
        torch.manual_seed(5)
        expected = algorithm.loss(tokens, mask)["loss"]
        torch.manual_seed(5)
        actual = restored.algorithm.loss(tokens, mask)["loss"]
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        samples, _ = restored.algorithm.generate(
            EulerSampler(steps=2), tokens, ~mask.bool()
        )
        assert torch.equal(samples[~mask.bool()], tokens[~mask.bool()])


@pytest.mark.parametrize("path_type", ["ve", "vp", "linear"])
def test_adaptation_observes_raw_ce_under_weighting_and_rescaling(path_type):
    algorithm = instantiate(configuration().algorithm)
    clock = CDCDGaussianSchedule(algorithm.schedule)
    algorithm.schedule = RescaledGaussianSchedule(clock, path_type=path_type)
    observations = []
    clock.observe = lambda time, loss, step, **kwargs: observations.append(loss)
    predictions = []
    original = algorithm.training_prediction

    def predict(*args):
        value = original(*args)
        predictions.append(value.detach())
        return value

    algorithm.training_prediction = predict
    with torch.no_grad():
        algorithm.objective.weighting.mlp[-1].bias.fill_(2)
    tokens, mask = batch()
    result = algorithm.loss(tokens, mask, training=True)
    raw_ce = -predictions[-1].gather(-1, tokens[..., None]).squeeze(-1)
    expected = (raw_ce * mask).sum(-1) / mask.sum(-1)
    torch.testing.assert_close(observations[0], expected)
    assert not torch.isclose(result["loss"], (raw_ce * mask).sum() / mask.sum())
    algorithm.loss(tokens, mask, training=False)
    assert len(observations) == 1


def test_two_argument_custom_objective_remains_supported():
    cfg = configuration()
    cfg.objective.weighting = None
    cfg.objective.label_smoothing = 0
    algorithm = instantiate(cfg.algorithm)
    tokens, mask = batch()
    torch.manual_seed(10)
    expected = algorithm.loss(tokens, mask)["loss"]
    del algorithm.objective
    algorithm.objective = lambda prediction, targets: (
        -prediction.gather(-1, targets[..., None]).squeeze(-1)
    )
    torch.manual_seed(10)
    actual = algorithm.loss(tokens, mask)["loss"]
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)

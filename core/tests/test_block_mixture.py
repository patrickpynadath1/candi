"""Block mixtures preserve the path and loss while changing training layout."""

from unittest.mock import patch
import pytest
import torch

from core.modifiers.block_diffusion.mixture import BlockSizeMixture
from core.sweeps.configuration import compose
from core.training.module import TrainingModule


def test_stratification_across_ranks_and_accumulation_is_resume_stable():
    mixture = BlockSizeMixture([1, 16], [0.25, 0.75], sampling="stratified", seed=13)

    def draw():
        return [
            mixture.sample(
                step=7, world_size=2, rank=r, accumulation_steps=4, accumulation_index=a
            )
            for a in range(4)
            for r in range(2)
        ]

    values = draw()
    torch.rand(100)
    assert draw() == values
    assert values.count(1) == 2 and values.count(16) == 6


@pytest.mark.parametrize(
    "sizes,probabilities",
    [([0], [1]), ([1, 1], [0.5, 0.5]), ([1, 2], [1]), ([1], [float("nan")])],
)
def test_invalid_mixtures_fail(sizes, probabilities):
    with pytest.raises(ValueError):
        BlockSizeMixture(sizes, probabilities)


def tiny(recipe):
    return compose(
        recipe,
        {
            "model.hidden_size": 16,
            "model.n_heads": 2,
            "model.n_blocks": 1,
            "model.cond_dim": 8,
            "model.dropout": 0.0,
            "task.length": 8,
            "block_diffusion.block_size": 4,
            "training.ema": 0,
        },
    )


@pytest.mark.parametrize("kind", ["masked", "uniform"])
def test_mixture_uses_selected_layout_and_exact_noise_without_mutating_defaults(kind):
    algorithm = TrainingModule(tiny(f"blockgen_{kind}_mixture")).algorithm
    tokens = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])
    mask = torch.ones_like(tokens, dtype=torch.bool)
    mask[:, 0] = False
    original_objective = algorithm.algorithm.objective
    original_sampler = algorithm.algorithm.time_sampler
    from core.modifiers.block_diffusion.layout import make_layout

    for size in (1, 16):
        with (
            patch.object(algorithm.training_layout, "sample", return_value=size),
            patch(
                "core.modifiers.block_diffusion.model.make_layout", wraps=make_layout
            ) as layout,
            patch.object(
                algorithm.process, "corrupt", wraps=algorithm.process.corrupt
            ) as corrupt,
        ):
            result = algorithm.loss(tokens, mask, training=True)
            assert torch.isfinite(result["loss"])
            result["loss"].backward()
            assert all(
                torch.isfinite(p.grad).all()
                for p in algorithm.parameters()
                if p.grad is not None
            )
            assert layout.call_args.args[1] == size
            if size == 1:
                assert torch.equal(corrupt.call_args.args[1], torch.zeros(1))
            else:
                assert (corrupt.call_args.args[1] > 0).all()
            algorithm.zero_grad()
        assert algorithm.block_size == algorithm.model.block_size == 4
        assert algorithm.model._training_block_size.get() is None
        assert algorithm.algorithm.objective is original_objective
        assert algorithm.algorithm.time_sampler is original_sampler
    # Validation remains the explicitly configured fixed-size protocol.
    with (
        patch.object(algorithm.training_layout, "sample", side_effect=AssertionError),
        patch(
            "core.modifiers.block_diffusion.model.make_layout", wraps=make_layout
        ) as layout,
    ):
        assert torch.isfinite(algorithm.loss(tokens, mask, training=False)["loss"])
        assert layout.call_args.args[1] == 4


def test_single_size_mixture_matches_existing_fixed_training():
    cfg = tiny("blockgen_masked")
    fixed = TrainingModule(cfg).algorithm
    mixed = TrainingModule(cfg).algorithm
    mixed.load_state_dict(fixed.state_dict())
    mixed.training_layout = BlockSizeMixture([4], [1.0])
    tokens = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])
    mask = torch.ones_like(tokens, dtype=torch.bool)
    torch.manual_seed(12)
    a = fixed.loss(tokens, mask, training=True)["loss"]
    torch.manual_seed(12)
    b = mixed.loss(tokens, mask, training=True)["loss"]
    torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_gaussian_reuses_mixture_and_call_scoped_time_objective():
    from core.tests.test_block_diffusion import configuration
    from core.modifiers.block_diffusion import BlockDiffusion
    from core.time_sampling.fixed import FixedTimeSampler
    from core.objectives.cross_entropy import CrossEntropy
    from hydra.utils import instantiate

    cfg = configuration("+experiment=gaussian_diffusion_sudoku")
    algorithm = BlockDiffusion(
        instantiate(cfg.algorithm),
        4,
        training=BlockSizeMixture(
            [1, 4],
            [0.5, 0.5],
            overrides={
                1: {"time_sampler": FixedTimeSampler(0.5), "objective": CrossEntropy()}
            },
        ),
    )
    tokens = torch.randint(0, cfg.task.vocab_size, (2, 8))
    mask = torch.ones_like(tokens, dtype=torch.bool)
    for size in (1, 4):
        with patch.object(algorithm.training_layout, "sample", return_value=size):
            loss = algorithm.loss(tokens, mask, training=True)["loss"]
            assert torch.isfinite(loss)
            loss.backward()
    with patch.object(algorithm.algorithm, "loss", side_effect=RuntimeError("test")):
        with pytest.raises(RuntimeError):
            algorithm.loss(tokens, mask, training=True)
    assert algorithm.model._training_block_size.get() is None

"""Integration contracts for independent views and shared model modifiers."""

from pathlib import Path
import pytest
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate

from core.modifiers.block_diffusion.training import BlockDiffusion
from core.modifiers.token_feedback.training import TokenFeedback


def compose_hybrid(first, second, feedback="none"):
    with initialize_config_dir(
        version_base=None, config_dir=str(Path(__file__).parents[2] / "configs")
    ):
        config = compose(
            config_name="config",
            overrides=[
                "+experiment=hybrid",
                f"hybrid_view@model.views.first={first}",
                f"hybrid_view@model.views.second={second}",
                "model.backbone.hidden_size=16",
                "model.backbone.n_heads=2",
                "model.backbone.n_blocks=2",
                "model.backbone.cond_dim=8",
                "model.backbone.dropout=0",
                "sampler.steps=2",
                f"self_conditioning.enabled={'true' if feedback == 'hidden' else 'false'}",
                "self_conditioning.source=hidden",
            ],
        )
    return instantiate(config.algorithm), instantiate(config.sampler)


@pytest.mark.parametrize("first", ["gaussian", "spherical", "simplex", "blurring"])
@pytest.mark.parametrize("second", ["masked", "uniform"])
@pytest.mark.parametrize("feedback", ["none", "hidden", "token"])
def test_compositions_train_and_sample(first, second, feedback):
    torch.manual_seed(42)
    algo, sampler = compose_hybrid(first, second, feedback)
    if feedback == "token":
        algo = TokenFeedback(algo, depth=0)
    tokens = torch.randint(0, 12, (2, 6))
    mask = torch.ones_like(tokens, dtype=torch.bool)
    mask[:, 0] = False
    result = algo.loss(tokens, mask, training=True)
    assert result["loss"].isfinite()
    result["loss"].backward()
    assert all(
        torch.isfinite(p.grad).all() for p in algo.parameters() if p.grad is not None
    )
    unused = [
        name
        for name, p in algo.named_parameters()
        if p.requires_grad and p.grad is None
    ]
    assert not unused, unused
    generated, info = algo.generate(sampler, tokens, ~mask)
    assert torch.equal(generated[:, 0], tokens[:, 0])
    assert generated.min() >= 0 and generated.max() < 12
    assert info["nfe"] == 3


@pytest.mark.parametrize(
    "first,second",
    [
        ("simplex", "gaussian"),
        ("spherical", "simplex"),
        ("spherical", "gaussian"),
        ("uniform", "spherical"),
        ("masked", "simplex"),
        ("blurring", "masked"),
    ],
)
def test_continuous_pairs_and_view_order_with_block_and_feedback(first, second):
    algo, sampler = compose_hybrid(first, second)
    algo = TokenFeedback(
        BlockDiffusion(algo, block_size=2, clean_loss="autoregressive"), depth=0
    )
    tokens = torch.randint(0, 12, (2, 6))
    mask = torch.ones_like(tokens, dtype=torch.bool)
    result = algo.loss(tokens, mask, training=True)
    assert result["loss"].isfinite()
    result["loss"].backward()
    generated, _ = algo.generate(sampler, tokens, ~mask)
    assert generated.shape == tokens.shape


@pytest.mark.parametrize("experiment", ["candi_hybrid", "uniform_simplex_hybrid"])
def test_existing_preset_composes_and_trains(experiment):
    with initialize_config_dir(
        version_base=None, config_dir=str(Path(__file__).parents[2] / "configs")
    ):
        config = compose(
            config_name="config",
            overrides=[
                f"+experiment={experiment}",
                "model.backbone.hidden_size=512",
                "model.backbone.n_blocks=1",
                "model.backbone.dropout=0",
            ],
        )
    algo = instantiate(config.algorithm)
    tokens = torch.randint(0, 12, (2, 6))
    result = algo.loss(tokens, torch.ones_like(tokens, dtype=torch.bool), training=True)
    assert result["loss"].isfinite()
    result["loss"].backward()


def test_adaptive_view_observes_local_time_and_restores_state():
    from core.noise_schedules.adaptive import SFLMAdaptiveSchedule
    from core.modifiers.hybrid.time_maps import PowerTimeMap

    algo, _ = compose_hybrid("spherical", "uniform")
    view = algo.views[0]
    view.schedule = SFLMAdaptiveSchedule(view.schedule, buffer_size=8, warmup_steps=0)
    view.time_map = PowerTimeMap(2)
    fixed = torch.tensor([0.2, 0.8])
    algo.time_sampler.sample = lambda *args, **kwargs: fixed
    tokens = torch.randint(0, 12, (2, 6))
    algo.loss(tokens, torch.ones_like(tokens, dtype=torch.bool), training=True, step=1)
    torch.testing.assert_close(view.schedule.t_buf[:2], fixed.square().double())
    saved = algo.state_dict()
    restored, _ = compose_hybrid("spherical", "uniform")
    restored.views[0].schedule = SFLMAdaptiveSchedule(
        restored.views[0].schedule, buffer_size=8, warmup_steps=0
    )
    restored.load_state_dict(saved)
    assert restored.views[0].schedule.num_observed == 2


def test_representation_schedule_contract_rejects_onehot_rank_on_embedding():
    from core.modifiers.hybrid.views import DiffusionView
    from core.noise_schedules.candi_rank import CANDIRankSchedule

    algo, _ = compose_hybrid("gaussian", "uniform")
    view = algo.views[0]
    with pytest.raises(ValueError, match="representation kind"):
        DiffusionView(
            view.process,
            CANDIRankSchedule(12, 0.05, 0.4),
            view.representation,
            view.update,
        )


@pytest.mark.parametrize("experiment", ["candi_hybrid", "uniform_simplex_hybrid"])
def test_pre_modifier_checkpoint_weights_still_load(experiment):
    from core.sweeps import compose

    config = compose(experiment, {"model.backbone.n_blocks": 1})
    source = instantiate(config.algorithm)
    saved = {}
    for key, value in source.state_dict().items():
        saved[
            key.replace(".views.", ".branches.").replace(
                ".representation.embedding", ".adapter.embedding"
            )
        ] = value.clone()
    target = instantiate(config.algorithm)
    target.load_state_dict(saved, strict=True)
    for key, value in target.state_dict().items():
        torch.testing.assert_close(value, source.state_dict()[key])


def test_saved_hybrid_config_upgrade_has_no_old_import_targets():
    from core.modifiers.hybrid.checkpoint import upgrade_config

    old = {
        "_target_": "core.algorithms.hybrid_diffusion.HybridDiffusion",
        "model": {
            "_target_": "core.models.hybrid.HybridModel",
            "branches": [
                {
                    "_target_": "core.hybrid.branches.GaussianBranch",
                    "representation": {
                        "_target_": "core.representations.one_hot.OneHotRepresentation",
                        "vocab_size": 12,
                    },
                    "adapter": {
                        "_target_": "core.models.hybrid.CANDIContinuousAdapter",
                        "state_dim": 12,
                    },
                    "time_map": {"_target_": "core.hybrid.time_maps.IdentityTimeMap"},
                }
            ],
        },
    }
    upgraded = upgrade_config(old)
    view = upgraded["model"]["views"][0]
    assert view["update"]["_target_"].endswith("GaussianPosteriorUpdate")
    assert view["representation"]["_target_"].endswith("CANDIRepresentation")
    assert "branches" in old["model"]  # Saved input isn't modified.
    assert "core.hybrid." not in str(upgraded)

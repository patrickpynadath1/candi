"""Forward resampling across process families and shared-view conditioning."""

from pathlib import Path
from unittest.mock import patch

import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from hydra.utils import instantiate


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="CUDA unavailable"
            ),
        ),
    ],
)
@pytest.mark.parametrize(
    "family",
    ["gaussian", "simplex", "spherical", "blurring", "masked", "uniform", "gidd"],
)
@pytest.mark.parametrize("source", ["none", "hidden", "posterior_mean"])
def test_pure_and_hybrid_forward_resampling(family, source, device):
    with initialize_config_dir(
        version_base=None, config_dir=str(Path(__file__).parents[2] / "configs")
    ):
        cfg = compose(
            config_name="config",
            overrides=[
                "+experiment=corruption_factorial",
                f"hybrid_view@model.views.content={family}",
                "sampler=forward_resampling",
                "sampler.steps=3",
                "model.backbone.hidden_size=16",
                "model.backbone.n_heads=2",
                "model.backbone.n_blocks=1",
                "model.backbone.cond_dim=8",
                "model.backbone.dropout=0",
                "training.ema=0",
                f"self_conditioning.enabled={str(source != 'none').lower()}",
                f"self_conditioning.source={source if source != 'none' else 'hidden'}",
            ],
        )
    tokens = torch.randint(0, 12, (3, 8), device=device)
    prompt = torch.zeros_like(tokens, dtype=torch.bool)
    prompt[0] = True
    prompt[1, :2] = True
    for hybrid in (False, True):
        if hybrid:
            OmegaConf.update(
                cfg,
                "model.views.discrete",
                OmegaConf.load("configs/hybrid_view/uniform.yaml"),
                force_add=True,
            )
        algo = instantiate(cfg.algorithm).to(device).eval()
        for greedy in (False, True):
            cfg.sampler.greedy_clean_estimate = greedy
            cfg.sampler.final_denoise = "greedy" if greedy else "sample"
            sampler = instantiate(cfg.sampler)
            # Spy on actual forward draws: every view must see the same clean
            # token sequence, including clamped prompts, at destination times.
            from contextlib import ExitStack

            with ExitStack() as stack:
                spies = [
                    stack.enter_context(
                        patch.object(
                            v.process, "corrupt_state", wraps=v.process.corrupt_state
                        )
                    )
                    for v in algo.views
                ]
                samples, info = algo.generate(sampler, tokens, prompt)
            assert samples.shape == tokens.shape
            assert torch.equal(samples[prompt], tokens[prompt])
            assert ((samples >= 0) & (samples < 12)).all()
            assert info == {"nfe": 4, "nfe_per_sample": [4, 4, 4]}
            for spy in spies:
                assert spy.call_count == 3
            if hybrid:
                for first, second in zip(
                    spies[0].call_args_list, spies[1].call_args_list
                ):
                    clean_tokens = second.args[0]
                    torch.testing.assert_close(
                        first.args[0], algo.views[0].representation.encode(clean_tokens)
                    )


def test_invalid_options():
    from core.samplers.forward_resampling import ForwardResamplingSampler

    for kwargs in (
        {"steps": 0},
        {"steps": 1.5},
        {"epsilon": 1},
        {"final_denoise": "invalid"},
    ):
        with pytest.raises(ValueError):
            ForwardResamplingSampler(**kwargs)

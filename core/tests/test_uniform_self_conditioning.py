"""Plug-in feedback supports both sources without posterior semantics."""

import pytest
import torch
from hydra.utils import instantiate

from core.sweeps.configuration import compose
from core.tests.test_self_conditioning import recorder


@pytest.mark.parametrize("source", ["hidden", "posterior_mean"])
@pytest.mark.parametrize("sampler", ["uniform_ancestral", "uniform_entropy_bounded"])
def test_plugin_bootstrap_and_sampling_feedback(source, sampler):
    cfg = compose(
        "uniform_diffusion",
        {
            "model.hidden_size": 16,
            "model.cond_dim": 8,
            "model.n_blocks": 1,
            "model.n_heads": 2,
            "model.dropout": 0.0,
            "self_conditioning.enabled": True,
            "self_conditioning.source": source,
            "self_conditioning.probability": 1.0,
            "sampler": sampler,
            "parameterization.temperature": 0.1,
            "parameterization.top_k": 1,
        },
    )
    algo = instantiate(cfg.algorithm)
    torch.nn.init.normal_(algo.model.output_layer.linear.weight, std=0.1)
    tokens = torch.randint(0, algo.process.vocab_size, (2, 7))
    keep = torch.full((2,), 0.5)
    calls, handle = recorder(algo.model)
    prediction = algo.training_prediction(tokens, keep)
    assert len(calls) == 2
    feedback = calls[1][1]["self_condition"]
    assert not feedback.requires_grad
    expected = (
        calls[0][2].hidden
        if source == "hidden"
        else (
            algo.predictor.training_plugin(calls[0][2].logits).exp()
            @ algo.model.vocab_embed.embedding
        )
    )
    torch.testing.assert_close(feedback, expected)
    (-prediction[..., 1].mean()).backward()
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in algo.model.conditioning_adapter.parameters()
    )
    handle.remove()
    loss = algo.loss(tokens, torch.ones_like(tokens, dtype=torch.bool))["loss"]
    assert torch.isfinite(loss)
    loss.backward()

    if sampler == "uniform_ancestral":
        cfg.sampler.steps = 3
    algo.predictor.top_k = 0
    algo.predictor.temperature = 1.0
    if sampler == "uniform_entropy_bounded":
        cfg.sampler.entropy_bound = 0.0
    sampler_obj = instantiate(cfg.sampler)
    prompt_mask = torch.zeros_like(tokens, dtype=torch.bool)
    prompt_mask[:, 0] = True
    calls, handle = recorder(algo.model)
    sampled, stats = algo.generate(sampler_obj, tokens, prompt_mask)
    handle.remove()
    assert len(calls) > 1
    assert calls[0][1]["self_condition"] is None
    assert calls[1][1]["self_condition"] is not None
    assert stats["nfe"] == len(calls)
    assert torch.equal(sampled[:, 0], tokens[:, 0])
    assert ((sampled >= 0) & (sampled < algo.process.vocab_size)).all()

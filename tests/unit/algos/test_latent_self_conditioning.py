"""Training detachment, carry lifetime, and actual masked/uniform integration."""

import pytest
import torch
from torch import nn
from models.self_conditioning import install_self_conditioning
from algos import algo_registry
from dataloader import get_tokenizer
from samplers import create_sampler, run_sampler, SamplingRequest
from tests.unit.algos.test_simplex import tiny_config
from tests.unit.algos.test_real_configs import (
    ALGORITHM_CONFIGS,
    _compose_tiny,
    _tokenizer_for,
)


class TinyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([nn.Linear(4, 4), nn.Linear(4, 4)])
        self.head = nn.Linear(4, 3)
        self.grad_modes = []

    def forward(self, x):
        self.grad_modes.append(torch.is_grad_enabled())
        for block in self.blocks:
            x = block(x)
        return self.head(x)


def test_installation_rejects_double_wrapping():
    model = TinyBackbone()
    install_self_conditioning(model, 4, 0.9)
    with pytest.raises(ValueError, match="already installed"):
        install_self_conditioning(model, 4, 0.9)


def test_detached_first_pass_and_zero_initialization():
    model = TinyBackbone()
    x = torch.randn(2, 3, 4, requires_grad=True)
    expected = model(x).detach()
    sc = install_self_conditioning(model, 4, 1.0)
    model.grad_modes.clear()
    actual = model(x)
    torch.testing.assert_close(actual, expected)
    actual.square().mean().backward()
    assert model.grad_modes == [False, True]
    assert sc.norm.weight.grad is not None and sc.norm.weight.grad.norm() > 0
    assert sc.previous is None and sc.current is None


def test_sampling_carry_one_pass_and_exception_cleanup():
    model = TinyBackbone()
    sc = install_self_conditioning(model, 4, 1.0)
    nn.init.ones_(sc.norm.weight)
    x = torch.randn(2, 3, 4)
    with pytest.raises(RuntimeError, match="test"):
        with torch.no_grad(), sc.trajectory():
            first = model(x)
            assert sc.previous is not None and not sc.previous.requires_grad
            second = model(x)
            assert not torch.equal(first, second)
            assert len(model.grad_modes) == 2
            raise RuntimeError("test")
    assert sc.previous is None and sc.current is None and not sc.sampling


def test_deepcopy_keeps_teacher_and_student_independent():
    import copy

    model = TinyBackbone()
    install_self_conditioning(model, 4, 1.0)
    teacher = copy.deepcopy(model)
    x = torch.randn(2, 3, 4)
    teacher(x).sum().backward()
    assert teacher.blocks[0].weight.grad is not None
    assert model.blocks[0].weight.grad is None


@pytest.mark.parametrize("algo,overrides", ALGORITHM_CONFIGS)
def test_optional_sc_real_algorithm_training(algo, overrides):
    config = _compose_tiny(
        algo,
        *overrides,
        "self_conditioning.enabled=true",
        "self_conditioning.probability=1.0",
    )
    model = algo_registry.get(config.algo.name)(config, _tokenizer_for(algo))
    x = torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]])
    loss = model._loss(x, torch.ones_like(x)).loss
    assert torch.isfinite(loss)
    loss.backward()
    assert any(
        p.grad is not None and torch.isfinite(p.grad).all() and p.grad.norm() > 0
        for p in model.backbone.parameters()
    )


@pytest.mark.parametrize("algo", ["mdlm", "duo_base"])
@pytest.mark.parametrize("pc", [False, True])
def test_masked_and_uniform_self_conditioning(algo, pc):
    config = tiny_config(
        True,
        **{
            "algo": algo,
            "sampling.predictor": "paper_discrete",
            "+algo.paper_time_conditioning": True,
        },
    )
    model = algo_registry.get(algo)(config, get_tokenizer(config))
    x = torch.randint(1, 10, (2, 6))
    mask = torch.ones_like(x)
    mask[:, :3] = 0
    loss = model._loss(x, mask).loss
    loss.backward()
    assert torch.isfinite(loss)
    model.eval()
    sampler = create_sampler(
        config, sampling_overrides={"predict_correct": pc, "time_grid": "cosine"}
    )
    result = run_sampler(
        sampler, model, SamplingRequest(2, 3, prompt_tokens=x, prompt_mask=~mask.bool())
    )
    assert torch.equal(result.samples[:, :3], x[:, :3])
    assert result.metadata["nfe"] == 3


@pytest.mark.parametrize("predictor", ["ar", "ar_cached"])
def test_autoregressive_sc_aligns_growing_prefixes(predictor):
    config = _compose_tiny(
        "ar", f"sampling.predictor={predictor}", "self_conditioning.enabled=true"
    )
    model = algo_registry.get("ar")(config, _tokenizer_for("ar")).eval()
    prompt = torch.tensor([[1, 2, 0, 0]])
    mask = torch.tensor([[True, True, False, False]])
    result = model.generate_samples(1, prompt_tokens=prompt, prompt_mask=mask)
    assert result.shape == prompt.shape
    assert torch.equal(result[:, :2], prompt[:, :2])

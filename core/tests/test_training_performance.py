"""Execution optimizations preserve gradients, checkpoint keys, and feedback."""

import copy
import pytest
import torch

from core.tests.view_inputs import view_prediction
from core.tests.test_self_conditioning import algorithm, inputs
from core.models.components import DiffusionBlocks


@pytest.mark.parametrize("optimization", ["compile", "checkpoint", "hidden_only"])
def test_execution_preserves_outputs_gradients_and_checkpoint_keys(optimization):
    torch.manual_seed(17)
    original = algorithm("hidden", "model.dropout=0")
    optimized = copy.deepcopy(original)
    if optimization == "compile":
        optimized.model.compile(backend="aot_eager", dynamic=True)
    elif optimization == "checkpoint":
        assert isinstance(optimized.model.block_execution, DiffusionBlocks)
        optimized.model.block_execution.activation_checkpointing = True
    else:
        optimized.model.hidden_only_bootstrap = True
    tokens, keep = inputs(original)
    expected = view_prediction(original.training_prediction, tokens, keep)
    actual = view_prediction(optimized.training_prediction, tokens, keep)
    torch.testing.assert_close(actual, expected)
    expected[..., 0].sum().backward()
    actual[..., 0].sum().backward()
    for (name, left), (other_name, right) in zip(
        original.named_parameters(), optimized.named_parameters()
    ):
        assert name == other_name
        assert (left.grad is None) == (right.grad is None)
        if left.grad is not None:
            torch.testing.assert_close(right.grad, left.grad, rtol=1e-4, atol=1e-5)
    assert original.state_dict().keys() == optimized.state_dict().keys()
    original.load_state_dict(optimized.state_dict(), strict=True)


def test_hidden_only_bootstrap_skips_unused_vocabulary_projection():
    model = algorithm("hidden", "model.dropout=0")
    model.model.hidden_only_bootstrap = True
    seen = []
    handle = model.model.output_layer.register_forward_hook(
        lambda *args: seen.append(1)
    )
    view_prediction(model.training_prediction, *inputs(model))
    handle.remove()
    assert len(seen) == 1  # Only the differentiable pass needs vocabulary logits.


def test_compiled_uniform_objective_matches_values_and_gradients():
    from core.objectives.uniform import UniformDiffusionObjective

    torch.manual_seed(11)
    eager = UniformDiffusionObjective(5)
    compiled = UniformDiffusionObjective(5)
    compiled.compile(backend="aot_eager", fullgraph=True)
    left = torch.randn(2, 7, 5, requires_grad=True)
    right = left.detach().clone().requires_grad_()
    tokens = torch.randint(0, 5, (2, 7))
    noisy = torch.randint(0, 5, (2, 7))
    keep = torch.tensor([[0.2], [0.8]])
    args = noisy, tokens, keep, -torch.ones_like(keep)
    expected = eager(left.log_softmax(-1), *args)
    actual = compiled(right.log_softmax(-1), *args)
    torch.testing.assert_close(actual, expected)
    expected.sum().backward()
    actual.sum().backward()
    torch.testing.assert_close(right.grad, left.grad)


ADAPTIVE = ["denoising_task/time_sampler=cdcd", "time_sampler.adaptive=true", "+algorithm.adaptive_time_views=[0]"]


@pytest.mark.parametrize(
    "experiment, overrides",
    [
        ("simplex", ["denoising_task/time_sampler=cdcd", "time_sampler.adaptive=true"]),
        ("uniform_diffusion", []),
        ("hybrid", ADAPTIVE),
        ("hybrid", [*ADAPTIVE, "information_path/hybrid_view@model.views.second=uniform"]),
    ],
    ids=["simplex", "uniform", "gaussian-masked", "gaussian-uniform"],
)
def test_self_conditioned_variants_compile_and_preserve_checkpoint_layout(experiment, overrides):
    from core.sweeps.configuration import compose
    from core.training.module import TrainingModule
    from core.training.performance import configure_performance

    cfg = compose(experiment, [*overrides, "self_conditioning.enabled=true", "self_conditioning.source=hidden"])
    cfg.task = {
        "_target_": "core.tests.test_entrypoint.TinyTask",
        "vocab_size": 32,
        "length": 8,
    }
    backbone = cfg.model.get("backbone", cfg.model)
    backbone.hidden_size = 32
    backbone.cond_dim = 16
    backbone.n_heads = 4
    backbone.n_blocks = 1
    backbone.dropout = 0
    cfg.self_conditioning.probability = 1
    cfg.training.performance = {
        "hidden_only_bootstrap": True,
        "compile": {
            "enabled": True,
            "backend": "aot_eager",
            "dynamic": True,
            "objective": True,
        },
    }
    module = TrainingModule(cfg)
    keys = list(module.state_dict())
    configure_performance(module)
    tokens = torch.randint(0, 31, (2, 8))
    mask = torch.ones_like(tokens, dtype=torch.bool)
    result = module.algorithm.loss(tokens, mask, training=True)
    assert torch.isfinite(result["loss"])
    result["loss"].backward()
    assert list(module.state_dict()) == keys


def test_fused_bf16_gradient_clipping_keeps_norm_limit():
    from core.tests.test_self_conditioning import config
    from core.training.module import TrainingModule

    cfg = config("hidden")
    cfg.optimizer.fused = True
    cfg.trainer.precision = "bf16-mixed"
    module = TrainingModule(cfg)
    parameter = next(module.parameters())
    optimizer = torch.optim.AdamW([parameter], fused=True)
    parameter.grad = torch.ones_like(parameter)
    module.configure_gradient_clipping(optimizer, 0.5, "norm")
    torch.testing.assert_close(parameter.grad.norm(), torch.tensor(0.5))


def test_resume_retains_explicit_fused_optimizer_execution():
    from core.tests.test_self_conditioning import config
    from core.training.module import TrainingModule

    cfg = config("hidden")
    cfg.optimizer.fused = False
    eager = TrainingModule(cfg).configure_optimizers()["optimizer"]
    saved = eager.state_dict()
    cfg.optimizer.fused = True
    fused = TrainingModule(cfg).configure_optimizers()["optimizer"]
    fused.load_state_dict(saved)
    assert all(
        group["fused"] is True and group["foreach"] is None
        for group in fused.param_groups
    )


def test_ddp_partition_workaround_keeps_model_compilation(monkeypatch):
    from types import SimpleNamespace
    from torch._dynamo import config as dynamo_config
    from core.training.performance import configure_performance
    from core.tests import networks

    monkeypatch.setattr(dynamo_config, 'optimize_ddp', True)
    model = networks.token(5, 16, 8, 1, 2)
    calls = []
    monkeypatch.setattr(model, 'compile', lambda **kwargs: calls.append(kwargs))
    module = SimpleNamespace(
        algorithm=model,
        config=SimpleNamespace(training={'performance': {'compile': {
            'enabled': True, 'backend': 'inductor', 'optimize_ddp': False}}}),
    )
    configure_performance(module)
    assert dynamo_config.optimize_ddp is False
    assert len(calls) == 1 and calls[0]['backend'] == 'inductor'

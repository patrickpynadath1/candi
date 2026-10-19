"""Execution optimizations preserve gradients, checkpoint keys, and feedback."""

import copy
import pytest
import torch
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
    expected = original.training_prediction(tokens, keep)
    actual = optimized.training_prediction(tokens, keep)
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
    model.training_prediction(*inputs(model))
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


@pytest.mark.parametrize(
    "experiment",
    [
        "tinygsm_simplex_sc",
        "tinygsm_simplex_masked_sc",
        "tinygsm_simplex_uniform_sc",
        "tinygsm_simplex_gaussian_sc",
        "tinygsm_simplex_sc_insertion",
        "tinygsm_masked_sc",
        "tinygsm_uniform_sc",
        "tinygsm_masked_gaussian_sc_adaptive",
        "tinygsm_uniform_gaussian_sc_adaptive",
        "tinygsm_masked_sc_insertion",
        "tinygsm_uniform_sc_insertion",
        "tinygsm_masked_gaussian_sc_adaptive_insertion",
        "tinygsm_uniform_gaussian_sc_adaptive_insertion",
    ],
)
def test_all_tinygsm_variants_compile_and_preserve_checkpoint_layout(experiment):
    from core.sweeps.configuration import compose
    from core.training.module import TrainingModule
    from core.training.performance import configure_performance

    cfg = compose(experiment)
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


def test_queue_overrides_apply_to_restored_snapshots(monkeypatch):
    import json
    from scripts import tinygsm_ray_queue as queue
    from core.sweeps.configuration import compose
    from omegaconf import OmegaConf

    cfg = compose("tinygsm_uniform_sc")
    monkeypatch.setenv(
        "CANDI_TINY_GSM_OVERRIDES",
        json.dumps(
            {
                "training": {"performance": {"compile": {"enabled": True}}},
                "loader": {"eval_batch_size": 16},
            }
        ),
    )
    monkeypatch.delenv("CANDI_MASKED_RESUME", raising=False)
    seen = []
    monkeypatch.setattr(queue, "run_trial", lambda spec: seen.append(spec["config"]))
    queue.run_job({"trial": {"config": OmegaConf.to_container(cfg)}})
    assert seen[0].training.performance.compile.enabled
    assert seen[0].callback.every_n_steps == 10000
    assert seen[0].loader.eval_batch_size == 16
    assert seen[0].loader.global_batch_size == 512


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

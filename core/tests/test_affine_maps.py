import copy
from pathlib import Path
import pytest
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf
from core.sweeps.configuration import compose
from core.training.procedures.affine_maps import (
    AffineMap,
    AffineStates,
    AffinePSDTarget,
    AffineMapSampler,
)


def base():
    config = compose(
        "corruption_factorial",
        overrides={
            "hybrid_view@model.views.content": "gaussian",
            "model.backbone.hidden_size": 32,
            "model.backbone.n_heads": 4,
            "model.backbone.n_blocks": 1,
            "model.backbone.cond_dim": 16,
            "model.backbone.dropout": 0.0,
            "self_conditioning.enabled": False,
        },
    )
    return instantiate(config.algorithm)


def batch():
    return dict(
        input_ids=torch.tensor([[1, 2, 3, 4], [2, 3, 1, 0]]),
        loss_mask=torch.tensor([[False, True, True, True]] * 2),
    )


def test_base_predictions_and_affine_composition():
    b = base().eval()
    m = AffineMap(b).eval()
    state = AffineStates()({"student": m}, batch(), 0)
    t, u, s = (state[k] for k in ("source", "middle", "destination"))
    actual = m(state["state"], t, s)
    expected = b.training_prediction((state["state"],), b.points(t), t)
    torch.testing.assert_close(actual, expected)
    # Avoid prompt clamping to test the algebra on every position.
    state["corruption_mask"][:] = True
    target = AffinePSDTarget()({"teacher": m}, state)["consistency"]
    first = m.transport(state["state"], m(state["state"], t, u).exp(), t, u)
    composed = m.transport(first, m(first, u, s).exp(), u, s)
    direct = m.transport(state["state"], target, t, s)
    torch.testing.assert_close(direct, composed, atol=1e-5, rtol=1e-4)
    assert state["state"].shape[-1] == 128  # embedding state, not vocab state


@pytest.mark.parametrize("endpoint", [False, True])
def test_procedure_gradients_sampling_and_reload(endpoint):
    m = AffineMap(base(), endpoint=endpoint)
    # Nonzero output weights exercise the JVP and destination adapter gradients.
    torch.nn.init.normal_(m.model.backbone.output_layer.linear.weight, std=0.02)
    for name, parameter in m.model.backbone.named_parameters():
        if "adaLN" in name and parameter.ndim > 1:
            torch.nn.init.normal_(parameter, std=0.02)
    name = "beckmann_inherited" if endpoint else "flow_map_inherited"
    cfg = OmegaConf.load(
        Path(__file__).parents[2] / "configs/procedure" / f"{name}.yaml"
    )
    proc = instantiate(cfg, student=m)
    result = proc(m, batch(), 0)
    assert torch.isfinite(result["loss"])
    result["loss"].backward()
    assert all(p.grad is None for p in proc.teacher.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in m.parameters())
    if not endpoint:
        assert m.destination_time.mlp[-1].weight.grad.abs().sum() > 0
    restored = copy.deepcopy(m)
    restored.load_state_dict(m.state_dict())
    sampler = AffineMapSampler(4)
    torch.manual_seed(1)
    tokens, metadata = restored.generate(
        sampler, batch()["input_ids"], ~batch()["loss_mask"]
    )
    assert metadata["nfe_per_sample"] == [4, 4]
    assert torch.equal(tokens[:, 0], batch()["input_ids"][:, 0])


def test_import_base_ema_into_adapter(tmp_path):
    from types import SimpleNamespace
    from core.training.ema import ExponentialMovingAverage
    from core.training.checkpoints import initialize_student

    original = base().eval()
    ema = ExponentialMovingAverage(original.parameters(), 0.99)
    # Distinguish EMA from raw weights to check the actual import policy.
    for value in ema.shadow_params:
        value.add_(0.001)
    path = tmp_path / "base.ckpt"
    torch.save(
        dict(
            core_format_version=1,
            state_dict={"algorithm." + k: v for k, v in original.state_dict().items()},
            ema=ema.state_dict(),
        ),
        path,
    )
    adapted = AffineMap(base()).eval()
    module = SimpleNamespace(
        algorithm=adapted, ema=ExponentialMovingAverage(adapted.parameters(), 0.99)
    )
    initialize_student(module, path)
    ema.copy_to(original.parameters())
    for actual, expected in zip(adapted.base.parameters(), original.parameters()):
        torch.testing.assert_close(actual, expected)
    assert len(module.ema.shadow_params) == sum(
        p.requires_grad for p in adapted.parameters()
    )
    torch.testing.assert_close(
        adapted.destination_time.mlp[-1].weight,
        torch.zeros_like(adapted.destination_time.mlp[-1].weight),
    )


def test_characteristic_target_points_toward_data():
    from types import SimpleNamespace
    from torch import nn
    from core.processes.gaussian import GaussianProcess
    from core.training.procedures.affine_maps import CharacteristicTarget

    class Predictor(nn.Module):
        def __init__(self):
            super().__init__()
            self.view = SimpleNamespace(
                process=GaussianProcess(),
                representation=SimpleNamespace(posterior_mean=lambda p: p),
            )

        def coefficients(self, t):
            # VE: sigma(t)=1+t, alpha=1, nonzero state velocity.
            return SimpleNamespace(alpha=torch.ones_like(t), sigma=1 + t)

        def forward(self, x, t, destination):
            return (x + t[:, None, None] * torch.tensor([1.0, -1.0])).log_softmax(-1)

    student, teacher = Predictor(), Predictor()
    x = torch.tensor([[[0.2, 0.5]]], dtype=torch.float64)
    t = torch.tensor([0.4], dtype=torch.float64)
    state = dict(
        state=x,
        source=t,
        corruption_mask=torch.ones(1, 1, dtype=torch.bool),
        tokens=torch.zeros(1, 1, dtype=torch.long),
    )
    target = CharacteristicTarget()({"teacher": teacher, "student": student}, state)[
        "consistency"
    ]
    value = student(x, t, t).exp()
    velocity = (x - teacher(x, t, t).exp()) / (1 + t)[:, None, None]
    eps = 1e-5
    later = student(x - eps * velocity, t - eps, t).exp()
    torch.testing.assert_close(
        target, value + (later - value) / eps, atol=1e-5, rtol=1e-4
    )
    assert not target.requires_grad

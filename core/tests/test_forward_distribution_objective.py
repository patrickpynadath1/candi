import pytest
import torch

from core.noise_schedules.gidd import GIDDSchedule
from core.processes.gidd import GIDDProcess
from core.objectives.forward_distribution import (
    ForwardDistributionObjective,
    ForwardKL,
    ObservedTokenIS,
)
from core.objectives.weighting import ObjectiveContext
from core.sweeps.configuration import compose
from core.training.module import TrainingModule


def test_terms_match_reference_and_composition_gradients():
    process = GIDDProcess(3)
    coefficients = GIDDSchedule()(torch.tensor([0.4]))
    targets = torch.tensor([[0, 1, 2]])
    observed = torch.tensor([[3, 0, 2]])
    logits = torch.randn(1, 3, 3, dtype=torch.float64, requires_grad=True)
    log_probs = logits.log_softmax(-1)
    context = ObjectiveContext(
        time=torch.tensor([0.4]),
        processes=(process,),
        coefficients=(coefficients,),
        states=(observed,),
    )
    q = process.marginal(torch.nn.functional.one_hot(targets, 3).double(), coefficients)
    p = process.marginal(log_probs.exp(), coefficients)
    expected_kl = torch.nn.functional.kl_div(p.log(), q, reduction="none").sum(-1)
    ratio = (q / p).gather(-1, observed[..., None]).squeeze(-1)
    expected_is = ratio - ratio.log() - 1
    objective = ForwardDistributionObjective(
        [ForwardKL(), ObservedTokenIS()], weights=[0.3, 0.7]
    )
    actual = objective(log_probs, targets, context=context)
    torch.testing.assert_close(actual, 0.3 * expected_kl + 0.7 * expected_is)
    actual.sum().backward()
    assert torch.isfinite(logits.grad).all() and logits.grad.abs().sum() > 0


@pytest.mark.parametrize("term", [ForwardKL(), ObservedTokenIS()])
def test_matching_distributions_and_zero_mass_are_finite(term):
    q = torch.tensor([[[0.0, 0.3, 0.7]]], dtype=torch.float64)
    p = q.clone().requires_grad_()
    result = term(q, p, torch.tensor([[2]]))
    torch.testing.assert_close(result, torch.zeros_like(result))
    result.sum().backward()
    assert torch.isfinite(p.grad).all()


@pytest.mark.parametrize(
    "objective", ["forward_kl", "observed_token_is", "forward_kl_is"]
)
@pytest.mark.parametrize("sc", [False, True])
def test_configured_objective_trains(objective, sc):
    config = compose(
        "gidd_corruption",
        {
            "objective": objective,
            "model.backbone.hidden_size": 16,
            "model.backbone.n_heads": 2,
            "model.backbone.n_blocks": 1,
            "model.backbone.cond_dim": 8,
            "training.ema": 0,
            "self_conditioning.enabled": sc,
        },
    )
    algorithm = TrainingModule(config).algorithm
    tokens = torch.randint(0, 12, (2, 8))
    mask = torch.ones_like(tokens, dtype=torch.bool)
    mask[:, 0] = False
    result = algorithm.loss(tokens, mask, mask, training=True)
    assert torch.isfinite(result["loss"])
    result["loss"].backward()
    gradients = [p.grad for p in algorithm.model.parameters() if p.grad is not None]
    assert gradients and all(torch.isfinite(g).all() for g in gradients)


def test_missing_context_is_explicit():
    objective = ForwardDistributionObjective([ForwardKL()])
    with pytest.raises(ValueError, match="view context"):
        objective(torch.zeros(1, 2, 3), torch.zeros(1, 2, dtype=torch.long))

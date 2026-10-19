import pytest
import torch

from core.noise_schedules.gidd import GIDDSchedule
from core.processes.gidd import GIDDProcess
from core.objectives.forward_distribution import ForwardKL, ObservedRatioPenalty
from core.objectives.composition import ComposedObjective, Component
from hydra.utils import instantiate
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
        representations=(None,),
    )
    q = process.marginal(torch.nn.functional.one_hot(targets, 3).double(), coefficients)
    p = process.marginal(log_probs.exp(), coefficients)
    expected_kl = torch.nn.functional.kl_div(p.log(), q, reduction="none").sum(-1)
    ratio = (q / p).gather(-1, observed[..., None]).squeeze(-1)
    expected_is = ratio - ratio.log() - 1
    config = compose("gidd_corruption", {"objective": "forward_kl_is"})
    objective = instantiate(config.objective)
    objective.terms[0].coefficient = 0.3
    objective.terms[1].coefficient = 0.7
    actual = objective(log_probs, targets, context=context)
    torch.testing.assert_close(actual, 0.3 * expected_kl + 0.7 * expected_is)
    actual.sum().backward()
    assert torch.isfinite(logits.grad).all() and logits.grad.abs().sum() > 0


@pytest.mark.parametrize("term", [ForwardKL(), ObservedRatioPenalty()])
def test_matching_distributions_and_zero_mass_are_finite(term):
    q = torch.tensor([[[0.0, 0.3, 0.7]]], dtype=torch.float64)
    p = q.clone().requires_grad_()
    qdist = torch.distributions.Categorical(probs=q)
    pdist = torch.distributions.Categorical(probs=p)
    result = (
        term(qdist, pdist)
        if isinstance(term, ForwardKL)
        else term(qdist, pdist, torch.tensor([[2]]))
    )
    torch.testing.assert_close(result, torch.zeros_like(result))
    result.sum().backward()
    assert torch.isfinite(p.grad).all()


@pytest.mark.parametrize(
    "objective",
    [
        "forward_kl",
        "observed_token_is",
        "forward_kl_is",
        "gidd_elbo",
        "gidd_clipped",
        "composed_ce",
    ],
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
    objective = instantiate(
        compose("gidd_corruption", {"objective": "forward_kl"}).objective
    )
    with pytest.raises(ValueError, match="view context"):
        objective(torch.zeros(1, 2, 3), torch.zeros(1, 2, dtype=torch.long))


def test_gaussian_kl_matches_weighted_squared_error():
    from core.processes.gaussian import GaussianProcess
    from core.noise_schedules.gaussian_path import GaussianPath

    a, sigma = torch.tensor([0.8]), torch.tensor([0.3])
    path = GaussianPath(a, sigma, torch.zeros(1), torch.ones(1))
    x = torch.randn(1, 4, 3, dtype=torch.float64)
    prediction = torch.randn_like(x, requires_grad=True)
    process = GaussianProcess()
    actual = ForwardKL()(
        process.forward_distribution(x, path),
        process.forward_distribution(prediction, path),
    )
    expected = (
        a.double()[:, None].square()
        / (2 * sigma.double()[:, None].square())
        * (x - prediction).square().sum(-1)
    )
    torch.testing.assert_close(actual, expected)
    actual.sum().backward()
    assert torch.isfinite(prediction.grad).all()


def test_weight_matches_reference_three_token_cases():
    from core.objectives.discrete_weighting import InterpolatingDiscreteWeight

    process = GIDDProcess(3)
    t = torch.tensor([0.2, 0.5, 0.8], dtype=torch.float64)
    path = GIDDSchedule(0.2)(t)
    clean = torch.nn.functional.one_hot(torch.zeros(3, 3, dtype=torch.long), 3).double()
    observed = torch.tensor([[0, 1, 3]]).expand(3, -1)
    target = process.forward_distribution(clean, path)
    actual = InterpolatingDiscreteWeight()(process, path, target, observed)
    c = 0.5 * (t * (1 - t)).sqrt()
    dc = 0.5 * (1 - 2 * t) / (t * (1 - t)) * c
    numer = c + (1 - t) * dc
    expected = torch.stack(
        (
            numer / 3 / ((1 - t) * (1 - t + c / 3)),
            numer / ((1 - t) * c),
            1 / ((1 - t) * t),
        ),
        -1,
    )
    torch.testing.assert_close(actual, expected)


def test_unknown_binding_rejected_at_construction():
    with pytest.raises(ValueError, match="unknown objective inputs"):
        ComposedObjective(
            [Component(ForwardKL(), {"target": "typo", "predicted": "targets"})]
        )


def test_insertion_config_binds_components_and_head_off_weight():
    config = compose(
        "insertion_continuous",
        {
            "variable_length.objective.content_weight": 0.0,
        },
    )
    objective = instantiate(config.variable_length.objective)
    from core.objectives.insertion import GapCountLoss, GapTokenLoss

    assert isinstance(objective.combination.terms[0].component, GapCountLoss)
    assert isinstance(objective.combination.terms[1].component, GapTokenLoss)
    assert objective.content_weight == 0


def test_gaussian_objective_training_uses_same_composition():
    config = compose(
        "gidd_corruption",
        {
            "hybrid_view@model.views.content": "gaussian",
            "objective": "gaussian_forward_kl",
            "model.backbone.hidden_size": 16,
            "model.backbone.n_heads": 2,
            "model.backbone.n_blocks": 1,
            "model.backbone.cond_dim": 8,
            "training.ema": 0,
        },
    )
    algorithm = TrainingModule(config).algorithm
    tokens = torch.randint(0, 12, (2, 8))
    mask = torch.ones_like(tokens, dtype=torch.bool)
    loss = algorithm.loss(tokens, mask, training=True)["loss"]
    assert torch.isfinite(loss)
    loss.backward()

import pytest
import torch
from hydra.utils import instantiate

from core.noise_schedules.gidd import GIDDSchedule
from core.processes.gidd import GIDDProcess
from core.sweeps.configuration import compose
from core.training.module import TrainingModule


@pytest.mark.parametrize("gamma", [0.5, 1.0, 2.0])
@pytest.mark.parametrize("p", [0.0, 0.2, 0.8])
def test_path_endpoints_peak_and_valid_forward_transitions(gamma, p):
    schedule = GIDDSchedule(p, gamma)
    t = torch.linspace(0, 1, 101, dtype=torch.float64)
    at = schedule(t)
    torch.testing.assert_close(at.alpha + at.mask + at.uniform, torch.ones_like(t))
    assert at.alpha[0] == 1 and at.mask[-1] == 1
    assert at.uniform[50].item() == pytest.approx(p)
    process = GIDDProcess(4)
    s = schedule(t[:-1])
    end = schedule(t[1:])
    conditional = process.mixing(end) - (end.alpha / s.alpha)[:, None] * process.mixing(
        s
    )
    assert conditional.min() >= -1e-12


def test_reverse_matches_explicit_forward_matrix_bayes():
    torch.manual_seed(4)
    process = GIDDProcess(4)
    schedule = GIDDSchedule(0.2)
    s, t = schedule(torch.tensor([0.3])), schedule(torch.tensor([0.7]))
    clean = torch.softmax(torch.randn(1, 5, 4, dtype=torch.float64), -1)
    state = torch.arange(5)[None]
    actual = process.reverse_probabilities(state, clean, t, s)
    transition = (t.alpha / s.alpha).item() * torch.eye(5, dtype=torch.float64)
    transition += process.mixing(t) - (t.alpha / s.alpha)[:, None] * process.mixing(s)
    prior = process.marginal(clean, s)
    joint = prior * transition[:, state[0]].T[None]
    expected = joint / joint.sum(-1, keepdim=True)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(actual.sum(-1), torch.ones(1, 5, dtype=torch.float64))
    # Visible tokens can change; at equal times, transitions are identity.
    assert actual[0, 0, 1] > 0
    torch.testing.assert_close(
        process.reverse_probabilities(state, clean, t, t),
        torch.eye(5, dtype=torch.float64)[None],
    )


def test_mask_only_limit_and_terminal_prior():
    process = GIDDProcess(3)
    schedule = GIDDSchedule(0)
    probs = torch.tensor([[[0.2, 0.3, 0.5], [0.2, 0.3, 0.5]]])
    state = torch.tensor([[0, 3]])
    result = process.reverse_probabilities(
        state, probs, schedule(torch.tensor([0.8])), schedule(torch.tensor([0.4]))
    )
    torch.testing.assert_close(
        result[0, 0], torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float64)
    )
    torch.testing.assert_close(
        result[0, 1], torch.tensor([0.1, 0.15, 0.25, 0.5], dtype=torch.float64)
    )
    prior = process.prior_state(state, schedule(torch.ones(1)))
    assert (prior == 3).all()
    terminal = process.reverse_probabilities(
        prior, probs, schedule(torch.ones(1)), schedule(torch.tensor([0.4]))
    )
    torch.testing.assert_close(
        terminal, process.marginal(probs, schedule(torch.tensor([0.4])))
    )


@pytest.mark.parametrize("sc", [False, True])
def test_gidd_recipe_trains_and_samples_with_prompt_preservation(sc):
    config = compose(
        "gidd_corruption",
        {
            "model.backbone.hidden_size": 16,
            "model.backbone.n_heads": 2,
            "model.backbone.n_blocks": 1,
            "model.backbone.cond_dim": 8,
            "training.ema": 0,
            "self_conditioning.enabled": sc,
            "sampler.steps": 3,
        },
    )
    algo = TrainingModule(config).algorithm
    tokens = torch.randint(0, 12, (2, 8))
    mask = torch.ones_like(tokens, dtype=torch.bool)
    mask[:, 0] = False
    result = algo.loss(tokens, mask, mask, training=True)
    assert torch.isfinite(result["loss"])
    result["loss"].backward()
    samples, info = algo.generate(instantiate(config.sampler), tokens, ~mask)
    assert (samples < 12).all()
    assert torch.equal(samples[~mask], tokens[~mask])
    assert info["nfe"] == 4

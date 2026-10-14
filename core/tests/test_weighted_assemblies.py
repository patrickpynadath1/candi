"""Time-based weighting stays usable when adding views or insertion."""

import pytest
import torch

from core.objectives.cross_entropy import CrossEntropy
from core.objectives.weighting import LearnedLossWeighting, PowerLossWeighting
from core.tests.test_hybrid_diffusion import make_algorithm
from core.tests.test_variable_length import assembly


@pytest.mark.parametrize("insertion", [False, True])
@pytest.mark.parametrize("learned", [False, True])
def test_time_weighting_receives_context_and_trains(insertion, learned):
    torch.manual_seed(42)
    algorithm = assembly() if insertion else make_algorithm()
    owner = algorithm.algorithm if insertion else algorithm
    weighting = (
        LearnedLossWeighting(coordinate="time", cond_dim=8, hidden_dim=8)
        if learned
        else PowerLossWeighting(coordinate="time", exponent=1)
    )
    objective = CrossEntropy(weighting=weighting)
    owner.objective = objective
    tokens = torch.tensor([[1, 2, 3, 4], [2, 3, 1, 4]])
    editable = torch.ones_like(tokens, dtype=torch.bool)
    editable[:, 0] = False
    seen = []

    def check_context(module, args, kwargs, output):
        context = kwargs["context"]
        assert context.time.shape == (2,)
        assert context.alpha is None and context.sigma is None
        assert ((context.time > 0) & (context.time < 1)).all()
        if not learned:
            expected = CrossEntropy()(*args) * context.time[:, None]
            torch.testing.assert_close(output, expected)
        seen.append(context.time)

    handle = objective.register_forward_hook(check_context, with_kwargs=True)
    result = algorithm.loss(tokens, editable, editable, training=True)
    handle.remove()
    assert len(seen) == 1
    assert torch.isfinite(result["loss"])
    result["loss"].backward()
    if learned:
        gradient = weighting.mlp[-1].bias.grad
        assert gradient is not None and torch.isfinite(gradient).all()
        assert gradient.abs().sum() > 0

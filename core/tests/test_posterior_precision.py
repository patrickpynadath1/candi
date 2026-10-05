"""Posterior probabilities and gradients must survive mixed precision and offsets."""

import pytest
import torch
from core.parameterizations.posterior import PosteriorParameterization
from core.parameterizations.utils import normalize_log_probs


@pytest.mark.parametrize(
    "dtype", [torch.bfloat16, torch.float16, torch.float32, torch.float64]
)
@pytest.mark.parametrize("offset", [-2048.0, 0.0, 2048.0])
def test_normalization_and_ce_gradient(dtype, offset):
    logits = (
        (torch.tensor([[[0.0, 8.0, -8.0, 16.0], [0.0, 0.0, 0.0, 0.0]]]) + offset)
        .to(dtype)
        .requires_grad_()
    )
    result = normalize_log_probs(logits)
    expected_dtype = torch.float64 if dtype == torch.float64 else torch.float32
    assert result.dtype == expected_dtype
    reference = logits.detach().to(expected_dtype).requires_grad_()
    expected = reference.log_softmax(-1)
    torch.testing.assert_close(result, expected)
    torch.testing.assert_close(result.exp().sum(-1), torch.ones_like(result[..., 0]))
    actual_grad = torch.autograd.grad(-result[..., 0].mean(), logits)[0]
    expected_grad = torch.autograd.grad(-expected[..., 0].mean(), reference)[0].to(
        dtype
    )
    torch.testing.assert_close(actual_grad, expected_grad)


@pytest.mark.parametrize(
    "path", ["training_posterior", "sampling_posterior", "greedy_posterior"]
)
def test_shaping_and_constraints_keep_normalized_fp32(path):
    predictor = PosteriorParameterization(
        mask_index=3, temperature=0.7, top_k=2, top_p=0.9
    )
    logits = torch.tensor(
        [[[-2048.0, -2048.0, -2048.0, -2048.0], [-2048.0, -2040.0, -2032.0, -2024.0]]],
        dtype=torch.bfloat16,
    )
    observed = torch.tensor([[3, 1]])
    result = getattr(predictor, path)(logits, observed)
    assert result.dtype == torch.float32
    torch.testing.assert_close(result.exp().sum(-1), torch.ones(1, 2))
    assert result.exp()[0, 1, 1] == 1
    assert result.exp()[..., 3].eq(0).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA autocast")
def test_cuda_autocast_backward_uses_stable_posterior():
    model = torch.nn.Linear(8, 12).cuda()
    with torch.no_grad():
        model.bias.fill_(-2048)
    inputs = torch.randn(4, 3, 8, device="cuda")
    target = torch.randint(12, (4, 3), device="cuda")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits = model(inputs)
        logits.retain_grad()
        prediction = PosteriorParameterization().training_posterior(logits, target)
        loss = -prediction.gather(-1, target[..., None]).mean()
    reference = torch.nn.functional.cross_entropy(
        logits.detach().float().flatten(0, 1), target.flatten()
    )
    torch.testing.assert_close(loss, reference)
    torch.testing.assert_close(
        prediction.exp().sum(-1), torch.ones_like(prediction[..., 0])
    )
    loss.backward()
    assert torch.isfinite(model.weight.grad).all()
    assert logits.grad.float().sum(-1).abs().max() < 0.001

"""Numerical checks, not evidence of training-quality equivalence."""

from types import SimpleNamespace
import pytest
import torch
from core.processes.simplex_math import log_gamma_sample, log_dirichlet_sample
from core.processes.dirichlet import DirichletProcess
from core.models.simplex_projection import simplex_projection

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@pytest.mark.parametrize("backend", ["torch", "triton"])
@pytest.mark.parametrize("a", [1e-8, 1e-4, 0.1, 1.0, 10.0, 1000.0])
def test_moments(backend, a):
    torch.manual_seed(731)
    shape = torch.full((120000, 3), a, device=DEVICE)
    log_g = log_gamma_sample(shape, backend).double()
    g = log_g.exp()
    alpha = torch.tensor(a, dtype=torch.float64)
    log_mean = torch.digamma(alpha).item()
    log_var = torch.polygamma(1, alpha).item()
    log_fourth_cumulant = torch.polygamma(3, alpha).item()
    assert abs(log_g.mean().item() - log_mean) < 7 * (log_var / g.numel()) ** 0.5
    assert (
        abs(log_g.var().item() - log_var)
        < 8 * ((log_fourth_cumulant + 2 * log_var**2) / g.numel()) ** 0.5
    )
    # Small shapes have high kurtosis: bound sample errors by theory SE.
    assert abs(g.mean().item() - a) < 7 * (a / g.numel()) ** 0.5
    assert abs(g.var().item() - a) < 8 * ((2 * a * a + 6 * a) / g.numel()) ** 0.5
    p = log_dirichlet_sample(shape, backend).exp().double()
    expected_var = (2 / 9) / (3 * a + 1)
    assert (p.mean(0) - 1 / 3).abs().max() < 0.012
    assert (p.var(0) - expected_var).abs().max() < 0.008
    torch.testing.assert_close(
        p.sum(-1),
        torch.ones(len(p), device=DEVICE, dtype=torch.float64),
        atol=2e-6,
        rtol=0,
    )


@pytest.mark.parametrize("backend", ["torch", "triton"])
def test_faces_seed_fallback(backend):
    a = torch.tensor([[1e-8, 2e-8, 0], [0, 2.0, 0]], device=DEVICE)
    torch.manual_seed(123)
    x = log_dirichlet_sample(a, backend)
    torch.manual_seed(123)
    assert torch.equal(x, log_dirichlet_sample(a, backend))
    assert not torch.equal(x, log_dirichlet_sample(a, backend))
    assert torch.isfinite(x[0, :2]).all()
    assert torch.isneginf(x[:, 2]).all()
    assert x[1, 1] == 0
    a = torch.ones(3, device=DEVICE, requires_grad=True)
    log_gamma_sample(a, backend).sum().backward()
    assert torch.isfinite(a.grad).all()
    for bad in [torch.tensor([-1.0]), torch.tensor([float("nan")])]:
        with pytest.raises(ValueError):
            log_gamma_sample(bad, backend)


@pytest.mark.parametrize("precision,tolerance", [("fp32", 2e-6), ("bf16", 0.015)])
@pytest.mark.parametrize("chunk", [0, 3, 16])
def test_projection(precision, tolerance, chunk):
    torch.manual_seed(31)
    state = (
        torch.randn(2, 7, 83, device=DEVICE).log_softmax(-1).detach().requires_grad_()
    )
    table = torch.randn(83, 17, device=DEVICE, requires_grad=True)
    grad = torch.randn(2, 7, 17, device=DEVICE)
    ref = state.exp() @ table
    expected = torch.autograd.grad(ref, (state, table), grad)
    actual = simplex_projection(state, table, precision, chunk)
    grads = torch.autograd.grad(actual, (state, table), grad)
    for x, y in zip((actual, *grads), (ref, *expected)):
        assert (x - y).norm() / y.norm() < tolerance


@pytest.mark.parametrize("backend", ["torch", "triton"])
def test_chunked_corruption(backend):
    x = torch.tensor([[0, 1, 2], [1, 2, 0]], device=DEVICE)
    path = SimpleNamespace(
        time=torch.tensor([0.0, 1.0], device=DEVICE),
        concentration=torch.ones(2, device=DEVICE),
    )
    mask = torch.tensor([[1, 1, 1], [0, 1, 1]], device=DEVICE)
    for chunk in [0, 1, 4]:
        process = DirichletProcess(
            3, precision="fp32", sampler_backend=backend, chunk_size=chunk
        ).to(DEVICE)
        state = process.corrupt(x, path, mask)
        assert state.dtype == torch.float32
        torch.testing.assert_close(state[0], process.encode(x)[0])
        torch.testing.assert_close(state[1, 0], process.encode(x)[1, 0])
        torch.testing.assert_close(
            state.logsumexp(-1), torch.zeros(2, 3, device=DEVICE), atol=1e-6, rtol=0
        )


@pytest.mark.skipif(DEVICE != "cuda", reason="CUDA required")
def test_compile():
    a = torch.ones(256, 3, device=DEVICE)
    f = torch.compile(lambda x: log_dirichlet_sample(x, "triton"))
    assert torch.isfinite(f(a)).all()
    assert not torch.equal(f(a), f(a))
    torch.manual_seed(91)
    first = f(a)
    torch.manual_seed(91)
    assert torch.equal(first, f(a))
    s = torch.randn(8, 32, device=DEVICE).log_softmax(-1).requires_grad_()
    w = torch.randn(32, 16, device=DEVICE, requires_grad=True)
    f = torch.compile(lambda x, y: simplex_projection(x, y, "bf16", 4))
    out = f(s, w)
    eager = simplex_projection(s, w, "bf16", 4)
    torch.testing.assert_close(out, eager)
    compiled_grads = torch.autograd.grad(out.sum(), (s, w))
    eager_grads = torch.autograd.grad(eager.sum(), (s, w))
    for actual, expected in zip(compiled_grads, eager_grads):
        torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("precision", ["fp32", "bf16"])
def test_both_model_adapters_preserve_table_gradients(precision):
    from core.models.components import SimplexExpectationInput
    from core.representations.noisy_inputs import SimplexRepresentation

    torch.manual_seed(9)
    state = torch.randn(2, 4, 11, device=DEVICE).log_softmax(-1)
    representation = SimplexRepresentation(
        11,
        7,
        normalize=True,
        scale=2.3,
        projection_precision=precision,
        projection_chunk_size=3,
    ).to(DEVICE)
    expected = (state.exp() @ representation._table()) * 2.3
    actual = representation.to_model_embedding(state, None)
    grads = torch.autograd.grad(actual.square().sum(), representation.embedding)[0]
    ref_grads = torch.autograd.grad(expected.square().sum(), representation.embedding)[
        0
    ]
    assert (actual - expected).norm() / expected.norm() < 0.015
    assert (grads - ref_grads).norm() / ref_grads.norm() < 0.025
    adapter = SimplexExpectationInput(precision, 3)
    table = SimpleNamespace(embedding=representation.embedding)
    out = adapter(state, table)
    assert torch.isfinite(torch.autograd.grad(out.sum(), table.embedding)[0]).all()


@pytest.mark.parametrize("chunk", [0, 127])
def test_chunked_distribution_and_reference_faces(chunk):
    torch.manual_seed(77)
    n = 18000
    tokens = torch.zeros(n, 1, dtype=torch.long, device=DEVICE)
    path = SimpleNamespace(
        time=torch.full((n,), 0.6, device=DEVICE),
        concentration=torch.full((n,), 3.0, device=DEVICE),
    )
    process = DirichletProcess(
        3,
        reference=[0.2, 0.8, 0],
        precision="fp32",
        sampler_backend="triton",
        chunk_size=chunk,
    ).to(DEVICE)
    p = process.corrupt(tokens, path).exp()[:, 0]
    mean = torch.tensor([0.52, 0.48, 0], device=DEVICE)
    torch.testing.assert_close(p.mean(0), mean, atol=0.012, rtol=0)
    torch.testing.assert_close(p.var(0), mean * (1 - mean) / 4, atol=0.008, rtol=0)
    assert (p[:, 2] == 0).all()


def test_corruption_autograd_fallback_and_fp32_bridge():
    process = DirichletProcess(
        3, precision="fp32", sampler_backend="triton", chunk_size=2
    ).to(DEVICE)
    c = torch.ones(2, device=DEVICE, requires_grad=True)
    path = SimpleNamespace(
        time=torch.full((2,), 0.6, device=DEVICE, dtype=torch.float64), concentration=c
    )
    tokens = torch.zeros(2, 3, device=DEVICE, dtype=torch.long)
    state = process.corrupt(tokens, path)
    state.exp()[..., 0].sum().backward()
    assert torch.isfinite(c.grad).all()
    dst = SimpleNamespace(
        time=torch.full((2,), 0.2, device=DEVICE, dtype=torch.float64),
        concentration=c.detach(),
    )
    source = SimpleNamespace(time=path.time, concentration=c.detach())
    for churn in [0.0, 0.3, 1.0]:
        reverse = process.reverse(state.detach(), tokens, dst, source, churn)
        assert reverse.dtype == torch.float32
        torch.testing.assert_close(
            reverse.logsumexp(-1), torch.zeros(2, 3, device=DEVICE), atol=2e-6, rtol=0
        )


@pytest.mark.skipif(DEVICE != "cuda", reason="CUDA required")
def test_compiled_chunked_corruption():
    process = DirichletProcess(
        7, precision="fp32", sampler_backend="triton", chunk_size=3
    ).cuda()
    path = SimpleNamespace(
        time=torch.tensor([0.0, 0.7], device=DEVICE),
        concentration=torch.ones(2, device=DEVICE),
    )
    tokens = torch.zeros(2, 4, dtype=torch.long, device=DEVICE)
    f = torch.compile(process.corrupt)
    state = f(tokens, path)
    torch.testing.assert_close(state[0], process.encode(tokens)[0])
    torch.testing.assert_close(
        state.logsumexp(-1), torch.zeros(2, 4, device=DEVICE), atol=1e-6, rtol=0
    )


@pytest.mark.parametrize("preset", ["view", "model"])
def test_hybrid_config_preserves_fp32_prompt_clamping(preset):
    from pathlib import Path
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    from core.modifiers.hybrid.views import prior

    root = Path(__file__).resolve().parents[2]
    view = (
        OmegaConf.load(root / "configs/hybrid_view/simplex.yaml")
        if preset == "view"
        else OmegaConf.load(root / "configs/model/hybrid_uniform_simplex.yaml").views[1]
    )
    config = OmegaConf.create(
        {
            "task": {"vocab_size": 3},
            "model": {"backbone": {"hidden_size": 8}},
            "view": OmegaConf.to_container(view, resolve=False),
        }
    )
    config.view.process.precision = "fp32"
    config.view.process.sampler_backend = "triton"
    config.view.process.chunk_size = 2
    config.view.representation.projection_precision = "bf16"
    config.view.representation.projection_chunk_size = 2
    view = instantiate(config.view).to(DEVICE)
    tokens = torch.zeros(2, 3, dtype=torch.long, device=DEVICE)
    mask = torch.tensor([[1, 0, 0], [0, 1, 0]], device=DEVICE, dtype=torch.bool)
    coefficients = SimpleNamespace(
        time=torch.ones(2, device=DEVICE), concentration=torch.ones(2, device=DEVICE)
    )
    state = prior(view, tokens, SimpleNamespace(coefficients=coefficients), mask)
    assert state.dtype == torch.float32
    torch.testing.assert_close(
        state[mask], view.representation.clean_state(tokens)[mask]
    )
    embedded = view.representation.to_model_embedding(state, None)
    embedded.sum().backward()
    assert torch.isfinite(view.representation.embedding.grad).all()

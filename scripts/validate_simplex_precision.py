"""Save numerical errors separately from timing runs (single visible GPU)."""

import json
from pathlib import Path
import torch
from core.processes.simplex_math import log_gamma_sample, log_dirichlet_sample
from core.models.simplex_projection import simplex_projection

assert torch.cuda.device_count() == 1
assert torch.cuda.get_device_name(0) == "NVIDIA RTX A6000"
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
result = {"seed": 731, "samples": 200000, "moments": [], "projection": []}
for name, dtype, backend in [
    ("fp64", torch.float64, "torch"),
    ("fp32", torch.float32, "torch"),
    ("fused", torch.float32, "triton"),
]:
    for concentration in [1e-8, 1e-4, 0.1, 1.0, 10.0, 1000.0]:
        torch.manual_seed(731)
        a = torch.full((200000, 3), concentration, device="cuda", dtype=dtype)
        log_g = log_gamma_sample(a, backend).double()
        g = log_g.exp()
        alpha = torch.tensor(concentration, dtype=torch.float64)
        log_mean = torch.digamma(alpha).item()
        log_var = torch.polygamma(1, alpha).item()
        p = log_dirichlet_sample(a, backend).exp().double()
        theory_var = (2 / 9) / (3 * concentration + 1)
        result["moments"].append(
            dict(
                sampler=name,
                concentration=concentration,
                log_gamma_mean=log_g.mean().item(),
                log_gamma_variance=log_g.var().item(),
                log_gamma_theory_mean=log_mean,
                log_gamma_theory_variance=log_var,
                log_gamma_mean_error=abs(log_g.mean().item() - log_mean),
                log_gamma_variance_error=abs(log_g.var().item() - log_var),
                gamma_mean=g.mean().item(),
                gamma_variance=g.var().item(),
                gamma_mean_error=abs(g.mean().item() - concentration),
                gamma_variance_error=abs(g.var().item() - concentration),
                dirichlet_mean=p.mean(0).tolist(),
                dirichlet_variance=p.var(0).tolist(),
                dirichlet_mean_max_error=(p.mean(0) - 1 / 3).abs().max().item(),
                dirichlet_variance_max_error=(p.var(0) - theory_var).abs().max().item(),
                normalization_max_error=(p.sum(-1) - 1).abs().max().item(),
            )
        )
for distribution in ["dense", "sparse"]:
    torch.manual_seed(991)
    shape = (2, 32, 4097)
    state = (
        torch.randn(shape, device="cuda").log_softmax(-1)
        if distribution == "dense"
        else log_dirichlet_sample(torch.full(shape, 1 / 4097, device="cuda"), "triton")
    ).requires_grad_()
    table = torch.randn(4097, 768, device="cuda", requires_grad=True)
    grad = torch.randn(2, 32, 768, device="cuda")
    ref = state.exp() @ table
    expected = torch.autograd.grad(ref, (state, table), grad)
    for precision in ["fp32", "bf16"]:
        for chunk in [0, 7, 32]:
            out = simplex_projection(state, table, precision, chunk)
            grads = torch.autograd.grad(out, (state, table), grad)
            errors = {
                key: {
                    "relative_l2": ((x - y).norm() / y.norm()).item(),
                    "max_absolute": (x - y).abs().max().item(),
                }
                for key, x, y in zip(
                    ["output", "state_gradient", "table_gradient"],
                    (out, *grads),
                    (ref, *expected),
                )
            }
            result["projection"].append(
                dict(
                    distribution=distribution,
                    precision=precision,
                    chunk=chunk,
                    errors=errors,
                )
            )
Path("results/simplex_precision/numerical-errors.json").write_text(
    json.dumps(result, indent=2)
)

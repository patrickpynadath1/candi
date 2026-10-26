"""Optional CUDA FP32 Marsaglia--Tsang sampler; no host rejection loop."""

import torch
import triton
import triton.language as tl


@triton.jit
def _sample(A, Output, Seed, N: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    a = tl.load(A + i, i < N, other=0).to(tl.float32)
    seed = tl.load(Seed).to(tl.uint64)
    safe = tl.where(a > 0, a, 1.0)
    boosted = tl.where(safe < 1.0, safe + 1.0, safe)
    d = boosted - 1.0 / 3.0
    c = tl.rsqrt(9.0 * d)
    pending = (i < N) & (a > 0)
    result = tl.full((BLOCK,), 0.0, tl.float32)
    iteration = 0
    while tl.sum(pending.to(tl.int32), 0) > 0:
        z = tl.randn(seed + 4 * iteration, i)
        u = tl.maximum(tl.rand(seed + 4 * iteration + 1, i), 1.1754943508222875e-38)
        tl.static_assert(z.dtype == tl.float32)
        tl.static_assert(u.dtype == tl.float32)
        base = 1.0 + c * z
        v = tl.maximum(base, 1.1754943508222875e-38)
        v = v * v * v
        take = (
            pending
            & (base > 0)
            & (
                (u < 1.0 - 0.0331 * z * z * z * z)
                | (tl.log(u) < 0.5 * z * z + d * (1.0 - v + tl.log(v)))
            )
        )
        result = tl.where(take, tl.log(d) + tl.log(v), result)
        pending = pending & ~take
        iteration += 1
    boost = tl.log(tl.maximum(tl.rand(seed + 2, i), 1.1754943508222875e-38))
    result += tl.where(safe < 1.0, boost / safe, 0.0)
    tl.store(Output + i, tl.where(a > 0, result, -float("inf")), i < N)


@torch.compiler.disable
def log_gamma_triton(shape):
    """Eager kernel boundary on torch 2.5; graph breaks under torch.compile.

    The CUDA generator supplies a fresh device seed without a CPU .item().
    Resetting torch.manual_seed reproduces draws for a fixed call layout.
    """
    shape = shape.contiguous()
    result = torch.empty_like(shape)
    if shape.numel() == 0:
        return result
    seed = torch.randint(0, 2**62, (), device=shape.device, dtype=torch.int64)
    _sample[(triton.cdiv(shape.numel(), 256),)](shape, result, seed, shape.numel(), 256)
    return result

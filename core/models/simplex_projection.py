"""Row-chunked expectation projection with recomputation in backward.

The full log state (and its gradient, when needed) still occupies B*L*V
storage. Only probability/cast temporaries are bounded by chunk_size*V.
"""

import torch


class _Projection(torch.autograd.Function):
    @staticmethod
    def forward(ctx, state, table, precision, chunk_size):
        ctx.save_for_backward(state, table)
        ctx.precision, ctx.chunk_size = precision, chunk_size
        flat = state.reshape(-1, state.shape[-1])
        dtype = torch.bfloat16 if precision == "bf16" else torch.float32
        out = torch.empty(
            (len(flat), table.shape[1]), device=state.device, dtype=torch.float32
        )
        with torch.autocast(state.device.type, enabled=False):
            weight = table.to(dtype)
            for start in range(0, len(flat), chunk_size):
                p = flat[start : start + chunk_size].float().exp()
                out[start : start + chunk_size] = (p.to(dtype) @ weight).float()
        return out.reshape(*state.shape[:-1], table.shape[1])

    @staticmethod
    def backward(ctx, grad):
        state, table = ctx.saved_tensors
        flat = state.reshape(-1, state.shape[-1])
        grad = grad.reshape(-1, table.shape[1])
        ds = torch.empty_like(flat) if ctx.needs_input_grad[0] else None
        dw = (
            torch.zeros_like(table, dtype=torch.float32)
            if ctx.needs_input_grad[1]
            else None
        )
        dtype = torch.bfloat16 if ctx.precision == "bf16" else torch.float32
        with torch.autocast(state.device.type, enabled=False):
            weight = table.to(dtype)
            for start in range(0, len(flat), ctx.chunk_size):
                end = start + ctx.chunk_size
                p = flat[start:end].float().exp()
                g = grad[start:end].to(dtype)
                if ds is not None:
                    ds[start:end] = (g @ weight.T).float() * p
                if dw is not None:
                    dw.add_((p.to(dtype).T @ g).float())
        return (
            None if ds is None else ds.reshape_as(state),
            None if dw is None else dw.to(table.dtype),
            None,
            None,
        )


def simplex_projection(state, table, precision="legacy", chunk_size=0):
    if precision not in {"legacy", "fp32", "bf16"}:
        raise ValueError("projection precision must be legacy, fp32 or bf16")
    if chunk_size < 0:
        raise ValueError("chunk_size must be nonnegative")
    if precision == "legacy":
        if chunk_size:
            raise ValueError("chunked projection requires fp32 or bf16 precision")
        return state.exp().to(table) @ table
    return _Projection.apply(
        state, table, precision, chunk_size or max(1, state.numel() // state.shape[-1])
    )

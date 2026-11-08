"""Token corruption and rank degradation along a learned CDCD schedule.

Read-only over a checkpoint of a pure continuous recipe (Gaussian embeddings,
simplex, spherical, vMF). For each global time u:

1. t = learned CDCD clock(u); coefficients = the path's schedule at t.
2. Draw clean tokens uniformly over the vocabulary and corrupt them with the
   path's own forward kernel (process.corrupt_state) and learned codebook.
3. Decode: scores = noisy @ unit codebook (simplex: the noisy simplex point).
   - error: P(argmax score != clean token)   (discrete corruption)
   - normalized rank: one-based midrank of the clean token, (rank-1)/(V-1)
     (continuous rank degradation). Independent noise gives 1-1/V and 0.5.

For Gaussian paths it also evaluates the closed-form rank of
core.noise_schedules.embedding_rank.embedding_table_rank on the learned table,
as a check on the Monte Carlo estimate.
"""

import math

from schedule_report import load_trained

# Extra u values near the ends, where error and rank change fastest.
EXTRA_U = (0.002, 0.005, 0.01, 0.02, 0.99, 0.995)
DECODER = "simplex: argmax of noisy simplex point; others: nearest unit learned codeword (dot product)"
RANK = "one-based midrank; normalized (rank-1)/(V-1)"


def codebook(view):
    """Unit-normalized learned token table, or None for the simplex path."""
    import torch

    table = getattr(getattr(view.representation, "state_representation", None), "table", None)
    if table is None:
        return None
    weight = table.weight if hasattr(table, "weight") else table
    return torch.nn.functional.normalize(weight.detach().float(), dim=-1)


def corruption_curve(algorithm, vocab_size, *, samples=8192, points=101, seed=12345):
    """{u, error, rank, error_se, rank_se[, rank_formula], vocab} along the learned clock."""
    import torch

    from core.noise_schedules.embedding_rank import embedding_table_rank

    algorithm.eval()
    clock, view = algorithm.time_sampler, algorithm.views[0]
    device = clock.ema_logits.device
    book = codebook(view)
    if book is not None and book.shape[0] != vocab_size:
        raise ValueError(f"codebook has {book.shape[0]} rows, vocabulary {vocab_size}")
    u = torch.unique(torch.cat([torch.linspace(0, 1, points), torch.tensor(EXTRA_U)])).to(device)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    tokens = torch.randint(vocab_size, (samples, 1), device=device)
    mask = torch.ones_like(tokens, dtype=torch.bool)
    out = {"u": [], "error": [], "rank": [], "error_se": [], "rank_se": []}
    formula = []
    with torch.no_grad():
        t = torch.where((u == 0) | (u == 1), u, clock(u)).clamp(0, 1)
        for a, b in zip(u, t):
            coefficients = view.schedule(b.expand(samples))
            clean = view.representation.encode(tokens) if book is None else book[tokens]
            noisy = view.process.corrupt_state(clean, coefficients, mask)
            scores = noisy[:, 0].float() if book is None else noisy[:, 0].float() @ book.T
            target = scores.gather(1, tokens)
            ranks = (scores > target).sum(-1).float() + 0.5 * ((scores == target).sum(-1) - 1).float()
            error = (scores.argmax(-1) != tokens[:, 0]).float()
            rank = ranks / (vocab_size - 1)
            out["u"].append(float(a))
            out["error"].append(float(error.mean()))
            out["error_se"].append(float(error.std() / math.sqrt(samples)))
            out["rank"].append(float(rank.mean()))
            out["rank_se"].append(float(rank.std() / math.sqrt(samples)))
            sigma = getattr(coefficients, "sigma", None)
            if book is not None and sigma is not None:
                # embedding_table_rank uses z = (1-p) e + p N(0, I); GaussianPath has alpha = 1 - sigma.
                formula.append(float(embedding_table_rank(book, sigma[0].clamp(1e-6, 1))))
    if formula:
        out["rank_formula"] = formula
    out = {key: [round(v, 5) for v in values] for key, values in out.items()}
    out["vocab"] = int(vocab_size)
    return out


def corruption_case(trained, *, samples=8192, points=101, device="cpu"):
    """Corruption curve and metadata for one case's ``trained.json`` record."""
    config, checkpoint, algorithm = load_trained(trained, device)
    vocab = int(config.task.vocab_size)
    return {
        "corruption": corruption_curve(algorithm, vocab, samples=samples, points=points),
        "meta": {"checkpoint": str(trained["checkpoint"]), "step": int(checkpoint["global_step"]), "vocab_size": vocab},
    }

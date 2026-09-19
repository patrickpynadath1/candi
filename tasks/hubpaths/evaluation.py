"""Sample diversity and calibration against the exact ordered-route law."""

from collections import Counter
import numpy as np
from .graph import N_SPECIAL, PAD
from .routes import RouteDistribution, route_shortest
from .verify import Reason, verify_route_batch


def evaluate_samples(
    graph, prompts, samples, hints=None, nll_bits=None, cache_dir=None
):
    """prompts: [P,W] token ids; samples: [P,K,seq_len] complete sequences.

    nll_bits, when supplied, is TOTAL path-slot NLL/bound per prompt, not per token.
    TV is conditioned on validity; invalidity is reported separately.
    """
    from .rsp import RSP

    prompts, samples = np.asarray(prompts), np.asarray(samples)
    if (
        samples.ndim != 3
        or prompts.shape != samples.shape[:1] + (graph.config.n_waypoints,)
        or samples.shape[1] < 1
    ):
        raise ValueError("expected prompts[P,W], samples[P,K,S], K >= 1")
    rsp = RSP(graph, cache_dir=cache_dir)
    records = []
    for i, prompt in enumerate(prompts):
        points = tuple(int(t) - N_SPECIAL for t in prompt)
        row_hints = hints[i] if hints is not None else {}
        distribution = RouteDistribution(rsp, points, row_hints)
        valid, reasons = verify_route_batch(
            graph,
            samples[i],
            prompts=np.repeat(prompt[None, :], len(samples[i]), axis=0),
            require_simple=graph.config.require_simple,
            hints=row_hints,
        )
        valid, reasons = valid.numpy(), reasons.numpy()
        if graph.config.length_mode == "exact":
            lengths = (samples[i, :, graph.config.n_waypoints + 1 :] != PAD).sum(1) - 1
            short = valid & (lengths != graph.config.max_steps)
            valid[short], reasons[short] = False, int(Reason.TOO_LONG)
        paths = [
            tuple(int(t) for t in row[graph.config.n_waypoints + 1 :] if t != PAD)
            for row in samples[i][valid]
        ]
        lengths = [len(path) - 1 for path in paths]
        shortest, _ = route_shortest(graph, points)
        length_hist = np.bincount(lengths, minlength=graph.config.max_steps + 1) / max(
            1, len(paths)
        )
        first_hist = np.bincount(
            [p[1] - N_SPECIAL for p in paths], minlength=graph.n_nodes
        ) / max(1, len(paths))
        record = {
            "shortest_dist": int(shortest),
            "sample_count": len(valid),
            "valid_count": int(valid.sum()),
            "validity": float(valid.mean()),
            "failure_reasons": {
                Reason(int(r)).name: int(n) for r, n in Counter(reasons[~valid]).items()
            },
            "shortest_rate": sum(n == shortest for n in lengths) / len(valid),
            "mean_stretch": float(np.mean(np.asarray(lengths) / shortest))
            if paths
            else None,
            "mean_n_steps": float(np.mean(lengths)) if paths else None,
            "unique_valid@K": len(set(paths)),
            "true_nll_of_samples": float(
                np.mean([-distribution.true_logprob(p) for p in paths])
            )
            if paths
            else None,
            "first_step_tv": float(
                np.abs(first_hist - distribution.first_step_marginal()).sum() / 2
            )
            if paths
            else None,
            "length_tv": float(
                np.abs(length_hist - distribution.length_distribution()).sum() / 2
            )
            if paths
            else None,
            "cond_entropy_bits": distribution.cond_entropy_bits(),
        }
        if nll_bits is not None:
            record["test_nll_bits"] = float(nll_bits[i])
            record["kl_bits"] = float(nll_bits[i] - record["cond_entropy_bits"])
        records.append(record)

    def summarize(rows):
        numeric = [
            "validity",
            "shortest_rate",
            "mean_stretch",
            "mean_n_steps",
            "unique_valid@K",
            "true_nll_of_samples",
            "first_step_tv",
            "length_tv",
            "cond_entropy_bits",
            "test_nll_bits",
            "kl_bits",
        ]
        output = {"prompts": len(rows)}
        for key in numeric:
            values = [r[key] for r in rows if r.get(key) is not None]
            output[key] = None
            if values:
                # Path-level metrics weight prompts by their valid sample count.
                if key in {"mean_stretch", "mean_n_steps", "true_nll_of_samples"}:
                    weights = [r["valid_count"] for r in rows if r.get(key) is not None]
                    output[key] = float(np.average(values, weights=weights))
                else:
                    output[key] = float(np.mean(values))
        reasons = Counter()
        for row in rows:
            reasons.update(row["failure_reasons"])
        output["failure_reasons"] = dict(reasons)
        output["prompts_without_valid_samples"] = sum(
            r["valid_count"] == 0 for r in rows
        )
        return output

    return {
        **summarize(records),
        "by_shortest_distance": {
            str(d): summarize([r for r in records if r["shortest_dist"] == d])
            for d in sorted({r["shortest_dist"] for r in records})
        },
        "exact_walk_statistics": not graph.config.require_simple,
        "tv_conditioning": "valid samples only; null when none valid",
        "records": records,
    }

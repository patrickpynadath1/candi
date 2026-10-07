"""Distribution metrics on log1p library-normalized counts, not paired-cell accuracy.

MMD is the biased squared RBF MMD with a reference-only median squared-distance
bandwidth. W2 is exact equal-weight empirical 2-Wasserstein (Hungarian matching),
using mean squared distance per gene. These conventions are explicit because the
scCANDI draft does not specify its bandwidth, normalization or OT estimator.
"""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist, pdist


def normalize_counts(counts):
    counts = np.asarray(counts, dtype=np.float64)
    return np.log1p(counts * (10000.0 / np.maximum(counts.sum(1, keepdims=True), 1.0)))


def distribution_metrics(real, generated):
    x, y = normalize_counts(real), normalize_counts(generated)
    if x.shape != y.shape or len(x) == 0:
        raise ValueError("Metrics require nonempty equally sized cell matrices")
    xx = cdist(x, x, "sqeuclidean") / x.shape[1]
    yy = cdist(y, y, "sqeuclidean") / x.shape[1]
    xy = cdist(x, y, "sqeuclidean") / x.shape[1]
    distances = pdist(x, "sqeuclidean") / x.shape[1]
    positive = distances[distances > 0]
    bandwidth = float(np.median(positive)) if positive.size else 1.0
    mmd = max(
        0.0,
        float(
            np.exp(-xx / (2 * bandwidth)).mean()
            + np.exp(-yy / (2 * bandwidth)).mean()
            - 2 * np.exp(-xy / (2 * bandwidth)).mean()
        ),
    )
    rows, cols = linear_sum_assignment(xy)
    return {
        "mmd": mmd,
        "wasserstein2": float(np.sqrt(xy[rows, cols].mean())),
        "rbf_bandwidth_squared": bandwidth,
    }


@dataclass(frozen=True)
class CellRecord:
    index: int
    cell_type: int
    generated: list[int]
    ground_truth: list[int]


@dataclass(frozen=True)
class CellAggregate:
    mmd: float
    wasserstein2: float
    rbf_bandwidth_squared: float
    real_zero_fraction: float
    generated_zero_fraction: float
    real_mean_count: float
    generated_mean_count: float
    num_total: int
    per_cell_type: dict
    conditional_macro_mmd: float | None
    conditional_macro_wasserstein2: float | None
    records: list[CellRecord]


class SingleCellMetrics:
    def __init__(self, vocab_size, conditional=False):
        self.vocab_size, self.conditional = vocab_size, conditional
        self.records = []

    def update(self, samples, examples, indices):
        if len(samples) != len(examples) or len(indices) != len(examples):
            raise ValueError("Sample/reference lengths differ")
        for sample, example, index in zip(samples, examples, indices):
            if sample.shape != example["input_ids"].shape:
                raise ValueError("Generated cell shape differs from reference")
            if self.conditional and int(sample[0]) != example["cell_type"]:
                raise ValueError("Sampler changed the fixed cell-type prefix")
            values = sample[int(self.conditional) :].numpy()
            if (values < 0).any() or (values >= self.vocab_size).any():
                raise ValueError(
                    "Generated cell contains mask or out-of-vocabulary tokens"
                )
            target = example["input_ids"][int(self.conditional) :].tolist()
            self.records.append(
                CellRecord(int(index), example["cell_type"], values.tolist(), target)
            )

    def compute(self):
        if not self.records:
            raise ValueError("Cannot score an empty evaluation")
        real = np.array([r.ground_truth for r in self.records])
        generated = np.array([r.generated for r in self.records])
        scores = distribution_metrics(real, generated)
        groups = {}
        if self.conditional:
            labels = np.array([r.cell_type for r in self.records])
            for label in np.unique(labels):
                selected = labels == label
                groups[str(label)] = {
                    **distribution_metrics(real[selected], generated[selected]),
                    "num_total": int(selected.sum()),
                }
        return CellAggregate(
            **scores,
            real_zero_fraction=float((real == 0).mean()),
            generated_zero_fraction=float((generated == 0).mean()),
            real_mean_count=float(real.mean()),
            generated_mean_count=float(generated.mean()),
            num_total=len(real),
            per_cell_type=groups,
            conditional_macro_mmd=float(np.mean([g["mmd"] for g in groups.values()]))
            if groups
            else None,
            conditional_macro_wasserstein2=float(
                np.mean([g["wasserstein2"] for g in groups.values()])
            )
            if groups
            else None,
            records=list(self.records),
        )

"""SOL: a double sliced Wasserstein distance between corpora of token measures.

SOL_theta(P, Q) = sqrt(d) * DSW(T#P, T#Q) (Kornhardt et al., "SOL: Measuring Gaps
between Text Distributions by Double Sliced Wasserstein Metrics", arXiv 2610.06513).
Each text becomes the uniform empirical measure of its per-token hidden states
under a frozen encoder. A first level of ``n_directions`` unit directions on the
sphere projects every measure to one dimension; its quantile function is
averaged over ``n_quantiles`` equal bins. A second level of ``n_projections``
Gaussian-process draws on the quantile grid maps each binned quantile function
to a scalar. Every (direction, draw) pair is one slice: the squared 1D W2
distance between the two corpora's empirical scalar distributions, averaged
over slices, scaled by the hidden dimension and square-rooted. Lower is better.

The estimator follows the reference implementation (github.com/GregorKornhardt/SOL,
Apache-2.0, ``sol_metric._math`` and ``_runtime``) convention for convention:
NumPy-seeded directions and GP draws (device independent), exact empirical
quantile-bin averages, raw (unnormalised) RBF GP draws with lengthscale 0.1,
exact transport for unequal corpus sizes and ``scale = d``. Fixed seeds mean
every scored corpus is compared on identical projections.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

QUANTILE_CONVENTION = "empirical_bin_average_v1"


@dataclass(frozen=True)
class SOLConfig:
    n_directions: int = 1024
    n_projections: int = 1024
    n_quantiles: int = 64
    lengthscale: float | None = 0.1
    seed: int = 0
    gp_seed: int = 0
    scale: float | None = None  # None: the hidden dimension, as in the paper

    def __post_init__(self):
        for name in ("n_directions", "n_projections", "n_quantiles"):
            value = getattr(self, name)
            if isinstance(value, bool) or int(value) != value or value < 1:
                raise ValueError(f"{name} must be a positive integer")
            object.__setattr__(self, name, int(value))
        for name in ("seed", "gp_seed"):
            value = getattr(self, name)
            if isinstance(value, bool) or int(value) != value or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
            object.__setattr__(self, name, int(value))
        if self.lengthscale is not None and not (
            math.isfinite(self.lengthscale) and self.lengthscale > 0
        ):
            raise ValueError("lengthscale must be positive and finite, or None")
        if self.scale is not None and not (math.isfinite(self.scale) and self.scale > 0):
            raise ValueError("scale must be positive and finite")


def directions(count, dimension, seed):
    """First-level directions, uniform on the unit sphere."""
    values = np.random.default_rng(seed).standard_normal((count, dimension))
    return values / np.linalg.norm(values, axis=1, keepdims=True)


def gp_directions(count, quantiles, lengthscale, seed):
    """Raw zero-mean RBF Gaussian-process draws on the quantile-bin midpoints."""
    values = np.random.default_rng(seed).standard_normal((count, quantiles))
    if lengthscale is not None:
        grid = (np.arange(quantiles) + 0.5) / quantiles
        covariance = np.exp(-0.5 * ((grid[:, None] - grid) / lengthscale) ** 2)
        for jitter in (1e-6, 1e-5, 1e-4):
            try:
                factor = np.linalg.cholesky(covariance + jitter * np.eye(quantiles))
                break
            except np.linalg.LinAlgError:
                continue
        else:
            raise np.linalg.LinAlgError("GP covariance factorization failed")
        values = values @ factor.T
    return values


def bin_average_plan(lengths, count):
    """Exact overlaps of ``count`` equal quantile bins with each document's atoms.

    Returns (indices, weights), each (documents, bins, atoms per bin): bin b of a
    document with T sorted atoms averages atoms[indices] with weights that sum to 1/count.
    """
    lengths = np.asarray(lengths, dtype=np.int64)
    if count < 1 or lengths.ndim != 1 or len(lengths) == 0 or np.any(lengths < 1):
        raise ValueError("positive bin count and nonempty documents required")
    t = lengths[:, None, None]
    bins = np.arange(count, dtype=np.int64)[None, :, None]
    width = (int(lengths.max()) + count - 1) // count + 1
    atoms = bins * t // count + np.arange(width, dtype=np.int64)
    mass = np.maximum(
        0, np.minimum((atoms + 1) * count, (bins + 1) * t) - np.maximum(atoms * count, bins * t)
    )
    return np.minimum(atoms, t - 1), mass / t


def transport_grid(nx, ny):
    """Exact atom overlaps of two sorted uniform empirical measures (None if nx == ny)."""
    if nx == ny:
        return None
    edges = np.unique(np.concatenate((np.arange(nx + 1) / nx, np.arange(ny + 1) / ny)))
    weights = np.diff(edges)
    grid = edges[:-1] + weights / 2
    ix = np.clip(np.ceil(grid * nx).astype(np.int64) - 1, 0, nx - 1)
    iy = np.clip(np.ceil(grid * ny).astype(np.int64) - 1, 0, ny - 1)
    return ix, iy, weights


def quantile_features(states, config, *, device=None, documents_per_batch=64):
    """Binned quantile functions, shape (documents, n_directions, n_quantiles).

    ``states`` is a list of (tokens, dimension) arrays or tensors, one per text.
    The result is float32 on the CPU; it does not depend on batching.
    """
    if not states:
        raise ValueError("a corpus needs at least one document")
    dimension = int(states[0].shape[1])
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    u = torch.as_tensor(
        directions(config.n_directions, dimension, config.seed), dtype=torch.float32, device=device
    )
    out = torch.empty(len(states), config.n_directions, config.n_quantiles)
    for start in range(0, len(states), documents_per_batch):
        batch = [torch.as_tensor(s, dtype=torch.float32) for s in states[start : start + documents_per_batch]]
        if any(s.ndim != 2 or s.shape[1] != dimension or len(s) == 0 for s in batch):
            raise ValueError("every document needs a nonempty (tokens, dimension) state matrix")
        lengths = np.array([len(s) for s in batch])
        longest = int(lengths.max())
        values = torch.zeros(len(batch), longest, config.n_directions, device=device)
        for row, s in enumerate(batch):
            values[row, : len(s)] = s.to(device) @ u.T
        valid = torch.arange(longest, device=device)[None, :] < torch.as_tensor(lengths, device=device)[:, None]
        values = values.masked_fill(~valid[..., None], float("inf"))
        ordered = values.transpose(1, 2).sort(dim=-1).values  # (docs, dirs, tokens)
        indices, weights = bin_average_plan(lengths, config.n_quantiles)
        index = torch.as_tensor(indices.reshape(len(batch), 1, -1), device=device)
        selected = torch.gather(ordered, 2, index.expand(-1, config.n_directions, -1))
        selected = selected.reshape(len(batch), config.n_directions, *indices.shape[1:])
        weight = torch.as_tensor(weights, dtype=torch.float32, device=device)[:, None]
        out[start : start + len(batch)] = (selected * weight).sum(-1).cpu()
    return out


def _slice_values(features, g, start, stop, quantiles):
    # (docs, dirs, Q) -> (dirs * projections, docs): one scalar per document and slice.
    block = features[:, start:stop].permute(1, 2, 0)
    return (torch.matmul(g, block) / quantiles).reshape(-1, features.shape[0])


def _w2_rows(x, y, transport):
    """Sum over rows of squared W2 between rows of x and y (already sorted)."""
    if transport is None:
        delta = x.double() - y.double()
        return float((delta * delta).sum() / x.shape[1])
    ix, iy, weights = transport
    delta = x[:, ix].double() - y[:, iy].double()
    return float(((delta * delta) * weights).sum())


def _scores(fx, fy, config, scale, *, device, slice_batch_size, resample=None):
    """Mean over slices of squared W2, optionally for bootstrap index sets."""
    fx, fy = fx.to(device), fy.to(device)
    g = torch.as_tensor(
        gp_directions(config.n_projections, config.n_quantiles, config.lengthscale, config.gp_seed),
        dtype=torch.float32,
        device=device,
    )
    resample = [None] if resample is None else resample
    totals = [0.0] * len(resample)
    nx, ny = fx.shape[0], fy.shape[0]
    transport = transport_grid(nx, ny)
    if transport is not None:
        transport = tuple(torch.as_tensor(a, device=device) for a in transport)
    width = max(1, min(config.n_directions, slice_batch_size // config.n_projections or 1))
    for start in range(0, config.n_directions, width):
        stop = min(start + width, config.n_directions)
        x = _slice_values(fx, g, start, stop, config.n_quantiles)
        y = _slice_values(fy, g, start, stop, config.n_quantiles)
        for k, index in enumerate(resample):
            xs, ys = (x, y) if index is None else (x[:, index], y[:, index])
            totals[k] += _w2_rows(xs.sort(-1).values, ys.sort(-1).values, transport)
    slices = config.n_directions * config.n_projections
    return [math.sqrt(scale * t / slices) for t in totals]


def sol_from_features(fx, fy, config, dimension, *, device=None, slice_batch_size=32768):
    """SOL between two feature tensors from ``quantile_features`` (same config)."""
    if fx.shape[1:] != fy.shape[1:] or fx.shape[1:] != (config.n_directions, config.n_quantiles):
        raise ValueError("feature shapes do not match the SOL configuration")
    scale = config.scale if config.scale is not None else float(dimension)
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    score = _scores(fx, fy, config, scale, device=device, slice_batch_size=slice_batch_size)[0]
    if not math.isfinite(score):
        raise FloatingPointError("nonfinite SOL score")
    return score


def sol_bootstrap(
    fx, fy, config, dimension, *, replicates=100, confidence=0.95, seed=0, device=None,
    slice_batch_size=32768, slices=(128, 128),
):
    """Bootstrap interval of SOL under paired resampling of prompts.

    Generated and reference documents are paired by prompt (same order, same n);
    each replicate draws one set of prompt indices with replacement for both sides.
    ``slices = (directions, projections)`` scores the replicates on the first slices of
    the same seeded families (a sub-estimator; ``None`` uses all of them) and shifts
    the replicate distribution to the full estimate: [full + q_lo(rep - sub),
    full + q_hi(rep - sub)], where sub is the sub-estimator on the original sample.
    This keeps the prompt-resampling spread at a fraction of the full cost.
    """
    if fx.shape[0] != fy.shape[0]:
        raise ValueError("a paired bootstrap needs one reference per generated text")
    if replicates < 1 or not 0 < confidence < 1:
        raise ValueError("replicates must be positive and confidence in (0, 1)")
    scale = config.scale if config.scale is not None else float(dimension)
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    full = _scores(fx, fy, config, scale, device=device, slice_batch_size=slice_batch_size)[0]
    sub_config = config
    if slices is not None:
        directions_, projections = (min(int(slices[0]), config.n_directions),
                                    min(int(slices[1]), config.n_projections))
        # The first k seeded draws of each family are the k-draw family: a true subset.
        sub_config = SOLConfig(directions_, projections, config.n_quantiles,
                               config.lengthscale, config.seed, config.gp_seed, config.scale)
        fx, fy = fx[:, :directions_], fy[:, :directions_]
    generator = np.random.default_rng(seed)
    n = fx.shape[0]
    indices = [None] + [
        torch.as_tensor(generator.integers(0, n, n), device=device) for _ in range(replicates)
    ]
    scores = _scores(
        fx, fy, sub_config, scale, device=device, slice_batch_size=slice_batch_size,
        resample=indices,
    )
    tail = (1 - confidence) / 2
    low, high = np.quantile(np.array(scores[1:]) - scores[0], [tail, 1 - tail])
    return {"low": float(max(0.0, full + low)), "high": float(full + high),
            "replicates": replicates, "confidence": confidence,
            "resampling": "paired_prompts",
            "slices": [sub_config.n_directions, sub_config.n_projections]}


# Paper-pinned encoder revisions (the reference implementation's MODELS.md).
PAPER_REVISIONS = {"gpt2-large": "32b71b12589c2f8d625668d2335a01cac3249519"}


class TextEncoder:
    """Frozen final-layer (``last_hidden_state``) token states, padding excluded.

    Loads the Hugging Face model on first use; tests pass ``model`` and
    ``tokenizer`` objects instead. An empty text (a generation that stopped at
    once) is encoded as the tokenizer's EOS token so it remains a valid measure.
    """

    auto_class = "AutoModel"  # subclasses load e.g. a causal LM head the same way

    def __init__(
        self,
        model_name="gpt2-large",
        *,
        revision=None,
        device="auto",
        dtype="auto",
        batch_size=16,
        max_tokens_per_batch=8192,
        max_length="auto",
        local_files_only=False,
        model=None,
        tokenizer=None,
    ):
        self.model_name = model_name
        self.revision = revision if revision is not None else PAPER_REVISIONS.get(model_name)
        self.device, self.dtype = device, dtype
        self.batch_size, self.max_tokens_per_batch = int(batch_size), int(max_tokens_per_batch)
        self.max_length, self.local_files_only = max_length, local_files_only
        self.model, self.tokenizer = model, tokenizer
        self._ready = False

    def _load(self):
        if self._ready:
            return
        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.dtype == "auto":  # The paper encodes in FP16 on GPUs.
            self.dtype = "float16" if str(self.device).startswith("cuda") else "float32"
        if self.model is None or self.tokenizer is None:
            import transformers

            options = {"revision": self.revision, "local_files_only": self.local_files_only}
            self.tokenizer = transformers.AutoTokenizer.from_pretrained(self.model_name, **options)
            self.model = getattr(transformers, self.auto_class).from_pretrained(
                self.model_name, dtype=getattr(torch, self.dtype), **options
            )
        self.model = self.model.to(self.device, getattr(torch, self.dtype)).eval()
        positions = getattr(self.model.config, "n_positions", None) or getattr(
            self.model.config, "max_position_embeddings", None
        )
        self.limit = positions if self.max_length == "auto" else self.max_length
        self._ready = True

    @property
    def provenance(self):
        return {
            "model": self.model_name,
            "revision": self.revision,
            "layer": "last_hidden_state",
            "max_length": self.max_length,
            "dtype": self.dtype,
            "padding": "excluded",
            "special_tokens": "included",
            "empty_text": "eos_token",
        }

    def _ids(self, text):
        if not text.strip():
            return [self.tokenizer.eos_token_id]
        ids = self.tokenizer(text, truncation=self.limit is not None, max_length=self.limit)["input_ids"]
        return ids or [self.tokenizer.eos_token_id]

    @torch.no_grad()
    def __call__(self, texts):
        self._load()
        items = [self._ids(text) for text in texts]
        states = [None] * len(items)
        order = sorted(range(len(items)), key=lambda i: len(items[i]))
        # Padding is masked out of attention and dropped from the states.
        pad = self.tokenizer.pad_token_id
        pad = self.tokenizer.eos_token_id if pad is None else pad
        start = 0
        while start < len(order):
            end = start + 1
            while (
                end < len(order)
                and end - start < self.batch_size
                and (end - start + 1) * len(items[order[end]]) <= self.max_tokens_per_batch
            ):
                end += 1
            rows = order[start:end]
            longest = max(len(items[i]) for i in rows)
            ids = torch.full((len(rows), longest), pad, dtype=torch.long)
            mask = torch.zeros_like(ids)
            for r, i in enumerate(rows):
                ids[r, : len(items[i])] = torch.tensor(items[i])
                mask[r, : len(items[i])] = 1
            hidden = self.model(
                input_ids=ids.to(self.device), attention_mask=mask.to(self.device)
            ).last_hidden_state
            for r, i in enumerate(rows):
                states[i] = hidden[r, : len(items[i])].float().cpu()
            start = end
        return states


class SOLScorer:
    """Score generated texts against references; reference features are cached.

    The cache key covers the encoder provenance, the SOL configuration and the
    multiset of reference texts (SOL does not depend on corpus order), so every
    trial scored against the same references reuses one featurization.
    """

    def __init__(
        self,
        encoder=None,
        *,
        n_directions=1024,
        n_projections=1024,
        n_quantiles=64,
        lengthscale=0.1,
        seed=0,
        gp_seed=0,
        scale=None,
        bootstrap=0,
        bootstrap_confidence=0.95,
        bootstrap_seed=0,
        bootstrap_slices=(128, 128),
        cache_dir=None,
        device="auto",
        slice_batch_size=32768,
    ):
        self.config = SOLConfig(
            n_directions, n_projections, n_quantiles, lengthscale, seed, gp_seed, scale
        )
        if isinstance(encoder, dict) or encoder is None:
            encoder = TextEncoder(**(encoder or {}))
        self.encoder = encoder
        self.bootstrap, self.bootstrap_confidence = int(bootstrap), float(bootstrap_confidence)
        self.bootstrap_seed = int(bootstrap_seed)
        self.bootstrap_slices = None if bootstrap_slices is None else tuple(bootstrap_slices)
        if cache_dir is None and os.environ.get("SDDC_STORAGE_ROOT"):
            cache_dir = Path(os.environ["SDDC_STORAGE_ROOT"]) / "sol_cache"
        self.cache_dir = None if cache_dir is None else Path(cache_dir).expanduser()
        self.device = device
        self.slice_batch_size = int(slice_batch_size)

    def _device(self):
        if self.device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return self.device

    def signature(self):
        return {"config": asdict(self.config), "encoder": self.encoder.provenance,
                "quantile_convention": QUANTILE_CONVENTION}

    def features(self, texts, *, cache=False):
        """(features in input order, dimension); ``cache`` reuses a reference featurization."""
        texts = list(texts)
        # Encode in sorted order: the cached tensor is then independent of input order.
        order = sorted(range(len(texts)), key=texts.__getitem__)
        path = None
        if cache and self.cache_dir is not None:
            digests = sorted(hashlib.sha256(t.encode()).hexdigest() for t in texts)
            key = hashlib.sha256(
                json.dumps({"signature": self.signature(), "texts": digests}, sort_keys=True).encode()
            ).hexdigest()[:32]
            path = self.cache_dir / f"references-{key}.pt"
        if path is not None and path.exists():
            saved = torch.load(path, map_location="cpu", weights_only=True)
            ordered, dimension = saved["features"], int(saved["dimension"])
        else:
            states = self.encoder([texts[i] for i in order])
            ordered = quantile_features(states, self.config, device=self._device())
            dimension = int(states[0].shape[1])
            if path is not None:
                path.parent.mkdir(parents=True, exist_ok=True)
                temporary = path.with_suffix(f".{os.getpid()}.tmp")
                torch.save({"features": ordered, "dimension": dimension}, temporary)
                temporary.replace(path)
        features = torch.empty_like(ordered)
        features[torch.as_tensor(order)] = ordered
        return features, dimension

    def score(self, generated, references):
        if len(generated) != len(references) or not generated:
            raise ValueError("SOL needs one reference per generated text")
        fy, dimension = self.features(references, cache=True)
        fx, _ = self.features(generated)
        device = self._device()
        result = {
            "sol": sol_from_features(
                fx, fy, self.config, dimension, device=device,
                slice_batch_size=self.slice_batch_size,
            )
        }
        if self.bootstrap:
            # Rows are paired: generated text i continues the prompt of reference i.
            result["sol_interval"] = sol_bootstrap(
                fx, fy, self.config, dimension, replicates=self.bootstrap,
                confidence=self.bootstrap_confidence, seed=self.bootstrap_seed,
                device=device, slice_batch_size=self.slice_batch_size,
                slices=self.bootstrap_slices,
            )
        return result

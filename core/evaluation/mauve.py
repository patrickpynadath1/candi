"""MAUVE (Pillutla et al., 2021) of generated continuations against references.

Each text is featurised as the final-layer hidden state of the LAST token of
prompt + continuation under a frozen causal LM (MAUVE's standard featurisation;
default TinyStories-33M at a pinned revision). The two feature clouds are compared
the way mauve-text 0.4.0 ``compute_mauve`` does by default:

- stack (generated, reference), L2-normalise each row, fit PCA on the union and keep
  the leading components that explain ``pca_variance`` (0.9) of the variance;
- k-means with ``num_buckets`` clusters on the union (5 restarts, 500 iterations,
  random-point initialisation, seed + 2), then one histogram per side;
- the divergence frontier at mixture weights ``linspace(1e-6, 1 - 1e-6, 25)``, each
  KL mapped through exp(-``scaling_factor`` * KL) with scaling factor 5, and MAUVE is
  the area under that curve (averaged over both axis orders, as mauve-text does).

``num_buckets="auto"`` is mauve-text's rule, max(2, round(min(n_p, n_q) / 10)); tuning
fixes the count so scores are comparable across trials at one sample size. Differences
from mauve-text, none of which changes the estimator:

- k-means is scikit-learn's Lloyd (``init="random"``, ``tol=0``), not faiss, so cluster
  assignments, and hence scores, differ by about mauve-text's own run-to-run noise;
- rows are put in a canonical order before clustering, so the score does not depend on
  sample order (in mauve-text it does, through the k-means initialisation);
- a text longer than the LM's context keeps its LAST tokens (it is truncated from the
  left of the prompt, so the featurised final token is the story's own ending), while
  mauve-text keeps the first ``max_text_length`` tokens. TinyStories stories are far
  shorter than either context (2,048 tokens for TinyStories-33M, 1,024 for GPT-2), so
  this never triggers there.

References are de-duplicated by example index: when several seeds continue one prompt
(pooled seeds), its reference counts once, as in the stability study (4,096 samples vs
2,048 references). Reference features are cached per (feature model, revision, index
set) under ``cache_dir``.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np
import torch

from .sol import TextEncoder

# Pinned revisions of the feature models used by the TinyStories evaluation.
MAUVE_REVISIONS = {
    "roneneldan/TinyStories-33M": "2ad0a164221b7c4d21cac7c46aec74f6f98dbfc8",
    "gpt2-large": "32b71b12589c2f8d625668d2335a01cac3249519",
}
REFERENCE_IMPLEMENTATION = "mauve-text 0.4.0 compute_mauve defaults; sklearn k-means"


def auto_buckets(n_p, n_q):
    """mauve-text's default bucket count."""
    return max(2, int(round(min(n_p, n_q) / 10)))


def _canonical(features):
    """Rows in an order fixed by their values, so the score ignores input order."""
    features = np.ascontiguousarray(features, dtype=np.float32)
    order = sorted(range(len(features)), key=lambda i: features[i].tobytes())
    return features[order]


def _kl(p, q):
    if np.any((p != 0) & (q == 0)):
        return np.inf
    keep = (p != 0) & (q != 0)
    return float(np.sum(p[keep] * np.log(p[keep] / q[keep])))


def mauve_from_histograms(p, q, scaling_factor=5, curve_points=25):
    """Area under the divergence frontier of two histograms (mauve-text's formula)."""
    from sklearn.metrics import auc

    curve = [[0, np.inf]]
    for w in np.linspace(1e-6, 1 - 1e-6, curve_points):
        r = w * p + (1 - w) * q
        curve.append([_kl(q, r), _kl(p, r)])
    curve.append([np.inf, 0])
    x, y = np.exp(-scaling_factor * np.asarray(curve)).T
    first, second = np.argsort(x), np.argsort(y)
    return float(0.5 * (auc(x[first], y[first]) + auc(y[second], x[second])))


def histograms(p_features, q_features, num_buckets, *, pca_variance=0.9, seed=25,
               kmeans_n_init=5, kmeans_max_iter=500, kmeans_seeds=1):
    """Normalised bucket histograms [(p, q), ...] from one joint PCA and k-means runs.

    Run j uses k-means seed ``seed + j + 2`` (mauve-text's offset) on the same projection.
    """
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import normalize

    if not 0 < pca_variance < 1:
        raise ValueError("pca_variance must lie in (0, 1)")
    q_features, p_features = _canonical(q_features), _canonical(p_features)
    data = normalize(np.vstack([q_features, p_features]), norm="l2", axis=1)
    if num_buckets > len(data):
        raise ValueError(f"{num_buckets} MAUVE buckets exceed the {len(data)} points")
    pca = PCA(n_components=None, whiten=False, random_state=seed + 1).fit(data)
    keep = int(np.argmax(np.cumsum(pca.explained_variance_ratio_) >= pca_variance)) + 1
    data = pca.transform(data)[:, :keep].astype(np.float32)
    count, out = len(q_features), []
    for j in range(kmeans_seeds):
        labels = KMeans(
            num_buckets, init="random", n_init=kmeans_n_init, max_iter=kmeans_max_iter,
            tol=0.0, random_state=seed + j + 2, algorithm="lloyd",
        ).fit(data).labels_
        q = np.bincount(labels[:count], minlength=num_buckets).astype(np.float64)
        p = np.bincount(labels[count:], minlength=num_buckets).astype(np.float64)
        out.append((p / p.sum(), q / q.sum()))
    return out


def mauve_values(p_features, q_features, *, num_buckets="auto", pca_variance=0.9, seed=25,
                 scaling_factor=5, kmeans_n_init=5, kmeans_max_iter=500, curve_points=25,
                 kmeans_seeds=1):
    """([MAUVE at MAUVE seeds seed, ..., seed + kmeans_seeds - 1], bucket count)."""
    if len(p_features) == 0 or len(q_features) == 0:
        raise ValueError("MAUVE needs two nonempty feature sets")
    buckets = auto_buckets(len(p_features), len(q_features)) if num_buckets == "auto" else num_buckets
    if isinstance(buckets, bool) or not isinstance(buckets, int) or buckets < 2:
        raise ValueError('num_buckets must be "auto" or an integer >= 2')
    if isinstance(kmeans_seeds, bool) or not isinstance(kmeans_seeds, int) or kmeans_seeds < 1:
        raise ValueError("kmeans_seeds must be a positive integer")
    pairs = histograms(p_features, q_features, buckets, pca_variance=pca_variance, seed=seed,
                       kmeans_n_init=kmeans_n_init, kmeans_max_iter=kmeans_max_iter,
                       kmeans_seeds=kmeans_seeds)
    return [mauve_from_histograms(p, q, scaling_factor, curve_points) for p, q in pairs], buckets


def mauve_from_features(p_features, q_features, **settings):
    """(MAUVE averaged over ``kmeans_seeds`` k-means seeds, bucket count)."""
    values, buckets = mauve_values(p_features, q_features, **settings)
    return float(np.mean(values)), buckets


def load_pretrained(auto, name, options):
    """``auto.from_pretrained``, or its weights loaded by hand for a ``.bin``-only repo.

    transformers 5.8 refuses ``torch.load`` on torch < 2.6 (e.g. TinyStories-33M, which
    ships only ``pytorch_model.bin``); the state dict is then read with
    ``weights_only=True``, prefixes adjusted for a base model, and tied output weights
    re-tied.
    """
    import transformers

    try:
        return auto.from_pretrained(name, **options)
    except ValueError as error:
        if "torch.load" not in str(error):
            raise
    from huggingface_hub import hf_hub_download

    model = auto.from_config(transformers.AutoConfig.from_pretrained(name, **options))
    state = torch.load(hf_hub_download(name, "pytorch_model.bin", **options),
                       map_location="cpu", weights_only=True)
    prefix = model.base_model_prefix + "."
    if not any(key.startswith(prefix) for key in model.state_dict()):
        state = {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}
    missing, _ = model.load_state_dict(state, strict=False)
    model.tie_weights()
    tied = {"lm_head.weight"} if getattr(model.config, "tie_word_embeddings", True) else set()
    if set(missing) - tied:
        raise ValueError(f"{name}: weights missing from pytorch_model.bin: {sorted(set(missing) - tied)}")
    return model


class MauveFeaturizer(TextEncoder):
    """Final-layer hidden state of each text's last token, as a (texts, hidden) array."""

    def __init__(self, model_name="roneneldan/TinyStories-33M", *, revision=None,
                 dtype="float32", batch_size=128, max_tokens_per_batch=16384, **options):
        super().__init__(model_name, revision=revision, dtype=dtype, batch_size=batch_size,
                         max_tokens_per_batch=max_tokens_per_batch, **options)
        if revision is None:
            self.revision = MAUVE_REVISIONS.get(model_name)

    def _load(self):
        if self.model is None or self.tokenizer is None:
            import transformers

            options = {"revision": self.revision, "local_files_only": self.local_files_only}
            self.tokenizer = transformers.AutoTokenizer.from_pretrained(self.model_name, **options)
            self.model = load_pretrained(transformers.AutoModel, self.model_name, options)
        super()._load()

    @property
    def provenance(self):
        return {"model": self.model_name, "revision": self.revision, "layer": "last_hidden_state",
                "token": "last", "max_length": self.max_length, "truncation": "left",
                "dtype": self.dtype}

    @torch.no_grad()
    def __call__(self, texts):
        self._load()
        eos = self.tokenizer.eos_token_id
        items = [(self.tokenizer(t)["input_ids"] if t else []) or [eos] for t in texts]
        if self.limit is not None:
            items = [ids[-self.limit:] for ids in items]
        out = np.zeros((len(items), self.model.config.hidden_size), dtype=np.float32)
        order = sorted(range(len(items)), key=lambda i: len(items[i]))
        start = 0
        while start < len(order):
            end = start + 1
            while (end < len(order) and end - start < self.batch_size
                   and (end - start + 1) * len(items[order[end]]) <= self.max_tokens_per_batch):
                end += 1
            rows = order[start:end]
            ids = torch.full((len(rows), len(items[rows[-1]])), eos, dtype=torch.long)
            mask = torch.zeros_like(ids)
            for r, i in enumerate(rows):
                ids[r, : len(items[i])] = torch.tensor(items[i])
                mask[r, : len(items[i])] = 1
            hidden = self.model(input_ids=ids.to(self.device),
                                attention_mask=mask.to(self.device)).last_hidden_state
            last = torch.tensor([len(items[i]) - 1 for i in rows], device=hidden.device)
            out[rows] = hidden[torch.arange(len(rows), device=hidden.device), last].float().cpu().numpy()
            start = end
        return out


class MauveScorer:
    """MAUVE of generated continuations against the references of the same prompts."""

    def __init__(self, featurizer=None, *, num_buckets="auto", pca_variance=0.9, seed=25,
                 scaling_factor=5, kmeans_n_init=5, kmeans_max_iter=500, curve_points=25,
                 kmeans_seeds=1, bootstrap=0, bootstrap_confidence=0.95, bootstrap_seed=0,
                 cache_dir=None):
        if isinstance(featurizer, dict) or featurizer is None:
            featurizer = MauveFeaturizer(**(featurizer or {}))
        self.featurizer = featurizer
        self.settings = {
            "num_buckets": num_buckets, "pca_variance": float(pca_variance), "seed": int(seed),
            "scaling_factor": float(scaling_factor), "kmeans_n_init": int(kmeans_n_init),
            "kmeans_max_iter": int(kmeans_max_iter), "curve_points": int(curve_points),
            "kmeans_seeds": int(kmeans_seeds),
        }
        self.bootstrap, self.bootstrap_confidence = int(bootstrap), float(bootstrap_confidence)
        self.bootstrap_seed = int(bootstrap_seed)
        if cache_dir is None and os.environ.get("SDDC_STORAGE_ROOT"):
            cache_dir = Path(os.environ["SDDC_STORAGE_ROOT"]) / "mauve_cache"
        self.cache_dir = None if cache_dir is None else Path(cache_dir).expanduser()

    def signature(self):
        return {"features": self.featurizer.provenance, **self.settings,
                "reference_implementation": REFERENCE_IMPLEMENTATION}

    def reference_features(self, keys, texts):
        """Features of distinct references, cached by featuriser and (index, text) set."""
        path = None
        if self.cache_dir is not None:
            digest = hashlib.sha256(json.dumps(
                {"features": self.featurizer.provenance,
                 "references": sorted([k, hashlib.sha256(t.encode()).hexdigest()]
                                      for k, t in zip(keys, texts))},
                sort_keys=True).encode()).hexdigest()[:32]
            path = self.cache_dir / f"references-{digest}.npz"
        if path is not None and path.exists():
            saved = np.load(path)
            row = {int(k): i for i, k in enumerate(saved["keys"])}
            return saved["features"][[row[int(k)] for k in keys]]
        features = self.featurizer(texts)
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_name(f"{path.stem}.{os.getpid()}.tmp.npz")
            np.savez(temporary, keys=np.asarray(keys, dtype=np.int64), features=features)
            temporary.replace(path)
        return features

    def score(self, prompts, generated, references, indices):
        if not (len(prompts) == len(generated) == len(references) == len(indices)) or not prompts:
            raise ValueError("MAUVE needs one prompt, reference and index per generated text")
        indices = [int(i) for i in indices]
        first = {}
        for row, index in enumerate(indices):
            first.setdefault(index, row)
        keys = sorted(first)
        p = self.reference_features(keys, [prompts[first[k]] + references[first[k]] for k in keys])
        q = self.featurizer([a + b for a, b in zip(prompts, generated)])
        values, buckets = mauve_values(p, q, **self.settings)
        value = float(np.mean(values))
        if not math.isfinite(value):
            raise FloatingPointError("nonfinite MAUVE")
        signature = {**self.signature(), "num_buckets_used": buckets,
                     "n_references": len(p), "n_generated": len(q)}
        result = {"mauve": value, "mauve_kmeans_values": values,
                  "mauve_kmeans_sd": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                  "mauve_signature": signature}
        if self.bootstrap:
            # Paired: resample prompts; each keeps its reference and all its generations.
            rows = {k: [r for r, i in enumerate(indices) if i == k] for k in keys}
            generator = np.random.default_rng(self.bootstrap_seed)
            settings = {**self.settings, "num_buckets": buckets}
            replicates = []
            for _ in range(self.bootstrap):
                draw = generator.integers(0, len(keys), len(keys))
                replicates.append(mauve_from_features(
                    p[draw], q[[r for j in draw for r in rows[keys[j]]]], **settings)[0])
            tail = (1 - self.bootstrap_confidence) / 2
            low, high = np.quantile(replicates, [tail, 1 - tail])
            result["mauve_interval"] = {
                "low": float(low), "high": float(high), "replicates": self.bootstrap,
                "confidence": self.bootstrap_confidence, "resampling": "paired_prompts",
            }
        return result

"""ref_w1: a cheap distributional score from per-document coding rates and likelihoods.

Two per-document signals of a continuation y given its prompt p, each compared between
the real continuations of a set of prompts and the generated ones by the exact 1-D
Wasserstein-1 distance (equal weight per document, unequal sizes handled exactly):

- **reference-conditioned coding rate** (the compression-as-evaluation metric): with a
  Zstandard dictionary built from real reference text disjoint from the evaluated
  documents, r(p, y) = (bits(p + y) - bits(p)) / bytes(y), each document an independent
  frame, so the prompt's own cost is removed;
- **likelihood**: the token-averaged negative log-likelihood (nats per token) of y given
  p under a frozen causal language model (default GPT-2 Large).

Each W1 is divided by its floor, the W1 between two fixed halves of the real list (the
spread expected between two real samples of this size), and the two are summed:

    ref_w1 = W1_rate / floor_rate + W1_nll / floor_nll        (lower is better)

Defaults follow the compression metric: Zstandard level 10, window log 23 and a 64 KiB
raw-content dictionary. An empty continuation is scored as a single newline
(compression) or the scorer's EOS token (likelihood), so it stays a valid, distant
document.
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


def wasserstein_1(a, b):
    """Exact W1 between two empirical 1-D measures with equal weights per point."""
    a, b = np.sort(np.asarray(a, dtype=np.float64)), np.sort(np.asarray(b, dtype=np.float64))
    if len(a) == 0 or len(b) == 0:
        raise ValueError("W1 needs two nonempty lists")
    values = np.concatenate([a, b])
    values.sort(kind="mergesort")
    gaps = np.diff(values)
    # |F_a - F_b| is constant between consecutive support points.
    fa = np.searchsorted(a, values[:-1], side="right") / len(a)
    fb = np.searchsorted(b, values[:-1], side="right") / len(b)
    return float(np.sum(np.abs(fa - fb) * gaps))


def build_dictionary(texts, size=65536):
    """Raw-content dictionary: the first ``size`` bytes of the fit texts, in order."""
    data = "\n\n".join(texts).encode("utf-8")[:size]
    if len(data) < size:
        raise ValueError(f"fit split gives {len(data)} bytes, fewer than the {size}-byte dictionary")
    return data


class RateCoder:
    """Per-document Zstandard coding rates with a fixed raw-content dictionary."""

    def __init__(self, dictionary, level=10, window_log=23):
        import zstandard

        self.dictionary, self.level, self.window_log = bytes(dictionary), int(level), int(window_log)
        self._dict = zstandard.ZstdCompressionDict(
            self.dictionary, dict_type=zstandard.DICT_TYPE_RAWCONTENT
        )
        params = zstandard.ZstdCompressionParameters.from_level(
            self.level, window_log=self.window_log, write_content_size=0,
            write_checksum=0, write_dict_id=0,
        )
        self._compressor = zstandard.ZstdCompressor(dict_data=self._dict, compression_params=params)

    @property
    def sha256(self):
        return hashlib.sha256(self.dictionary).hexdigest()

    def bits(self, data):
        """Compressed size in bits of one independent frame."""
        if isinstance(data, str):
            data = data.encode("utf-8")
        return 8 * len(self._compressor.compress(data))

    def rate(self, text):
        data = text.encode("utf-8") or b"\n"
        return self.bits(data) / len(data)

    def conditional_rate(self, prompt, continuation):
        """Bits per byte of ``continuation`` once ``prompt`` is paid for."""
        head = prompt.encode("utf-8")
        tail = continuation.encode("utf-8") or b"\n"
        return (self.bits(head + tail) - self.bits(head)) / len(tail)


class CausalScorer(TextEncoder):
    """Token-averaged NLL of continuations given prompts under a frozen causal LM."""

    auto_class = "AutoModelForCausalLM"

    @torch.no_grad()
    def nll(self, prompts, continuations):
        self._load()
        eos = self.tokenizer.eos_token_id
        pad = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else eos
        items = []
        for prompt, continuation in zip(prompts, continuations):
            head = self.tokenizer(prompt)["input_ids"] if prompt else []
            tail = self.tokenizer(continuation)["input_ids"] if continuation.strip() else []
            tail = tail or [eos]
            if self.limit is not None and len(head) + len(tail) > self.limit:
                tail = tail[: self.limit - 1]
                head = head[-(self.limit - len(tail)):]
            if not head:  # the first token has no prediction; condition on EOS
                head = [eos]
            items.append((head, tail))
        result = [0.0] * len(items)
        order = sorted(range(len(items)), key=lambda i: len(items[i][0]) + len(items[i][1]))
        for start in range(0, len(order), self.batch_size):
            rows = order[start : start + self.batch_size]
            longest = max(len(items[i][0]) + len(items[i][1]) for i in rows)
            ids = torch.full((len(rows), longest), pad, dtype=torch.long)
            mask = torch.zeros_like(ids)
            for r, i in enumerate(rows):
                sequence = items[i][0] + items[i][1]
                ids[r, : len(sequence)] = torch.tensor(sequence)
                mask[r, : len(sequence)] = 1
            logits = self.model(input_ids=ids.to(self.device), attention_mask=mask.to(self.device)).logits
            logp = torch.log_softmax(logits.float(), -1).cpu()
            for r, i in enumerate(rows):
                head, tail = items[i]
                positions = torch.arange(len(head) - 1, len(head) + len(tail) - 1)
                target = torch.tensor(tail)
                result[i] = float(-logp[r, positions, target].mean())
        return result

    @property
    def provenance(self):
        return {**super().provenance, "layer": "lm_head", "quantity": "mean_nll_nats_per_token"}


def halves(keys):
    """Two fixed halves of a list, by the hash of each item's key (order independent)."""
    order = sorted(range(len(keys)), key=lambda i: hashlib.sha256(keys[i].encode()).hexdigest())
    half = len(order) // 2
    return order[:half], order[half:]


class RefW1Scorer:
    """Score generated continuations against the real continuations of the same prompts.

    ``fit_texts`` (real stories disjoint from every evaluated prompt) build the
    dictionary; ``dictionary_sha256`` pins it. ``nll`` configures the scorer LM
    (``None`` drops the likelihood term). Real-side signals and floors are cached per
    reference set under ``cache_dir``.
    """

    def __init__(
        self,
        fit_texts,
        *,
        dictionary_bytes=65536,
        level=10,
        window_log=23,
        dictionary_sha256=None,
        nll=None,
        bootstrap=0,
        bootstrap_confidence=0.95,
        bootstrap_seed=0,
        cache_dir=None,
    ):
        self.coder = RateCoder(build_dictionary(fit_texts, dictionary_bytes), level, window_log)
        if dictionary_sha256 is not None and self.coder.sha256 != dictionary_sha256:
            raise ValueError(
                f"ref_w1 dictionary sha256 {self.coder.sha256} differs from the pinned "
                f"{dictionary_sha256}: the fit split or its data changed"
            )
        if isinstance(nll, dict):
            nll = CausalScorer(**nll)
        self.scorer = nll
        self.bootstrap, self.bootstrap_confidence = int(bootstrap), float(bootstrap_confidence)
        self.bootstrap_seed = int(bootstrap_seed)
        if cache_dir is None and os.environ.get("SDDC_STORAGE_ROOT"):
            cache_dir = Path(os.environ["SDDC_STORAGE_ROOT"]) / "ref_w1_cache"
        self.cache_dir = None if cache_dir is None else Path(cache_dir).expanduser()

    def signature(self):
        return {
            "dictionary_sha256": self.coder.sha256,
            "dictionary_bytes": len(self.coder.dictionary),
            "level": self.coder.level,
            "window_log": self.coder.window_log,
            "nll": None if self.scorer is None else self.scorer.provenance,
        }

    def signals(self, prompts, continuations):
        """{"rate": [...], "nll": [...]} per document, in input order."""
        out = {"rate": [self.coder.conditional_rate(p, c) for p, c in zip(prompts, continuations)]}
        if self.scorer is not None:
            out["nll"] = self.scorer.nll(prompts, continuations)
        return out

    def reference(self, prompts, references):
        """Real-side signals and floors, cached by the multiset of (prompt, reference)."""
        pairs = [json.dumps([p, r]) for p, r in zip(prompts, references)]
        path = None
        if self.cache_dir is not None:
            key = hashlib.sha256(json.dumps(
                {"signature": self.signature(), "pairs": sorted(hashlib.sha256(x.encode()).hexdigest() for x in pairs)},
                sort_keys=True).encode()).hexdigest()[:32]
            path = self.cache_dir / f"reference-{key}.json"
        if path is not None and path.exists():
            saved = json.loads(path.read_text())
            by_pair = dict(zip(saved["pairs"], zip(*[saved["signals"][k] for k in saved["names"]])))
            signals = {k: [by_pair[x][j] for x in pairs] for j, k in enumerate(saved["names"])}
            return signals, saved["floors"]
        signals = self.signals(prompts, references)
        first, second = halves(pairs)
        if not first or not second:
            raise ValueError("ref_w1 needs at least two reference documents for its floors")
        floors = {
            k: wasserstein_1([v[i] for i in first], [v[i] for i in second]) for k, v in signals.items()
        }
        if any(f <= 0 for f in floors.values()):
            raise ValueError(f"degenerate ref_w1 floor {floors}; use more reference documents")
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            names = sorted(signals)
            temporary = path.with_suffix(f".{os.getpid()}.tmp")
            temporary.write_text(json.dumps({"pairs": pairs, "names": names,
                                             "signals": {k: signals[k] for k in names},
                                             "floors": floors}))
            temporary.replace(path)
        return signals, floors

    @staticmethod
    def combine(distances, floors):
        return sum(distances[k] / floors[k] for k in distances)

    def score(self, prompts, generated, references):
        if not (len(prompts) == len(generated) == len(references)) or not prompts:
            raise ValueError("ref_w1 needs one prompt and reference per generated text")
        real, floors = self.reference(prompts, references)
        fake = self.signals(prompts, generated)
        distances = {k: wasserstein_1(real[k], fake[k]) for k in real}
        result = {
            "ref_w1": self.combine(distances, floors),
            "ref_w1_rate": distances["rate"],
            "ref_w1_rate_floor": floors["rate"],
            "ref_w1_nll": distances.get("nll"),
            "ref_w1_nll_floor": floors.get("nll"),
            "ref_w1_signature": self.signature(),
        }
        if self.bootstrap:
            generator = np.random.default_rng(self.bootstrap_seed)
            n, replicates = len(prompts), []
            for _ in range(self.bootstrap):
                index = generator.integers(0, n, n)  # paired: prompt i on both sides
                replicates.append(self.combine(
                    {k: wasserstein_1(np.asarray(real[k])[index], np.asarray(fake[k])[index])
                     for k in real}, floors))
            tail = (1 - self.bootstrap_confidence) / 2
            low, high = np.quantile(replicates, [tail, 1 - tail])
            result["ref_w1_interval"] = {
                "low": float(low), "high": float(high), "replicates": self.bootstrap,
                "confidence": self.bootstrap_confidence, "resampling": "paired_prompts",
            }
        if not math.isfinite(result["ref_w1"]):
            raise FloatingPointError("nonfinite ref_w1")
        return result

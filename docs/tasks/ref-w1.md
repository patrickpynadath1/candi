# ref_w1: coding-rate and likelihood Wasserstein distances

`core/evaluation/ref_w1.py` is a cheap, sample-efficient distributional score for
prompted text. TinyStories reruns tune with it. [SOL](sol.md) stays the test metric.
It builds on the compression-as-evaluation metric (W1 between per-document,
reference-conditioned coding rates) and adds a likelihood term.

## Definition

For a prompt p and continuation y, two per-document signals are computed.

- **Prompt-conditioned coding rate**, with a Zstandard dictionary built from real
  reference text:

      r(p, y) = (bits(p + y) - bits(p)) / bytes(y)

  Each `bits(·)` is one independent frame with the same dictionary. The prompt's own
  cost is subtracted, so prompt difficulty does not enter.
  - Defaults: level 10, window log 23, and a 64 KiB raw-content dictionary.
  - The dictionary is the first 65,536 bytes of `"\n\n".join(prompt + reference)` over the
    fit stories, in order.
  - Frames carry no checksum, content size or dictionary id.
- **Token-averaged NLL** (nats per token) of y given p under a frozen causal LM. The
  default is GPT-2 Large at the SOL encoder's pinned revision, loaded the same way. Any
  Hugging Face causal LM can be configured instead, such as an in-domain TinyStories
  checkpoint (`nll.model_name`). `nll: null` drops the term.

Each signal is compared between the real continuations of a set of prompts and the
generated continuations of the same prompts. The comparison is the exact 1-D
Wasserstein-1: equal weight per document, unequal sizes integrated exactly with no
truncation. Each distance is divided by its **floor**, the W1 between two fixed halves of
the real list. The halves are split by the hash of each (prompt, reference) pair, so the
split does not depend on order. The floor is the spread expected between two real samples
of this size.

    ref_w1 = W1_rate / floor_rate + W1_nll / floor_nll        (lower is better)

Real data scores about 1 per term. On TinyStories validation data the rate-only term is:

| Generated text | Rate term |
| --- | --- |
| Other real stories | ~1.1 |
| Truncated stories | ~18 |
| Uniform random letters | ~108 |
| Repeated phrase | ~119 |
| Empty | ~357 |

Repetition lowers the rate and noise raises it, so the score grows in both directions of
degeneration.

An empty continuation is scored as a newline (compression) or the LM's EOS token
(likelihood). `ref_w1_interval` (with `bootstrap: N`) is a percentile interval under
paired resampling of prompts, with the floors held fixed. It is cheap, because everything
is 1-D.

## Use in TinyStories

`+task/ref_w1=gpt2_large task.ref_w1.fit_indices=[...] task.primary_metric=ref_w1`, or the
`task.ref_w1` override of an evaluation spec.

The dictionary's fit stories are validation stories given by index (`fit_split`,
`fit_indices`). They must be disjoint from every evaluated prompt.
`dictionary_sha256` pins the dictionary: a mismatch, meaning the data or the split
changed, is refused.

Real-side signals and floors are cached per reference set under
`$SDDC_STORAGE_ROOT/ref_w1_cache`. The fields recorded are:

- `ref_w1` and its components `ref_w1_rate` and `ref_w1_nll`;
- the floors `ref_w1_rate_floor` and `ref_w1_nll_floor`;
- `ref_w1_signature`.

The `zstandard` package is in the `scoring` extra.

Tests: `core/tests/test_ref_w1.py`.

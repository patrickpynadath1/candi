# SOL: a distributional text metric

`core/evaluation/sol.py` implements SOL (Kornhardt et al., *SOL: Measuring Gaps
between Text Distributions by Double Sliced Wasserstein Metrics*, arXiv 2610.06513),
and `TinyStoriesMetrics` records it as `sol` when `task.sol` is configured.

## Definition

Each text is the uniform empirical measure of its per-token hidden states under a
frozen encoder T. SOL_θ(P, Q) = sqrt(d) · DSW(T#P, T#Q), where d is the hidden size:

1. **Outer slices.** `n_directions` unit directions ξ on the sphere project each
   text's token states to one dimension; the quantile function of that 1D measure is
   averaged over `n_quantiles` equal bins (exact bin averages of the empirical CDF).
2. **Inner slices.** `n_projections` draws g of a zero-mean Gaussian process (RBF
   kernel, lengthscale 0.1, on the bin midpoints) map each binned quantile function
   to a scalar ⟨g, Q⟩ / n_quantiles.
3. For every (ξ, g) slice, the squared W2 distance between the two corpora's scalar
   distributions (exact transport when the corpus sizes differ). SOL is the square
   root of d times their mean.

Lower is better; SOL(P, P) = 0 and SOL is symmetric and invariant to document order.

## Implementation and the reference code

The estimator is a Torch port of the reference implementation
(github.com/GregorKornhardt/SOL, Apache-2.0) and matches its `SOL.from_tokens`
to float32 precision (`core/tests/test_sol.py`, which runs the comparison when
`sol_metric` is installed). The same conventions hold: NumPy-seeded directions and GP
draws (seed 0 each, so every trial is scored on identical slices, on any device), raw
unnormalised GP draws with Cholesky jitter, `scale = d`.

| Setting | Value | Note |
| --- | --- | --- |
| Encoder | `gpt2-large`, revision `32b71b12…` | The reference code's default is Dream 7B; GPT-2 Large is its documented alternative with a pinned paper revision and 1024 slices per level |
| Layer | `last_hidden_state` (after the final layer norm) | as the reference `TextEncoder` |
| Precision | FP16 on CUDA, FP32 on CPU | as the reference; states are cast to float32 |
| Slices | 1024 directions x 1024 GP draws | configurable; the paper reports up to 8192 |
| Quantile bins | 64 | |
| Truncation | the model's 1024 positions | TinyStories continuations are far shorter |

Deviations, both for generated text the reference code rejects: an empty
continuation (EOS at once) is encoded as the encoder's EOS token, so it is a valid,
distant measure instead of an error; and `sol_interval` (optional) is a percentile
bootstrap with **paired** resampling of prompts (generated text i and reference i are
drawn together), which the reference code does not provide.

## Use in TinyStories

The reference set is the real held-out continuations of the same prompts (same n).
During tuning these are validation stories; the test references are used once, for the
winner. Reference features are cached under `$SDDC_STORAGE_ROOT/sol_cache`, keyed by
the encoder provenance, the SOL settings and the multiset of reference texts, so
every trial against the same prompts reuses one featurization. The encoder loads on
first scoring only: training and CPU tests never need GPT-2 Large.

```bash
uv run --locked python main.py mode=evaluate evaluation.checkpoint=/path/last.ckpt \
  +task/sol=gpt2_large task.primary_metric=sol task.calculate_bleu=true
```

`configs/task/sol/gpt2_large.yaml` holds the settings; `bootstrap: N` adds
`sol_interval` (N full-cost replicates; successive-halving sweeps ask for it only when
re-scoring finalists). `result.json` records `sol`, `sol_signature` (settings and
encoder provenance) and, with BLEU on, `corpus_bleu`, `corpus_bleu_no_bp` and
`bleu_brevity_penalty`.

Cost: encoding n texts with GPT-2 Large is one forward pass each; scoring sorts
1024 x 1024 slices of n values per corpus (about a second for n = 1024 on a GPU).

# MAUVE: distribution match on language-model features

`core/evaluation/mauve.py` computes MAUVE (Pillutla et al., 2021) between the generated
continuations and the reference continuations of the same prompts. TinyStories tunes with
it on TinyStories-33M features. The decision, its rationale and the stability study are in
[the tuning-metric decision record](../methods/tinystories-tuning-metric.md). Higher is
better: 1 means the two samples fall into the same histogram.

## Definition

- **Features.** Each text is `prompt + continuation`. Its feature is the final-layer
  hidden state of its last token under a frozen causal LM (`featurizer.model_name`, at a
  pinned revision, FP32). This is MAUVE's standard featurisation.
  - TinyStories-33M ships only `pytorch_model.bin`, which transformers 5.8 refuses to
    load on torch < 2.6. The weights are then read with `torch.load(weights_only=True)`.
  - A text longer than the model's context keeps its last tokens, so it is cut from the
    left of the prompt. mauve-text keeps the first tokens instead. TinyStories texts are
    far below both contexts (2,048 tokens for TinyStories-33M, 1,024 for GPT-2 Large).
- **References.** One per distinct example index. When seeds are pooled, a prompt's
  reference counts once, so the test compares 4,096 samples with 2,048 references.
  Reference features are cached per (model, revision, index and text set) under
  `$SDDC_STORAGE_ROOT/mauve_cache`.
- **Estimator.** The in-repo estimator follows mauve-text 0.4.0's `compute_mauve`
  defaults, with numpy, scipy and scikit-learn instead of faiss:
  1. L2-normalise every row, fit PCA on the union and keep the components explaining
     `pca_variance` (0.9) of the variance.
  2. Run k-means with `num_buckets` clusters on the union: random-point initialisation,
     5 restarts, 500 iterations, seed `seed + 2` (`seed` defaults to 25).
  3. Build one histogram per side. Compute the divergence frontier at 25 mixture
     weights, mapped by exp(-5 KL), and take the area under it.
  - `num_buckets: auto` is mauve-text's rule, max(2, round(min(n_p, n_q) / 10)).
    Tuning fixes it per stage, because the auto value grows with n.
  - Rows are put in a canonical order first, so the score does not depend on sample
    order.

Validation against mauve-text 0.4.0 used identical saved features of the 56
stability-study sample sets (`/data/sddc/analysis/mauve-dev/validation/`).

- The two implementations differ by about as much as mauve-text differs from itself under
  another k-means seed:
  - at 4,096 vs 2,048: mean |difference| 0.013 against mauve-text's own 0.013 (seed 25
    vs 26), Spearman 0.99;
  - at n = 128 and 256: 0.04-0.06 against 0.04-0.05.
- Averaged over 10 k-means seeds, the two agree to a mean of 0.011-0.017, with bias
  within ±0.006.
- **Quantisation noise is large at tuning sizes.** At 128-256 stories, the k-means seed
  alone moves a score by an SD of about 0.03-0.045 in both implementations.

## Use in TinyStories

`+task/mauve=tinystories_33m task.primary_metric=mauve`, or the `task.mauve` override of
an evaluation spec. `task.mauve_gpt2_large` takes the same settings with a GPT-2 Large
featuriser and is reported only. The recorded fields are:

- `mauve`, `mauve_signature` (the featuriser, the settings, the bucket count used and the
  sample sizes), and `mauve_interval` with `bootstrap: N` (paired prompt resampling);
- the same fields prefixed `mauve_gpt2_large` for the cross-check.

`experiments/rerun/evaluation/tinystories-bpe-10000-mauve.json` tunes with MAUVE and
sets `score.pool_seeds: true` (see [nested tuning](../guides/nested-tuning.md)). It uses
13 buckets in stage 1 (64 prompts x 2 seeds), 26 in stage 2 (256 x 1) and 205 on test
(2,048 x 2), where SOL and GPT-2 Large MAUVE are also reported. The extra-budget test
measurements (128 prompts x 1 seed) use 13 buckets (`extra_test_overrides`). Without
that override they would inherit 205 buckets, and a MAUVE of about 0.02 came out of a
128 vs 128 sample.

Tests: `core/tests/test_mauve.py`.

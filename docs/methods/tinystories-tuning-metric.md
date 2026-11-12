# TinyStories tuning metric: MAUVE with TinyStories-33M features

Decision (2026-11-12): sampler tuning on TinyStories selects by MAUVE (Pillutla et al.,
2021) computed on TinyStories-33M features, replacing `ref_w1`. The test split reports
the same MAUVE, GPT-2 Large MAUVE as a cross-check, and a per-sample quality measure
(SOL and the Jev judge) next to it. The full rerun of the TinyStories cases uses this
metric from the first trial, so the search itself is guided by it.

## What the tuning metric has to do

- Measure distribution match, not per-sample quality. A diffusion LM models the data
  distribution, and the post frames comparisons as divergences. Per-sample scores (a judge,
  or likelihood under a reference LM) reward sharpening: lower temperature, shorter and
  safer stories. On the rerun, tuned samplers wrote shorter stories than the untuned native
  sampler in 27 of 28 models (e.g. 368 vs 654 bytes), which a per-sample metric scores as
  better. MAUVE penalises both poor samples and lost diversity.
- Be stable at tuning sizes: stage 1 scores 128 stories per setting (64 prompts x 2 seeds),
  stage 2 scores 256.
- Be cheap enough to run on every trial.
- Need no defence of its own beyond the published method.

## Why TinyStories-33M features

The stability test (`/data/sddc/analysis/mauve-stability-20261112/`; 56 sample sets = 28
models x {tuned, untuned}; subsamples of n prompts, 10 draws, sampler and references on
the same prompts; full value n = 4,096 vs 2,048 references):

| features | n | per-draw Spearman vs full (mean / min) | tuned-vs-untuned calls matching full (of 28) |
|---|---|---|---|
| TinyStories-33M | 256 | 0.90 / 0.85 | 27.5 (min 27); 21/21 on large gaps |
| TinyStories-33M | 128 | 0.76 / 0.63 | 25.1 (min 23) |
| GPT-2 Large | 256 | 0.87 / 0.80 | 24.6 (min 21) |
| GPT-2 small | 256 | 0.68 / 0.47 | 20.8 (min 17) |

- TinyStories-33M is in-domain: it separates TinyStories-style samples, while GPT-2 small
  saturates (values squeezed into 0.75-0.98, no gap above 0.2).
- It is the most stable of the three at 128-256 stories, and about 30x cheaper than GPT-2
  Large: MAUVE from features takes about 0.1 s at n = 256 on CPU, featurising 256 texts
  0.12-0.20 s on one H100; reference features are cached per prompt set.
- It is a distinct metric, not a cheap copy of GPT-2 Large MAUVE: at full n they agree at
  Spearman 0.79 and on 23 of 28 tuned-vs-untuned calls. GPT-2 Large MAUVE is therefore
  reported on test as a cross-check.

## What was ruled out

- `ref_w1`: needs its own defence, and on the 8 models where it and the Jev judge disagree
  it had no independent support.
- A KL estimate from per-document NLL minus a zstd self-entropy: zstd's redundancy (about
  1.4 nats/byte beyond GPT-2 Large) grows with length and does not cancel; the estimate
  ranks by length (correlation -0.98), falls as temperature rises, and scores a repeated
  sentence as better than the reference (`/data/sddc/analysis/kl-eval-20261112/`).
- GPT-2 small MAUVE: saturates.
- A judge as the tuning target: rewards sharpening and collapse.

## Settings

- Features: the last-token final hidden state of TinyStories-33M on prompt + continuation
  (MAUVE's standard featurisation).
- References: the reference continuations of the same prompts the sampler was scored on.
- A fixed bucket count, so scores are comparable across trials within a stage (the
  automatic count grows with n). Recorded in the evaluation file.
- MAUVE's own seed fixed.

## Caveats

- One training seed per model, as for every result in the post.
- The stability test drew prompts from the test split, not the validation tuning prompts.
- Final-token features may weigh story length and endings heavily; this is part of the
  distribution shift being measured, but it is not separated out.

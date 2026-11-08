# Continuous schedules

**Question.** How do learned (CDCD) noise schedules spend training, and what does
self-conditioning (SC) do to continuous methods?

This example backs the post's section
[Continuous Noise Schedules](https://patrickpynadath1.github.io/blog/diffusion-lm-design-choices).
Every pure continuous recipe (Gaussian embeddings, simplex, spherical, vMF) learns its
noise schedule with CDCD ([time sampling](../../docs/components/time-sampling.md)): the
sampler fits the expected loss as a monotone function of the noise level and draws training
times through its inverse. So every stretch of the warped time `u` should carry the same
share of the loss. On the final checkpoints the example measures:

1. **Clock linearity.** The measured loss in 20 equal-width bins of `u`, on validation and
   on training data. If the clock works, the loss rises in a straight line in `u`.
2. **The clocks.** `u` mapped to physical time `t` and to each path's own noise parameter
   (Gaussian `sigma`, spherical keep rate, vMF `kappa`).
3. **Corruption along the clock**, in two units every method shares: how often the nearest
   codeword is the wrong token, and the normalized rank of the true token.
4. **Performance** of each method with and without SC, with the sampler chosen on validation.

## Protocol

| | |
| --- | --- |
| Grid | {Gaussian embeddings, simplex, spherical, vMF} x {no SC, SC} x {Sudoku-hard, text8, TinyStories BPE-10k} = 24 cases, one training seed |
| Schedule | CDCD learned clock (`path_schedule=cdcd`) on the continuous view, with the inverse-CDF clamp fix in `core/time_sampling/cdcd.py` |
| Backbone | 8 blocks, width 512, 8 heads, AdaLN (the `corruption_factorial` recipe) |
| Training | 20k steps, batch 256, lr 3e-4 with 2.5k warmup, EMA 0.9999, bf16 |
| SC | `posterior_mean` (expected embedding of the previous prediction), probability 0.9 |
| Samplers | `native` (temperature, 16 trials), `native_greedy` (2), `eb_greedy` (temperature and entropy bound, 16), `renoise_sample` (temperature, 16), `renoise_greedy` (2) |
| Budget | 128 NFE for every sampler |
| Selection | The best sampler on 128 validation examples x seeds 41, 42; only that sampler sees the 256 test examples x seeds 1041, 1042 |
| Scores | Sudoku: solve rate. text8: unique valid words (length >= 5) per sample. TinyStories: corpus BLEU of the continuation against the real one |

The splits, scores and task overrides are in `evaluation/<task>.json`. The sweep is
[`configs/sweep/continuous_schedules.yaml`](../../configs/sweep/continuous_schedules.yaml).
The renoising samplers redraw the whole state from the forward kernel at the next time
after each prediction (`ComposedSampler` with `ForwardCorruption`).

## Run

```bash
# 1. Prepare the 24 cases (no training).
uv run --locked --extra tuning --extra scoring python -m core.sweeps.prepare -m \
  +sweep=continuous_schedules preparation.output=$CANDI_STORAGE_ROOT/plans/continuous-schedules
uv run --locked --extra tuning --extra scoring python -m core.sweeps.prepare collect \
  $CANDI_STORAGE_ROOT/plans/continuous-schedules
# 2. Train and tune; only the validation winner sees the test split.
uv run --locked --extra tuning --extra scoring python -m core.sweeps.nested_tuning \
  $CANDI_STORAGE_ROOT/plans/continuous-schedules/manifest.json \
  --output $CANDI_STORAGE_ROOT/runs/continuous-schedules --address auto --case-concurrency 4
# 3. Tables. With checkpoints present this also measures the schedules (one GPU: minutes;
#    CPU works but is slow for TinyStories). --no-schedules skips that part.
uv run --locked --extra scoring python experiments/continuous-schedules/analyze.py \
  $CANDI_STORAGE_ROOT/runs/continuous-schedules --output $CANDI_STORAGE_ROOT/analysis/continuous-schedules
```

TinyStories needs the BPE tokenizer in `artifacts/tokenizers/tinystories-bpe/` (in the repo)
and the TinyStories parquet files (`TINYSTORIES_DATA_DIR`). `analyze.py` writes
`cdcd_performance.json` and `cdcd_schedules.json` in the schema of the post's data files
and prints them next to `expected.csv`. It takes several run roots; a case found in more than
one is merged, and its sampler is chosen by validation over all the policies tried.

The schedule tables come from `schedule_report.py` (clock curve and loss bins) and
`corruption_report.py` (token error and rank). They read a case's `trained.json`, use the
raw training weights (not EMA), the training-mode SC policy and no gradients or sampler
updates, and reuse the first 128 examples of each split in every bin with fixed seeds.

## Cost

About 80 GPU hours on an RTX A6000-class GPU. The figures come from the stored runs'
timestamps and evaluation times (A6000 for Sudoku and text8, MI210 for TinyStories):

| Task | Training (20k steps) | Sampler tuning (52 trials x 2 seeds) and test | Cases |
| --- | --- | --- | --- |
| Sudoku-hard | about 1.6 h | about 0.6 h | 8 |
| text8 | about 2 h | about 1 h | 8 |
| TinyStories BPE-10k | about 3.5 h | about 1.5 h | 8 |

The schedule analysis adds well under an hour on one GPU.

## Results

The published numbers are in `expected.csv`. Test score of the validation-selected sampler;
in parentheses, the relative gain from SC on the text tasks.

| Method | Sudoku, no SC | Sudoku, SC | text8, no SC | text8, SC | TinyStories BPE, no SC | TinyStories BPE, SC |
| --- | --- | --- | --- | --- | --- | --- |
| Gaussian | 52.0% | 92.8% | 12.84 | 14.39 (+12%) | 6.81 | 7.86 (+15%) |
| Simplex | 26.4% | 91.6% | 11.04 | 11.97 (+8%) | 6.33 | 8.50 (+34%) |
| Spherical | 59.6% | 92.6% | 9.61 | 13.11 (+36%) | 7.55 | 8.27 (+9%) |
| vMF | 55.1% | 95.7% | 9.98 | 13.16 (+32%) | 7.06 | 7.67 (+9%) |

On text, a straight line explains at least 99.8% of the variation in the measured loss
(R² of the least-squares fit over the 20 bins), on training and on validation data, for every
method. On Sudoku it does not, even on training data: the models overfit their training
puzzles.

**Reproduction check.** On the stored runs (the ruqi6000 campaigns for Sudoku and text8, the
MI210 campaign for TinyStories BPE-10k), `analyze.py` reproduces all 24 rows of the post's
`cdcd_performance.json` exactly: score, per-seed scores, validation score, selected sampler,
`renoise_tried`, and for TinyStories the per-sampler lists, ties and the BLEU breakdown
without the brevity penalty. The stored runs hold no checkpoints, so the schedule, linearity and
corruption figures can only be rebuilt from new runs (see the caveats).

## Caveats

- **The TinyStories performance rows come from a different protocol.** They were run on an
  AMD MI210 by a collaborator, with another source commit, torch build and data partition
  (TinyStories filtered to an allowed-ids list). Recipe, model, training budget, samplers and
  128 NFE match. Their renoising samplers were `ForwardResamplingSampler`, named `renoising`
  and `renoising_greedy`; `analyze.py` reports them as `renoise_sample` and `renoise_greedy`.
  `evaluation/tinystories-bpe-10000.json` is that campaign's split, the only one recorded.
  Treat TinyStories reruns as a new measurement, not a reproduction.
- **The published schedule panels used byte-level TinyStories** (vocabulary 260), and their
  checkpoints are not available. This example uses BPE-10k throughout, so its TinyStories
  schedules, linearity and corruption curves will differ from the post's. The
  TinyStories linearity rows in `expected.csv` are the byte-level values, for reference.
- **Renoising samplers everywhere.** In the published runs, renoising was tuned in a second run
  on the same checkpoints, for every Sudoku cell and for text8 Gaussian, and merged by validation. It was never tried
  for text8 simplex, spherical and vMF (`renoise_tried` is false there). This example tunes all
  five samplers in every case, so those three text8 cells may select a renoising sampler. For
  simplex, `native` already redraws the state from the forward kernel, so `native` and
  `renoise_sample` score the same (TinyStories simplex shows that tie).
- **Ties.** Validation ties go to the first sampler listed. The published Sudoku simplex-SC
  and vMF-SC rows had ties (`native_greedy` with both renoising samplers; `native` with
  `native_greedy`) that the post's file does not record; `analyze.py` writes them as `tie`.
- **One training seed per cell.** The SC gains are large next to the spread between the two
  evaluation seeds, but each comes from one training run. Sudoku numbers should not be read: the
  models overfit, and their clocks do not describe held-out loss.
- The backbone is now built by `core.models.modular.ModularDiT` rather than the older
  `HybridModel` wrapper; the hyperparameters are the same (see
  [modular DiT](../../docs/design/modular-dit.md)).

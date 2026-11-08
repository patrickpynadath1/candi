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
| Schedule | CDCD learned clock (`information_path/path_schedule=cdcd`) on the continuous view, with the inverse-CDF clamp fix in `core/time_sampling/cdcd.py` |
| Backbone | 8 blocks, width 512, 8 heads, AdaLN (the `corruption_factorial` recipe) |
| Training | 20k steps, batch 256, lr 3e-4 with 2.5k warmup, EMA 0.9999, bf16 |
| SC | `posterior_mean` (expected embedding of the previous prediction), probability 0.9 |
| Samplers | `native` (temperature, 16 trials), `native_greedy` (2), `eb_greedy` (temperature and entropy bound, 16), `renoise_sample` (temperature, 16), `renoise_greedy` (2) |
| Budget | 128 NFE for every sampler |
| Hardware | the published performance runs: one AMD MI210 per case (torch 2.9.1+rocm6.3) |
| Selection | The best sampler on 128 validation examples x seeds 41, 42; only that sampler sees the 256 test examples x seeds 1041, 1042 |
| Scores | Sudoku: solve rate. text8: unique valid words (length >= 5) per sample. TinyStories: corpus BLEU of the continuation against the real one |

The splits, scores and task overrides are in `evaluation/<task>.json`. The sweep is
[`configs/sweep/continuous_schedules.yaml`](../../configs/sweep/continuous_schedules.yaml).
The samplers are [`configs/inference/inference_policies/hybrid_renoising.yaml`](../../configs/inference/inference_policies/hybrid_renoising.yaml).
The renoising samplers redraw the whole state from the forward kernel at the next time
after each prediction (`ForwardResamplingSampler`); they are named `renoising` and
`renoising_greedy` there, and `analyze.py` reports them as `renoise_sample` and
`renoise_greedy`, the post's names.

## Run

```bash
# 0. Data the published runs used that is too large to commit (see the caveats).
python scripts/prepare_data.py text8 --parquet PATH/afmck-text8/data \
  --raw-dir $SDDC_STORAGE_ROOT/data_cache/text8/raw
python scripts/prepare_data.py tinystories --parquet PATH/roneneldan-TinyStories/data \
  --output $SDDC_STORAGE_ROOT/data_cache/tinystories_bpe/allowed_ids.json
# 1. Prepare the 24 cases (no training).
uv run --locked --extra tuning --extra scoring python -m core.sweeps.prepare -m \
  +sweep=continuous_schedules preparation.output=$SDDC_STORAGE_ROOT/plans/continuous-schedules
uv run --locked --extra tuning --extra scoring python -m core.sweeps.prepare collect \
  $SDDC_STORAGE_ROOT/plans/continuous-schedules
# 2. Train and tune; only the validation winner sees the test split.
uv run --locked --extra tuning --extra scoring python -m core.sweeps.nested_tuning \
  $SDDC_STORAGE_ROOT/plans/continuous-schedules/manifest.json \
  --output $SDDC_STORAGE_ROOT/runs/continuous-schedules --address auto --case-concurrency 4
# 3. Tables. With checkpoints present this also measures the schedules (one GPU: minutes;
#    CPU works but is slow for TinyStories). --no-schedules skips that part.
uv run --locked --extra scoring python experiments/continuous-schedules/analyze.py \
  $SDDC_STORAGE_ROOT/runs/continuous-schedules --output $SDDC_STORAGE_ROOT/analysis/continuous-schedules
```

TinyStories needs the BPE tokenizer in `artifacts/tokenizers/tinystories-bpe/` (in the repo),
the TinyStories parquet files (`TINYSTORIES_DATA_DIR`) and the story cohort at
`${data.cache_dir}/allowed_ids.json`. `analyze.py` writes
`cdcd_performance.json` and `cdcd_schedules.json` in the schema of the post's data files
and prints them next to `expected.csv`. It takes several run roots; a case found in more than
one is merged, and its sampler is chosen by validation over all the policies tried.

The schedule tables come from `schedule_report.py` (clock curve and loss bins) and
`corruption_report.py` (token error and rank). They read a case's `trained.json`, use the
raw training weights (not EMA), the training-mode SC policy and no gradients or sampler
updates, and reuse the first 128 examples of each split in every bin with fixed seeds.

## Cost

About 125 GPU hours on AMD MI210s (81 training, 43 sampler tuning), from the archive's
file-timestamp estimates (`wall_hours_approx_*`):

| Task | Training (20k steps), no SC / SC | Sampler tuning (52 trials x 2 seeds) and test | Cases |
| --- | --- | --- | --- |
| Sudoku-hard | 1.2-1.4 h / 1.5-1.7 h | 0.9-1.1 h | 8 |
| text8 | 1.5-1.7 h / 1.8-2.1 h | 1.3-1.5 h | 8 |
| TinyStories BPE-10k | 3.2-15.6 h / 4.1-16.8 h | 1.8-5.2 h | 8 |

The long TinyStories cells are the simplex ones, whose Dirichlet process is slow at a
10k vocabulary. The schedule analysis adds well under an hour on one GPU.

## Results

The published numbers are in `expected.csv`. Test score of the validation-selected sampler;
in parentheses, the relative gain from SC on the text tasks.

| Method | Sudoku, no SC | Sudoku, SC | text8, no SC | text8, SC | TinyStories BPE, no SC | TinyStories BPE, SC |
| --- | --- | --- | --- | --- | --- | --- |
| Gaussian | 53.3% | 90.6% | 12.96 | 14.07 (+9%) | 6.81 | 7.86 (+15%) |
| Simplex | 23.4% | 89.1% | 11.53 | 13.34 (+16%) | 6.33 | 8.50 (+34%) |
| Spherical | 50.0% | 92.2% | 12.98 | 12.97 (-0%) | 7.55 | 8.27 (+9%) |
| vMF | 56.5% | 93.8% | 12.78 | 13.04 (+2%) | 7.06 | 7.67 (+9%) |

On text8, the spherical and vMF models without SC gain about 3 words per sample from the
renoising samplers, which closes their gap to SC; the SC models still prefer `native`.

On text, a straight line explains at least 99.8% of the variation in the measured loss
(R² of the least-squares fit over the 20 bins), on training and on validation data, for every
method (measured on the earlier A6000 checkpoints; see the caveats). On Sudoku it does not, even on training data: the models overfit their training
puzzles.

**Reproduction check.** On the stored runs (the MI210 archive
`portable-v2-mi210-partial-20261106-0013`), `analyze.py` reproduces all 24 rows of the post's
`cdcd_performance.json` exactly: score, per-seed scores, validation score, selected sampler,
`renoise_tried`, the per-sampler lists, ties and, for TinyStories, the BLEU breakdown without
the brevity penalty. Every resolved config this sweep prepares matches its published case
key by key, apart from machine paths, the newer `ModularDiT` model layout and two
settings applied at evaluation time through the evaluation file. The archive holds no
checkpoints, so the schedule, linearity and corruption figures can only be rebuilt from new
runs (see the caveats).

## Caveats

- **The schedule panels come from earlier checkpoints.** The post's clock, linearity and
  corruption figures (`cdcd_schedules.json`, the `linearity_r2_*` rows of `expected.csv`)
  were measured on an earlier run of this grid on an RTX A6000, whose checkpoints are the
  only ones measured so far. Its TinyStories models were byte-level (vocabulary 260), and its
  text8 simplex, spherical and vMF cells were not tuned with renoising. This example trains
  BPE-10k throughout, so its TinyStories schedules will differ from the post's panels.
- **Code and hardware.** The published performance runs used source commit `aa4ee07` plus a
  patch, torch 2.9.1+rocm6.3 on MI210s. The backbone is now built by
  `core.models.modular.ModularDiT` (same hyperparameters), and the training loader draws a
  per-epoch shuffled order of whole optimizer steps. A rerun follows a different training
  trajectory; expect close numbers, not equal ones.
- **Data.** text8 is the Hugging Face `afmck/text8` copy written as the task's split files,
  and TinyStories is restricted to the cohort shared with the
  [tokenizer ladder](../tokenizer-ladder/); `scripts/prepare_data.py` rebuilds both, so the
  evaluation indices pick the same examples.
- **Simplex and renoising.** For simplex, `native` already redraws the state from the forward
  kernel, so `native` and `renoise_sample` tie on validation (text8 and TinyStories with and
  without SC, Sudoku with SC); the first listed, `native`, is evaluated. Sudoku without SC is
  the exception: there the tuned temperatures differ and `renoise_sample` wins.
- **Sampler collapses.** Some renoising and entropy-bounded policies score near zero on some
  checkpoints; validation selection is unaffected.
- **One training seed per cell.** Several SC gains are within the spread between the two
  evaluation seeds. Sudoku numbers should not be read: the models overfit, and their clocks
  do not describe held-out loss.

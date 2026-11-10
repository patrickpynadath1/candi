# rerun: ref_w1 tuning and SOL testing, Ψ and EB-churn samplers, two-stage selection

An inference-only rerun of [sc-vs-depth](../sc-vs-depth/),
[continuous-schedules](../continuous-schedules/) and [hybrid-sc](../hybrid-sc/) on the
checkpoints of their H100 reproduction (`blog-repro-h100`). It also covers
[sc-sampler-choice](../sc-sampler-choice/), which only reads the sc-vs-depth runs'
per-policy validation scores. block-encoder-decoder is not included.

The Gaussian-embedding arm is not rerun on its old checkpoints. The embedding default is
now 512 dimensions, which would no longer match those 128-dimensional checkpoints, so
the arm is **retrained** at 512 by the `*_gauss512` sweeps under the same evaluation
protocol (see below). The reruns cover the other 72 cases.

The recipes, data cohorts, seeds and checkpoints are those of the source runs. The cases'
pinned configs equal the source runs' `protocol.json` configs apart from two
training-only keys that later code adds with their default values
(`prediction.fixed_feedback`, `prediction.selected_gradients`), and the CDCD cases'
learned clock layout: the source runs held it as the `cdcd` training time sampler,
the cases as the learned time map `algorithm.time_map` (`cdcd_loss_cdf`) with the
`antithetic` sampler that drew the same times. The source protocols migrate to
exactly the case configs (`core.training.legacy_checkpoints.migrate_time_map_config_dict`,
checked on all 99 source protocols), and the checkpoints' `time_sampler.*` state
loads as `time_map.*`, so evaluation is unchanged.

What changes:

1. **TinyStories metrics.** Tuning uses [ref_w1](../../docs/tasks/ref-w1.md): W1 between
   real and generated prompt-conditioned Zstandard coding rates, plus W1 between GPT-2
   Large NLLs, each over its real-vs-real floor. [SOL](../../docs/tasks/sol.md) (GPT-2
   Large, 1024 x 1024 slices) replaces BLEU as the test metric. ref_w1 and BLEU (with and
   without brevity penalty) are recorded on the test split too. Sudoku (accuracy) and
   text8 (unique valid words of length 5 or more) keep their metrics.
2. **Samplers.** The five `hybrid_renoising` policies are kept exactly. Two policies are
   added to every case:
   - `eb_churn`: EB with random uncommitment, tuning rate, stop fraction, entropy bound
     and temperature (24 trials);
   - `psi`: the [Ψ-sampler](../../docs/components/psi-sampler.md), tuning schedule, η,
     loop t_on/t_off, top_p and temperature (24 trials). It applies to every corruption
     (Gaussian, simplex, spherical, vMF, masked, uniform and the hybrids).
3. **Tuning.** Two-stage selection at NFE 128. Every trial is scored on the first 64 prompts
   of the tuning pool with the tuning seeds. Each policy's top 3 are then re-scored on the
   next, disjoint 256 prompts with a fresh seed (2041). The best stage-2 score wins and
   records a 200-replicate bootstrap interval: per example for Sudoku and text8, and by
   ref_w1's paired-prompt bootstrap for TinyStories.
   - **Optuna samplers:** GP (`GPSampler`) for the policies with small continuous spaces
     (native, eb_greedy, renoising, eb_churn). TPE with `multivariate` and `group` for
     psi, whose space is categorical and conditional. native_greedy and
     renoising_greedy (one two-valued choice) keep TPE, with both values as initial points,
     so each is evaluated once.
4. **Test.** The overall validation winner is scored once on the test split at NFE 128.
   Afterwards, as a smoke measurement, **each policy's** validation winner is scored once
   at NFE 512 on a fixed 128-prompt subset of the test partition with one test seed
   (1041), in `final-extra/nfe-512/<policy>/`. This is reported only; no selection
   reads it.

## Partitions

`evaluation/<task>.json`, drawn with `numpy.random.default_rng(20261111 + task)`. Each
test partition excludes the 128 tuning indices that the source runs' training callbacks
generated from.

| Task | Validation examples | Tuning pool | Test | Before (tuning / test) |
| --- | --- | --- | --- | --- |
| Sudoku-hard | 2,000 | 1,024 | 976 (all the rest) | 128 / 256 |
| text8 | 19,531 | 2,048 | 2,048 | 128 / 256 |
| TinyStories BPE-10k | 13,092 | 2,048 | 2,048 | 128 / 256 |

Stage 1 uses the first `stage1` prompts of the tuning pool (in draw order) and stage 2 the next `stage2`; the rest of the pool is unused. Tuning seeds are 41 and
42, test seeds 1041 and 1042. text8's size is the packed validation split
(5,000,000 characters in 256-character rows: 19,531). Its official test split is also
available (`evaluation.test_dataset: test`) if a still larger untouched test set is
wanted.

**ref_w1 dictionary fit split (TinyStories).** 128 validation stories drawn with
`numpy.random.default_rng(20261112)` from outside the tuning pool, the test partition and
the source runs' callback indices (`overrides["task.ref_w1"].fit_indices`). The
dictionary is the first 65,536 bytes of `"\n\n".join(prompt + reference)` over them,
94,100 bytes available. It has SHA-256 `8ba6e3fd0d973c21b83d9837eae5d3809b02de1531e7d29d18a7f937c7246437`
on the cohort cache `stories_validation_16009c882a778c54728e38f8.json`, and the scorer
refuses any other dictionary.

## Sweeps

| Sweep | Cases | What |
| --- | --- | --- |
| `sc_vs_depth_gauss512` | 9 | train: Gaussian embedding (CDCD clock) x {8L, 8L + SC, 16L} x 3 tasks |
| `continuous_schedules_gauss512` | 6 | train: Gaussian embedding x {no SC, SC} x 3 tasks |
| `hybrid_sc_gauss512` | 12 | train: {masked, uniform} view x Gaussian embedding x {no SC, SC} x 3 tasks |
| `sc_vs_depth_rerun` | 18 | evaluate: {masked, uniform} x {8L, 8L + SC, 16L} x 3 tasks |
| `continuous_schedules_rerun` | 18 | evaluate: {simplex, spherical, vMF} x {no SC, SC} x 3 tasks |
| `hybrid_sc_rerun` | 36 | evaluate: {masked, uniform} x {simplex, spherical, vMF} x {no SC, SC} x 3 tasks |

The gauss512 cases (`<source case>-gauss512`) use the source sweeps' recipes, data
cohorts and seeds, trained from scratch for 20k steps with global batch 256. The only
differences are the 512-dimensional embedding (all three Gaussian keys follow
`state_representation.dimension`) and `trainer.accumulate_grad_batches: 1`.

Both kinds share `configs/sweep/rerun_protocol.yaml`, which sets the evaluation:
- the partitions in `evaluation/`;
- two-stage selection, with ref_w1 tuning and SOL testing on TinyStories;
- the 7 policies;
- the per-policy NFE-512 smoke test;
- evaluation batch 128 for TinyStories and 256 for text8 and Sudoku;
- manifests ordered longest task first (TinyStories, text8, Sudoku).

## Run

**Launch the gauss512 training sweeps first**: they train as well as evaluate, so they are
the long pole. Then launch the reruns, which only evaluate. On the H100 machine, where
`/data/sddc/runs/blog-repro-h100/<sweep>/` holds the source runs (also in
`s3://sddc-patrick-data-029636175027/runs/blog-repro-h100/`; override the root with
`SDDC_SOURCE_RUNS`):

```bash
for sweep in sc_vs_depth_gauss512 continuous_schedules_gauss512 hybrid_sc_gauss512 \
             sc_vs_depth_rerun continuous_schedules_rerun hybrid_sc_rerun; do
  name=${sweep//_/-}
  uv run --locked --extra tuning --extra scoring python -m core.sweeps.prepare -m \
    +sweep=$sweep preparation.output=$SDDC_STORAGE_ROOT/plans/$name
  uv run --locked --extra tuning --extra scoring python -m core.sweeps.prepare collect \
    $SDDC_STORAGE_ROOT/plans/$name
  uv run --locked --extra tuning --extra scoring python -m core.sweeps.nested_tuning \
    $SDDC_STORAGE_ROOT/plans/$name/manifest.json --check
done
```

then run `nested_tuning ... --output $SDDC_STORAGE_ROOT/runs/<name> --address auto` per
plan, in the same order.
- Every rerun case reads
  `<source case>/training/candidates/44136fa355b3678a1146/training/checkpoints/last.ckpt`,
  the single fixed training candidate, with the same hash in all three sweeps.
- `--check` refuses a missing checkpoint.
- GPT-2 Large downloads at its pinned revision on first scoring.

## Cost

These are sequence network calls (sequences x NFE) from `generation_cost`, an upper bound:
EB stops early. The current column is the source sweeps' own tuning plus test. The rows
count all 99 evaluated cases per experiment: the 72 reruns plus the 27 gauss512 cases,
which share the protocol and so the evaluation cost. Training the 27 gauss512 cases
comes on top.

| Sweep (cases) | Stage 1 | Stage 2 | Test @128 (winner) | Smoke @512 (per policy) | Total | Current | Ratio |
| --- | --- | --- | --- | --- | --- | --- | --- |
| sc_vs_depth_rerun (27) | 44.2M | 16.8M | 11.7M | 12.4M | 85.1M | 47.8M | 1.8x |
| continuous_schedules_rerun (24) | 39.3M | 14.9M | 10.4M | 11.0M | 75.7M | 42.5M | 1.8x |
| hybrid_sc_rerun (48) | 78.6M | 29.9M | 20.8M | 22.0M | 151.3M | 84.9M | 1.8x |
| **All (99)** | **162.2M** | **61.6M** | **42.8M** | **45.4M** | **312.1M** | **175.2M** | **1.8x** |

By task, over all three sweeps (33 cases each):

| Task | Stage 1 | Stage 2 | Test @128 | Smoke @512 | Total | Current | Ratio |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Sudoku-hard | 54.1M | 20.5M | 8.2M | 15.1M | 98.0M | 58.4M | 1.7x |
| text8 | 54.1M | 20.5M | 17.3M | 15.1M | 107.1M | 58.4M | 1.8x |
| TinyStories | 54.1M | 20.5M | 17.3M | 15.1M | 107.1M | 58.4M | 1.8x |

Per case: 112 trials x 64 prompts x 2 seeds at stage 1, and 7 policies x 3 finalists x
256 prompts x 1 seed at stage 2. The NFE-512 smoke measurement is 7 policies x 128 prompts
x 1 seed.

Scoring overhead on TinyStories:

- tuning: ref_w1 costs one GPT-2 Large forward pass per generated continuation (the NLL)
  plus Zstandard compression;
- test: SOL adds one GPT-2 Large encoding pass per generated story.


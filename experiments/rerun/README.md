# rerun: SOL scoring, Ψ and EB-churn samplers, successive-halving tuning

An inference-only rerun of [sc-vs-depth](../sc-vs-depth/),
[continuous-schedules](../continuous-schedules/) and [hybrid-sc](../hybrid-sc/) on the
checkpoints of their H100 reproduction (`blog-repro-h100`). It also covers
[sc-sampler-choice](../sc-sampler-choice/), which only reads the sc-vs-depth runs'
per-policy validation scores. block-encoder-decoder is not included.

The recipes, data cohorts, seeds and checkpoints are those of the source runs. The cases'
pinned configs equal the source runs' `protocol.json` configs apart from two
training-only keys that later code adds with their default values
(`prediction.fixed_feedback`, `prediction.selected_gradients`).

What changes:

1. **TinyStories score.** [SOL](../../docs/tasks/sol.md) against the real continuations
   of the same prompts (GPT-2 Large, 1024 x 1024 slices, lower is better) replaces BLEU.
   BLEU with and without its brevity penalty is still recorded. Sudoku (accuracy) and
   text8 (unique valid words of length 5 or more) keep their metrics.
2. **Samplers.** The five `hybrid_renoising` policies are kept exactly. Two policies are
   added to every case:
   - `eb_churn`: EB with random uncommitment, tuning rate, stop fraction, entropy bound
     and temperature (24 trials);
   - `psi`: the [Ψ-sampler](../../docs/components/psi-sampler.md), tuning schedule, η,
     loop t_on/t_off, top_p and temperature (24 trials). It applies to every corruption
     (Gaussian, simplex, spherical, vMF, masked, uniform and the hybrids).
3. **Tuning.** Successive halving at NFE 128. Every trial is scored on 64 tuning prompts,
   the best third on 256, the best third of those on 1024. The top 3 are then re-scored
   on the 1024 with a fresh seed (2041), with a SOL bootstrap interval on TinyStories.
4. **Test.** The overall validation winner is scored once on the test split at NFE 128.
   Afterwards, **each policy's** validation winner is scored once more on the test split
   at NFE 512 (`final-extra/nfe-512/<policy>/`). This is a reported measurement only; no
   selection reads it.

## Partitions

`evaluation/<task>.json`, drawn with `numpy.random.default_rng(20261111 + task)`. Each
test partition excludes the 128 tuning indices that the source runs' training callbacks
generated from.

| Task | Validation examples | Tuning pool | Test | Before (tuning / test) |
| --- | --- | --- | --- | --- |
| Sudoku-hard | 2,000 | 1,024 | 976 (all the rest) | 128 / 256 |
| text8 | 19,531 | 2,048 | 2,048 | 128 / 256 |
| TinyStories BPE-10k | 13,092 | 2,048 | 2,048 | 128 / 256 |

Halving rungs are prefixes of the tuning pool (in draw order). Tuning seeds are 41 and
42, test seeds 1041 and 1042. text8's size is the packed validation split
(5,000,000 characters in 256-character rows: 19,531). Its official test split is also
available (`evaluation.test_dataset: test`) if a still larger untouched test set is
wanted.

## Run

On the H100 machine, where `/data/sddc/runs/blog-repro-h100/<sweep>/` holds the source
runs (also in `s3://sddc-patrick-data-029636175027/runs/blog-repro-h100/`; override the
root with `SDDC_SOURCE_RUNS`):

```bash
for sweep in sc_vs_depth continuous_schedules hybrid_sc; do
  name=${sweep//_/-}-rerun
  uv run --locked --extra tuning --extra scoring python -m core.sweeps.prepare -m \
    +sweep=${sweep}_rerun preparation.output=$SDDC_STORAGE_ROOT/plans/$name
  uv run --locked --extra tuning --extra scoring python -m core.sweeps.prepare collect \
    $SDDC_STORAGE_ROOT/plans/$name
  uv run --locked --extra tuning --extra scoring python -m core.sweeps.nested_tuning \
    $SDDC_STORAGE_ROOT/plans/$name/manifest.json --check
done
```

then `nested_tuning ... --output $SDDC_STORAGE_ROOT/runs/<name> --address auto` per plan.
Every case reads `<source case>/training/candidates/44136fa355b3678a1146/training/checkpoints/last.ckpt`
(the single fixed training candidate, the same hash in all three sweeps). `--check` refuses
a missing checkpoint. GPT-2 Large downloads at its pinned revision on first scoring.

## Cost

These are sequence network calls (sequences x NFE) from `generation_cost`, an upper bound:
EB stops early. The current column is the source sweeps' own tuning plus test.

| Sweep (cases) | Tuning @128 | Test @128 (winner) | Per-policy test @512 | Total | Current | Ratio |
| --- | --- | --- | --- | --- | --- | --- |
| sc_vs_depth_rerun (27) | 256.6M | 11.7M | 327.2M | 595.5M | 47.8M | 12.5x |
| continuous_schedules_rerun (24) | 228.1M | 10.4M | 290.9M | 529.3M | 42.5M | 12.5x |
| hybrid_sc_rerun (48) | 456.1M | 20.8M | 581.7M | 1,058.6M | 84.9M | 12.5x |
| **All (99)** | **940.8M** | **42.9M** | **1,199.8M** | **2,183.4M** | **175.2M** | **12.5x** |

By task, the ratios are 9.5x for Sudoku (976 test puzzles) and 14.0x for text8 and
TinyStories. Tuning alone is 5.6x the current tuning: 7 policies with 112 trials,
each trial at most 64 + 256 + 1024 prompts, against 52 trials of 128 prompts. The
per-policy NFE-512 test measurement (7 policies x 2,048 x 2 seeds x 512 calls per case)
is 55% of the total. One test seed for it, or only the winner and the native policy, would
halve or more than halve it. SOL scoring adds one GPT-2 Large pass per generated story.

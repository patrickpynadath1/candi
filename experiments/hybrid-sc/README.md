# Hybrid views and self-conditioning

**Question.** [Continuous schedules](../continuous-schedules/) asks what self-conditioning
does to continuous diffusion. Does the answer change when the continuous path is paired with
a discrete view, masked or uniform, in one two-view (hybrid) model?

**Grid.** 2 discrete views (masked, uniform) x 4 continuous paths (Gaussian embedding,
simplex, spherical, spherical vMF) x {no SC, SC (`modifiers/feedback=expected_embedding`)}
x 3 tasks (Sudoku-hard, text8, TinyStories with the 10k BPE tokenizer) = 48 cases, one
training seed each.

**Protocol.** Identical to continuous-schedules, so its 24 cases are the no-discrete-view
baseline: the CDCD learned clock for the continuous view (view 0; the discrete view keeps the
shared uniform clock), the 8-layer / 512-wide backbone, 20k training steps, the
`hybrid_renoising` sampler policies tuned on validation, 128 network calls per sample, and
the same evaluation splits (`evaluation/` is a copy of continuous-schedules'). Only the
winner of each case is scored on the test split.

## Run

Data preparation is the same as for [continuous-schedules](../continuous-schedules/)
(text8 splits and the TinyStories cohort, `scripts/prepare_data.py`).

```bash
uv run --locked --extra tuning --extra scoring python -m core.sweeps.prepare -m \
  +sweep=hybrid_sc preparation.output=$SDDC_STORAGE_ROOT/plans/hybrid-sc
uv run --locked --extra tuning --extra scoring python -m core.sweeps.prepare collect \
  $SDDC_STORAGE_ROOT/plans/hybrid-sc
uv run --locked --extra tuning --extra scoring python -m core.sweeps.nested_tuning \
  $SDDC_STORAGE_ROOT/plans/hybrid-sc/manifest.json \
  --output $SDDC_STORAGE_ROOT/runs/hybrid-sc --address auto --case-concurrency 4
uv run --locked python -m core.analysis.status $SDDC_STORAGE_ROOT/runs/hybrid-sc
```

## Cost

Not yet measured. Expect about the continuous-schedules cost per case plus the extra view:
roughly 1 to 1.5 H100-hours per case including sampler tuning, about 50 to 70 H100-hours
for the grid.

## Results

None yet; there are no published numbers for this question, so there is no `expected.csv`
or `analyze.py` yet.

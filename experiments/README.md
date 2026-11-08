# Worked examples

Each directory answers one question from the blog post
[A Few Design Choices for Diffusion Language Models](https://patrickpynadath1.github.io/blog/diffusion-lm-design-choices)
with the codebase: the sweep that runs it, the evaluation protocol, the numbers we got, and
the analysis that turns run outputs into the post's tables. Rerunning an example should
give numbers close to ours. Most cells are a single training seed, so expect seed noise.

| Example | Question | Sweep |
| --- | --- | --- |
| [sc-vs-depth](sc-vs-depth/) | Is self-conditioning just a deeper network? | `+sweep=sc_vs_depth` |
| [sc-sampler-choice](sc-sampler-choice/) | Does self-conditioning change how the model should be sampled? | `+sweep=sc_sampler_choice` |
| [tokenizer-ladder](tokenizer-ladder/) | Does the tokenizer change the picture? | `+sweep=tokenizer_ladder` |
| [continuous-schedules](continuous-schedules/) | How do learned (CDCD) noise schedules spend training? | `+sweep=continuous_schedules` |
| [block-encoder-decoder](block-encoder-decoder/) | Block diffusion: scale the encoder or the decoder? | `+sweep=block_encoder_decoder` |

## Running an example

Every example uses the same three steps (see [nested tuning](../docs/guides/nested-tuning.md)):

```bash
# 1. Expand the sweep into a plan of cases (no training).
uv run --locked --extra tuning --extra scoring python -m core.sweeps.prepare -m \
  +sweep=sc_vs_depth preparation.output=$CANDI_STORAGE_ROOT/plans/sc-vs-depth
uv run --locked --extra tuning --extra scoring python -m core.sweeps.prepare collect \
  $CANDI_STORAGE_ROOT/plans/sc-vs-depth
# 2. Train each case and tune its samplers on validation; only the winner sees the test split.
uv run --locked --extra tuning --extra scoring python -m core.sweeps.nested_tuning \
  $CANDI_STORAGE_ROOT/plans/sc-vs-depth/manifest.json \
  --output $CANDI_STORAGE_ROOT/runs/sc-vs-depth --address auto --case-concurrency 4
# 3. Tables in the post's format, compared with the published numbers.
uv run --locked --extra scoring python experiments/sc-vs-depth/analyze.py \
  $CANDI_STORAGE_ROOT/runs/sc-vs-depth
```

Each README states the example's cost (GPU hours), its protocol, and how its numbers can
differ from the post's.

## What an example directory holds

| File | Contents |
| --- | --- |
| `README.md` | The question, the protocol, how to run, the cost, the results and caveats. |
| `evaluation/<task>.json` | The committed evaluation spec per task: tuning/test indices, score, seeds, task overrides (`preparation.evaluation_file`). |
| `expected.csv` | The published numbers, one row per cell, with a `note` where the published value came from a different setting. |
| `analyze.py` | Turns a run tree into the post's table (same JSON schema as the website's data) and compares it with `expected.csv`. |

The sweep itself is `configs/sweep/<name>.yaml`; Hydra needs it under `configs/`.
Analysis code shared by the examples is `core/analysis`.

## Rules for each example (also for contributors)

- **One protocol per example.** All of an example's cells share one training and evaluation
  protocol. Where a published number came from a different setting, say so in
  `expected.csv` and the README; do not mix protocols inside an example.
- **Named after the question.** Config groups and files use plain names (`objective=
  masked_elbo`), never campaign, machine or date names.
- **Self-contained.** Splits, tokenizers and settings needed to rerun live in the repo; no
  server paths, accounts or W&B entities.
- **Verified twice:** `analyze.py` reproduces the post's JSON from the stored runs, and a
  CPU smoke test (`core/tests/test_experiment_<name>.py`) prepares the sweep, runs one tiny
  case through nested tuning and analyzes it.

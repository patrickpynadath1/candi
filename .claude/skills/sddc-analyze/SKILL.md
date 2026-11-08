---
name: sddc-analyze
description: Turn SDDC nested-tuning run trees into result tables - run a worked example's analyze.py, compare against its expected.csv, read cases with core.analysis, and judge whether a rerun reproduces published numbers. Use after runs finish, when building figure or blog data, or when numbers look surprising.
---

# Analyze runs

## 1. A worked example

Each example in `experiments/` has an `analyze.py` that reads run roots and writes the
post's tables, then prints a comparison with its `expected.csv`:

```bash
uv run --locked --extra tuning --extra scoring python experiments/sc-vs-depth/analyze.py \
  $SDDC_STORAGE_ROOT/runs/sc-vs-depth --output $SDDC_STORAGE_ROOT/analysis/sc-vs-depth
```

| Example | Reads | Writes | Notes |
| --- | --- | --- | --- |
| `sc-vs-depth` | its runs | `sc_depth_results.json` | `--website FILE` diffs against a published JSON |
| `sc-sampler-choice` | sc-vs-depth's runs | `sampler_choice.json` | no runs of its own |
| `tokenizer-ladder` | its runs | `vocab_ladder.json` | |
| `continuous-schedules` | its runs | `cdcd_performance.json`, `cdcd_schedules.json` | schedules need the checkpoints; `--no-schedules` skips them |
| `block-encoder-decoder` | its runs | `block_scaling.json` | training hours come from file mtimes |

Several roots can be passed; cases with the same name are merged (`merge_cases`).

## 2. Ad hoc questions

```python
from core.analysis import load_cases, bleu_parts
cases = load_cases("runs/<name>")          # every protocol.json below the root
for case in cases:
    best, tied = case.winner()               # best validation policy (ties: first listed)
    print(case.name, case.selected, case.test, {p: v.validation for p, v in case.policies.items()})
    records = case.records()                 # {test seed: generated records} of the winner
```

`case.selected`/`case.test` come from `final/result.json`: the policy that won on
validation and its single test score. Never pick a policy by its test score.

## 3. Judge a rerun

- A rerun follows the same protocol, so numbers should be **close, not identical**: one
  training seed, different hardware and kernels. The example READMEs say how a rerun can
  differ; compare against their stated tolerances.
- Read the comparison column by column: a single-cell gap is usually noise; a whole row or a
  flipped ranking is a protocol difference (wrong split, NFE, policy set) until shown
  otherwise.
- Check the selection: the winner should be the best validation policy, the test split
  scored once, budgets matched (NFE), and the evaluation file the committed one.
- For BLEU on TinyStories, report BLEU with and without the brevity penalty and the length
  ratio (`bleu_parts`): a method can "win" on short outputs.

## Done when

`analyze.py` writes its files, every difference from `expected.csv` is within the README's
tolerance or explained, and the numbers you report trace to a `final/result.json`.

---
name: sddc-add-task
description: Add a dataset or task to SDDC - the Task class (datasets, collation, prompt/fixed-mask generation that never leaks answers, metrics on final tokens), its configs and task-suite entry, data preparation with checksums, and a committed evaluation split for sweeps. Use when a new benchmark, dataset, tokenizer or evaluation mode is needed.
---

# Add a task

Reference: `docs/extending/adding-a-task.md`, the task pages in `docs/tasks/`, and the
closest existing task (`core/tasks/nqueens.py` is small; `core/tasks/tinystories.py` covers
tokenizers and BLEU).

## 1. Implement `core/tasks/<name>.py`

Subclass `Task` from `core/tasks/base.py`:

| Method | Contract |
| --- | --- |
| `prepare_data()` | build or verify the cache; deterministic, checksummed if downloaded |
| `training_dataset()`, `validation_dataset()` | the splits; validation is what tuning sees |
| `evaluation_dataset(name)`, `prepare_evaluation(name)` | named evaluation sets and their settings |
| `collate_training(examples)` | `input_ids`, `loss_mask`, `corruption_mask` (the only fields the training wrapper forwards to `algorithm.loss`) |
| `prepare_generation(examples)` | the prompt tokens and `prompt_mask` (True = fixed). Positions to be generated must not carry the answer: fill them with a placeholder |
| `create_metric_accumulator(name)` | scores final token sequences only, independent of how they were generated; declares a `PrimaryMetric` |

Put dataset loading in `core/data/<name>.py`. Write caches under the configured
`data.cache_dir` (`${oc.env:SDDC_STORAGE_ROOT,...}/data_cache/...`), never into the repo.

## 2. Configure

- `configs/task/<name>.yaml`: the `_target_` and task fields; vocabulary through
  `token_space` (absorbing mask appended, native, or absent: `configs/information_path/token_space/`).
- `configs/data/<name>.yaml` for dataset settings.
- For sweeps, a task-suite entry `configs/task/battery/<name>.yaml` that sets
  `preparation.task_name`, the data cache and fixed sizes (copy `task/battery/text8.yaml`).
- Downloads: add a subcommand to `scripts/prepare_data.py` that pins the source revision and
  checks SHA-256; set `download: false` in the battery config so a missing cache fails
  loudly.

## 3. Evaluation split for sweeps

A committed `experiments/<example>/evaluation/<task_name>.json` (or a new file next to your
sweep) with `task`, `score {kind: metric, metric}`, `direction`, `seeds`, `test_seeds`,
`tuning_indices`, `test_indices`, and a `source` line saying how the indices were drawn.
Tuning and test indices must be disjoint and drawn from the validation set with a fixed seed.

## 4. Tests

Model them on `core/tests/test_nqueens.py` and `core/tests/test_tasks.py`:

- collation produces the three masks with the expected semantics;
- `prepare_generation` never exposes target tokens in non-prompt positions (assert it
  directly);
- the metric scores a known-correct and a known-wrong sequence correctly;
- one tiny train/evaluate through the runner (`core/tests/test_entrypoint.py` pattern).

## 5. Document

A page in `docs/tasks/`, linked from `docs/tasks/overview.md` and `docs/index.md`: data
source and license, splits, metric, and how to prepare the data.

## Done when

The task tests pass, `uv run --locked python main.py task=<name> trainer.max_steps=2 ...`
(tiny overrides as in skill `sddc-new-recipe`) trains and evaluates on CPU, and
`scripts/doctor.py` reports its data cache when present.

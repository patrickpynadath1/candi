# Adding a task

A Task owns the dataset problem and generation metrics, independently of the model,
corruption process, and sampler. Implement the interface in
[core/tasks/base.py](../../core/tasks/base.py) and configure it through `task`.
You should not need to modify the shared training runner for a new task.

## Define the data contract

| Method/property | Responsibility |
| --- | --- |
| `name`, `primary_metric` | Stable task name and `PrimaryMetric(name, mode, dataset)`. |
| `prepare_data()` | Prepare training/validation resources; compatible with Lightning preparation. |
| `training_dataset()`, `validation_dataset()` | Return training and loss-validation datasets. |
| `collate_training(examples)` | Return token IDs and masks consumed by algorithms. |
| `evaluation_dataset(name)` | Select the task's named generation evaluation set. |
| `prepare_evaluation(name)` | Optional evaluation-only resource preparation. |
| `prepare_generation(examples)` | Return prompt tokens and fixed-position mask without answer leakage. |
| `evaluation_batches(dataset, indices, batch_size)` | Optional batching policy, e.g. grouping compatible prompt lengths. |
| `create_metric_accumulator(name)` | Return an accumulator with `update` and `compute`. |

Training batches usually contain `input_ids` and `loss_mask`, with explicit
`corruption_mask` where needed. Collation may also expose `prompt_mask` or
`padding_mask`, but the shared training wrapper does not forward those fields to
`algorithm.loss`. Define special tokens and loss/corruption semantics precisely;
additional model inputs require an intentional integration change. Do not assume
universal variable-length or padding support.

For conditional generation, return only the prompt as fixed positions; answer
labels stay in the evaluation examples for scoring. For unconditional generation,
return an unfixed placeholder sequence of the requested shape. The sampler/process
initializes its own noisy state.

## Make metrics independent of the generator

The accumulator receives final generated token sequences, reference examples, and
stable indices. `compute()` returns a task-specific aggregate dataclass containing
record dataclasses; the evaluator serializes it with `dataclasses.asdict` and adds
execution metadata. A record should retain enough information to inspect failures.

Select a primary **generation** metric with a clear direction. Avoid requiring a
model likelihood for tasks expected to work with methods that do not define one.
The training loss validation pathway remains separate.

Reuse [text word metrics](../../core/tasks/metrics/text.py),
[causal reference scoring](../../core/tasks/metrics/generative.py), or
[GSM8K extraction](../../core/tasks/metrics/gsm8k.py) when their mathematics matches
the task. Code execution uses the reusable
[sandbox module](../../core/tasks/sandbox.py); read its restrictions before reusing
it. Do not duplicate those mechanisms inside a Task.

## Register configuration

Add `configs/task/<name>.yaml` with its `_target_` and explicit dependencies. Tasks
that own data and token-space selections can use this pattern:

```yaml
defaults:
  - /data@_global_.data: my_data
  - /token_space@_global_.token_space: append_mask
  - _self_
_target_: core.tasks.my_task.MyTask
data: ${data}
vocab_size: 100
length: 128
```

This is an illustrative new task, not a shipped config. Create the referenced
`data` option and choose the actual vocabulary/token-space policy. A native mask
or no-mask method may require a different token-space config. Keep the generic
`datamodule=task`, `evaluator=generation`, and `callback=task` when applicable.

## Test the boundaries

Verify tokenization and packing, deterministic split selection, no prompt-answer
leakage, and masking semantics. Test metric aggregation across uneven batches:
corpus-weighted perplexity is different from averaging per-sequence perplexities,
and unique counts across a corpus are different from sums of per-sample unique
counts.

Exercise one tiny train/checkpoint/evaluate cycle through the shared runner and
check that standalone evaluation does not prepare training data unnecessarily.
Offline fixtures or injected scorers make metric mathematics testable without
large downloads. A fixture-based smoke test should not be reported as a real-corpus
benchmark.

Use [TinyTask integration tests](../../core/tests/test_entrypoint.py),
[OpenWebText metric tests](../../core/tests/test_openwebtext.py), and
[task documentation](../tasks/overview.md) as concrete examples.

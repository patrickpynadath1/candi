# Tasks and generation metrics

A Task defines the data problem independently of the algorithm: datasets,
tokenization, training masks, generation conditioning, and the metric used to
judge generated samples. All task metrics are generation-based. A training or
validation objective is logged separately; it is not assumed to be a comparable
likelihood across methods.

| Task | Generation input | Primary metric | Config |
| --- | --- | --- | --- |
| [Sudoku](sudoku.md) | Puzzle prompt | Exact solution accuracy, maximize | `task=sudoku` (default) |
| [Text8](text8.md) | Unconditional | Training-vocabulary valid-word rate, maximize | `+experiment=text8` |
| [OpenWebText](openwebtext.md) | Unconditional | External-reference generative perplexity, minimize | `+experiment=openwebtext` |
| [TinyGSM](tinygsm.md) | GSM8K question | Numeric answer accuracy, maximize | `+experiment=tinygsm` |

The language experiment presets also set model and batch settings. Selecting a
task alone changes data/token-space defaults, not all architecture settings.
Maze and Zebra are not implemented in this tree.

## Boundary with the training and evaluation infrastructure

[Task](../../core/tasks/base.py) supplies `prepare_data`, `training_dataset`,
`validation_dataset`, `collate_training`, `prepare_generation`, and
`create_metric_accumulator`. A separate evaluation dataset can be selected by
`evaluation_dataset`; TinyGSM uses this to evaluate GSM8K without preparing its
training corpus. `evaluation_batches` can group examples by prompt length.

Training batches expose `input_ids`, `loss_mask`, `corruption_mask`,
`prompt_mask`, and `padding_mask`. True mask entries mean respectively: score
this position, permit corruption, preserve conditioning, and padding. Tasks
must sanitize generation inputs so a held-out answer is never passed to the
sampler. Unconditional tasks supply zero placeholders and no fixed positions.

The shared generation evaluator calls the sampler, passes generated sequences
and reference examples to a task accumulator, and serializes its aggregate
dataclass and per-example record dataclasses. It adds runtime information such
as model evaluation counts (NFE). `PrimaryMetric` names the scalar, its
`min`/`max` direction, and the evaluation dataset; it does not replace the other
stored measurements.

Reusable word and causal-reference metrics live in
[tasks/metrics](../../core/tasks/metrics/); generated-code execution lives in
[tasks/sandbox.py](../../core/tasks/sandbox.py). These helpers do not depend on a
specific diffusion algorithm.

See [sampling and evaluation](../guides/sampling-and-evaluation.md),
[adding a task](../extending/adding-a-task.md), and
[validation coverage](../validation/coverage.md).

# TinyGSM and GSM8K evaluation

[TinyGSMTask](../../core/tasks/tinygsm.py) trains on question/code pairs and
evaluates generated answers on GSM8K. These are separate datasets: standalone
GSM8K evaluation does not require loading TinyGSM training data.

```bash
uv run python main.py +experiment=tinygsm data.max_examples=10000
uv run python main.py +experiment=tinygsm mode=evaluate \
  evaluation.checkpoint=/path/to/last.ckpt evaluation.max_examples=128
```

## Data and masks

The [default config](../../configs/data/tinygsm.yaml) uses the SmolLM-135M
tokenizer, task vocabulary 49,153 and length 512. It loads finite TinyGSM data;
`data.max_examples` materializes a bounded subset. Streaming training is not
supported. `data.data_path` can supply a JSON list of `{question, code}` records.

Sequences are BOS + question + separator + code + EOS, padded to task length.
Defaults exclude prompts from loss/corruption, include padding in the legacy
training mask, and filter overlong examples. `train_on_prompt`, `train_on_pad`,
`filter_too_long`, and `wrap` are explicit options; filtering and wrapping cannot
both be enabled. Padding is also identified separately in `padding_mask`.
Validation uses `val_ratio=.01` and `val_seed=42`. Cache identity includes
tokenizer, separator, preparation settings, and local source content.

The evaluation dataset is named `gsm8k`. By default it uses the GSM8K test split;
`task.evaluation.data_path` accepts local JSON records with `prompt` and
`response_ground_truth`. Only the question prefix enters generation, and batches
group equal prompt lengths. Completions are cut at their first EOS before
scoring. Longer prompts consume the available generation window.

## Answer scoring and reusable sandbox

`task.evaluation.scoring=code` extracts Python and executes it through
[tasks/sandbox.py](../../core/tasks/sandbox.py). The scorer calls
`simple_math_problem()` if defined, parses its return value, then falls back to
a numeric answer from stdout after successful execution. `scoring=numeric`
parses generated text directly. Both compare the parsed answer against the gold
answer with absolute tolerance 1e−3.

The code sandbox uses Bubblewrap filesystem/network isolation, restricted
syntax and builtins, resource limits, and a timeout (default three seconds).
`math` is the permitted import. Install Bubblewrap and provide an environment
that permits its isolation primitives for code evaluation. Execution fails
closed when unavailable; there is no unsandboxed fallback. Failures are recorded
rather than silently interpreted as successful execution.

[TinyGSMRecord and TinyGSMAggregate](../../core/tasks/tinygsm_metrics.py) retain
prompts, completions, ground truth, correctness, EOS/truncation counts, mean
completion length, and parse/execution/timeout/sandbox-error rates. Accuracy is
the mean correctness. `task.evaluation.bootstrap_size>1` enables seeded
resampling for standard error and a percentile 95% interval. The default of one
reports zero standard error and endpoints equal to accuracy; it is not an
uncertainty estimate. Execution diagnostics are `null` in numeric mode.

Tests cover real sandbox execution and a small training/checkpoint/evaluation
workflow. Full training convergence or pretrained GSM8K accuracy reproduction
has not been established. See [coverage](../validation/coverage.md).

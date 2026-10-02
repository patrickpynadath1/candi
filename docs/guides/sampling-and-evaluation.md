# Sampling and evaluation

`mode=evaluate` and `mode=sample` both generate sequences and compute task metrics
through the shared evaluator. There is currently no separate metrics-free sample
export mode. Use the same architecture, vocabulary, representation, and modifiers
as training; a checkpoint does not automatically reconstruct its recipe for you.

## Evaluate a checkpoint

```bash
uv run python main.py mode=evaluate \
  evaluation.checkpoint=/absolute/path/to/mdlm_sudoku.ckpt \
  sampler=entropy_bounded sampler.entropy_bound=0.1 \
  hydra.run.dir=core/outputs/evaluation
```

That example assumes the default MDLM Sudoku architecture. Select the matching
experiment/config when evaluating another method. For a saved resolved config,
use the [checkpointing workflow](checkpointing.md) to avoid guessing dimensions.

The default selects EMA weights when available. Use final/raw weights explicitly:

```bash
uv run python main.py mode=evaluate \
  evaluation.checkpoint=/absolute/path/to/mdlm_sudoku.ckpt \
  evaluation.use_ema=false hydra.run.dir=core/outputs/evaluation_raw
```

Use this override explicitly for a checkpoint without EMA. Missing EMA state is
an error only if the configured module has EMA enabled; with `training.ema=0`,
the evaluator has no EMA object and already uses raw weights. `evaluation.device` is independent
of the training accelerator. `loader.eval_batch_size` controls generation batches;
`evaluation.max_examples` limits the selected task evaluation set. Text tasks also
have task-specific evaluation sample-count settings.

## Samplers and shaping

```bash
uv run python main.py mode=evaluate \
  evaluation.checkpoint=/absolute/path/to/mdlm_sudoku.ckpt \
  sampler=ancestral sampler.steps=180 parameterization.temperature=0.9
```

Sampler choices must match the method's process and predictor. Posterior inference
shaping belongs to the parameterization: temperature/top-k/top-p and commitment
settings are not sampler-owned logit transformations. The training pathway excludes
inference-only shaping. See [predictors](../components/predictors.md) and
[samplers](../components/samplers.md) for family-specific behavior.

For MDLM ancestral sampling, the default final greedy denoise adds one evaluation
beyond the requested reverse steps. Entropy-bounded sampling reports data-dependent
counts. Use measured `mean_nfe`, not `sampler.steps`, when comparing compute budgets.

## What gets written

The run directory contains a resolved `config.yaml` and `results.json` with:

- Task aggregate metrics and per-example generated records.
- Primary metric name, direction, and dataset identifier.
- Mean/min/max NFE, per-record NFE, and elapsed evaluation time.
- Checkpoint step, evaluation seed, parameterization settings, and sampler settings.

[Task pages](../tasks/overview.md) define the metrics. OpenWebText's generative
perplexity is measured by an external causal reference model, not by asking the
diffusion model for its own likelihood. The latest shared-BOS/EOS boundary behavior
is documented in [OpenWebText](../tasks/openwebtext.md).

The evaluator prepares only evaluation resources through the Task interface and
runs under a context that restores the previous model weights/mode afterwards.
Standalone generation is single-device; `trainer.devices` does not distribute it.
The BF16 autocast path is selected when `trainer.precision=bf16-mixed` on CUDA.
Other precision strings do not automatically provide additional generation
precision implementations.

## Reproducibility and comparisons

Keep checkpoint weights, task data, sample count, seeds, sampler semantics, and
precision fixed for a controlled comparison. Changing batching can change RNG
consumption even with the same seed. A small generation check establishes execution,
not a confidence interval or full benchmark result. Use [frontier sweeps](frontier-sweeps.md)
for paired grids and consult [validation coverage](../validation/coverage.md)
before claiming parity.

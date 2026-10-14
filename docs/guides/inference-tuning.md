# Inference hyperparameter tuning

Inference-only tuning is a fixed outer candidate in the unified nested tuner.
There is no separate inference execution engine. See [nested tuning](nested-tuning.md)
for the full configuration and training-search extension.

Build a manifest from completed sweep artifacts:

```bash
uv run --locked --extra tuning --extra scoring --extra tracking \
  python -m examples.tune_three_tasks \
  --all-methods \
  --results /tmp/three-task-results-so-far.csv \
  --output /tmp/tuning-manifest.json
uv run --locked --extra tuning --extra scoring --extra tracking \
  python -m core.sweeps.nested_tuning /tmp/tuning-manifest.json \
  --output /home/patrick/.cache/duo/nested-tuning/three-tasks \
  --concurrency 4
```

Start or connect a Ray cluster before launching. Install dependencies with
`uv sync --locked --extra tuning --extra scoring --extra tracking`. Start Ray
through `uv run` with those same extras so workers use the same environment.

Without `--all-methods`, the builder includes masked/uniform with SC off/on on
Sudoku, Text8 and TinyStories (12 checkpoints). With it, all 60 checkpoints from
the original and insertion sweeps are included, each as one outer candidate.
The full manifest contains 60 experiment cases and 156 sampler searches.

- Original methods: ancestral temperature 0.5–1.5 (16 candidates); top-1 ancestral
  final decoding greedy/sample (2 candidates); greedy EB entropy bound 0.02–0.5
  on a log scale (16 candidates). Defaults are explicit initial candidates.
- Insertion methods: their saved native sampler with sampled or greedy final
  decoding, each tuning temperature 0.5–1.5 (16 candidates).
- Fixed budget: 128 NFE; EB may terminate early. Insertion capacity, birth law and
  epsilon remain unchanged. No EB insertion sampler is substituted.
- Metrics: Sudoku accuracy; Text8 mean unique valid words of length at least five
  per sample (counts for length at least six also saved); TinyStories continuation
  corpus BLEU against references, excluding the prompt. No judge is run.
- Partitions: 128 tuning and 256 final examples, excluding previously evaluated
  examples; generation seeds 41/42 for tuning and 1041/1042 for final evaluation.

All sampler choices are selected on validation. Only the winning sampler and
settings for each checkpoint reach final evaluation. This differs from the old
engine, which tested every sampler separately. Lexical counts and BLEU are not
complete measures of coherence. The reserved final examples come from the existing
validation corpus; they are not a claim of an untouched external test corpus.

Saved historical results in `.cache/duo/inference-tuning/three-tasks-20261013` are
preserved. Do not treat interrupted historical trials as completed or pass their
old manifests directly to the new runner; regenerate the nested manifest and use
a new output root.

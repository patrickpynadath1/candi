# Training and inference hyperparameter selection

`python -m core.sweeps.nested_tuning` runs one workflow:

1. An outer Optuna candidate obtains a checkpoint.
2. Every configured sampler receives an inference search on validation examples.
3. The best validation score across those samplers is the outer candidate's score.
4. The outer optimizer selects the training parameters and checkpoint using that score.
5. The selected checkpoint, sampler and inference parameters receive one final evaluation.

Inference-only tuning uses exactly the same workflow: one outer candidate with
an empty training search space obtains an existing checkpoint. Only checkpoint
acquisition differs; there is no separate inference-only selection/evaluation path.

The driver holds no GPU reservation. Ray workers run training and generation;
a training worker exits before the inner search starts. Outer candidates execute
sequentially and inner evaluations use the requested concurrency.

## Configuration

A manifest is a JSON list of study specifications. The three-task builder now
emits this format directly. Here is a minimal inference-only study:

```python
spec = {
    "name": "my-method-my-task",
    "config": "/path/to/config.yaml",
    "training": {
        "checkpoint": "/path/to/last.ckpt",
        "space": {},
        "trials": 1,
    },
    "evaluation": {
        "dataset": "validation",
        "score": {"kind": "metric", "metric": "accuracy"},
        "direction": "maximize",
        "nfe": 128,
        "seeds": [41, 42],
        "test_seeds": [1041, 1042],
        "tuning_indices": [0, 1],  # Supply your full reserved validation IDs.
        "test_indices": [2, 3],    # Disjoint held-out IDs in the same dataset view.
    },
    "inference": [{
        "name": "ancestral",
        "sampler": {
            "_target_": "core.samplers.hybrid.HybridSampler",
            "steps": 127,
            "final_denoise": "greedy",
        },
        "budget_parameter": "steps",
        "space": {
            "algorithm.predictor.temperature": {
                "type": "float", "low": 0.5, "high": 1.5,
            },
        },
        "initial_points": [{"algorithm.predictor.temperature": 1.0}],
        "trials": 16,
    }],
    "resources": {"cpu": 4, "gpu": 1},
}
# json.dump([spec], open("nested-manifest.json", "w"), indent=2)
```

To search training hyperparameters, replace just `training`:

```python
spec["training"] = {
    "space": {
        "optimizer.lr": {"type": "float", "low": 1e-4, "high": 1e-3, "log": True},
        "optimizer.weight_decay": [0.0, 0.01, 0.1],
    },
    "trials": 8,
    "seed": 42,
    "initial_points": [{"optimizer.lr": 3e-4, "optimizer.weight_decay": 0.0}],
}
```

Training budget, training seed, task and method stay in the source config. Each
training candidate starts fresh (except retries of that same candidate). Training
search keys are paths to existing configuration fields. Use unresolved Hydra
configs when changing linked fields; in resolved snapshots, change the actual
consumed field, such as `algorithm.model.backbone.dropout`, rather than a detached
copy under `model`.

Both search levels support categorical lists, continuous float distributions and
integer distributions (`type: int`, inclusive `low`/`high`, optional `step` or
`log`). Add sampler policies to `inference` to compare multiple samplers. All
policies use the same validation metric and examples. Each has its own budget.

```bash
python -m core.sweeps.nested_tuning nested-manifest.json \
  --output /home/patrick/.cache/duo/nested-tuning \
  --concurrency 4
```

This connects to the existing Ray cluster. Each worker requests one GPU by default;
CPU-only tests can specify `resources: {"cpu": 1, "gpu": 0}`.

## Validation boundary and monitoring

Periodic generation callbacks remain enabled according to the training config.
A task adapter restricts **both generation callbacks and validation-loss batches**
to the specified validation IDs. Training data is unchanged. Validation workers
receive no final-test indices or seeds. Callback subset indices are local to the
restricted dataset; the training config records the mapping to original indices.

Outer candidates use the final training checkpoint (`last.ckpt`) and the inner
search's validation score. They never receive a test score. Final evaluation runs
once, after selecting the outer candidate and sampler. It does not feed back to
Optuna. The final evaluator uses the original task definition, not the restricted
training-time adapter, so final indices retain their original meaning.

The implementation currently selects scalar task metrics (including accuracy,
Text8 lexical counts and BLEU). It does not run judge scoring or early pruning.
Parallel Optuna proposals can depend on completion order despite fixed seeds.
Matching search budgets and checking winners across training seeds remain
experimental-design decisions.

## Persistence and current running jobs

Each study saves a local configuration snapshot and full protocol. Workers use
that snapshot, including when the checkpoint already exists. Each search
has an Optuna SQLite database, candidate artifacts and `selected.json`. Training
candidates save `trained.json`, the training checkpoint/config, and an unwrapped
`evaluation-config.yaml`; each contains its own inference studies. The root saves
`selected.json` and `final/result.json`, including the chosen configuration.

Rerunning the same manifest resumes interrupted candidates and reuses completed
training/evaluation stages. A failed validation trial stops selection until it is
successfully retried. Changing a protocol or source config requires a new study
name. File locks prevent concurrent drivers or duplicate candidates from writing
the same artifacts simultaneously. Checkpoints and output directories must be on
storage visible to every Ray worker. W&B logging uses the source config's tracking
settings through the existing training/evaluation runner. A top-level `wandb`
configuration can override the source config tracking settings.

The old standalone inference execution engine has been removed. All new searches
use this unified entry point, including inference-only studies. Historical sweep
results are preserved, but old Ray Tune execution state is not resumed by this
engine. Use a new output directory for a new run.

To retrain the three-task sweep with the same validation/final partitions and
sampler searches, including insertion variants:

```bash
uv sync --locked --extra tuning --extra scoring --extra tracking
uv run --locked --extra tuning --extra scoring --extra tracking \
  python -m examples.retrain_three_tasks \
  --previous-manifest /path/to/previous/manifest.json \
  --output /path/to/new-study \
  --project adaptive-three-tasks \
  --adaptive-continuous
uv run --locked --extra tuning --extra scoring --extra tracking \
  python -m core.sweeps.nested_tuning /path/to/new-study/manifest.json \
  --output /path/to/new-study --address auto \
  --study-concurrency 4 --concurrency 1
```

Start Ray through the same `uv run` command and extras. `--study-concurrency`
allows independent training/tuning studies to progress together; `--concurrency`
is the number of inference candidates per study. Ray reserves a GPU only for an
actual training or evaluation worker, never for the outer study coordinator.

`--adaptive-continuous` applies `configs/time_map/cdcd.yaml` to Gaussian,
blurring, simplex, and spherical views. It learns base-path progress from masked
per-example training cross-entropy, after 1,000 warmup updates, and refits every
50 updates once the 12,800-observation buffer is full. Gaussian's existing
adaptive wrapper is replaced by this equivalent view-clock placement, so it is
not warped twice. Discrete and insertion schedules remain as configured.
The learned clock is a registered module saved with the model and reused during
inference; validation and generation do not update it. Each study trains once
from scratch, then tunes inference, then evaluates only the validation-selected
winner. Existing checkpoints and study directories are not overwritten.

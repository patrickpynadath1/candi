# Base algorithm sweep

[`examples/base_algorithm_sweep.py`](../../examples/base_algorithm_sweep.py) defines
concrete `TrainingJob`/`InferenceJob` dataclasses and runs them through the dynamic
Ray Tune queue. Training completion adds inference jobs immediately. This catalogue
covers the eight families below; it does not add autoregressive, SEDD, D3PM, DUO,
or hybrid experiments.

Preview without preparing data, constructing models or starting Ray:

```bash
uv run python examples/base_algorithm_sweep.py
uv run python examples/base_algorithm_sweep.py --seeds 1 --verbose
```

## Catalogue

| Family | Training variants before seed/conditioning expansion | Inference per checkpoint |
| --- | --- | --- |
| Gaussian | 20: one-hot analytic rank × linear/geometric, one-hot argmax-error linear/geometric, embedding rank linear/geometric, exponential VE and linear interpolation on both representations; each with fixed/CDCD time mapping | Euler and Heun × step counts |
| Simplex | 2: reference constant-linear concentration × binned adaptive time sampling off/on | Step counts × churn 0/1 |
| Spherical flow | 4: binned CDCD clock off/on × truncation off/on | Spherical sampler × step counts |
| Uniform plug-in | 2: fixed/CDCD keep-rate schedules | Uniform ancestral × step counts; entropy-bounded × bounds |
| Uniform posterior + CE | 2: fixed/CDCD keep-rate schedules | Uniform ancestral × step counts; entropy-bounded × bounds |
| MDLM | 2: fixed/CDCD keep-rate schedules | Ancestral × step counts; entropy-bounded × bounds |
| RePlaid | 1: reference learned schedule, allocation and objective | Reference Gaussian DDPM sampler × step counts |
| Blurring | 1: reference position/feature blurring (`feature_rho=.5`) | Blur ancestral × step counts |

The simplex concentration is `constant_linear`, with `nu0=.4`, `nu1=.75` and
`pivot=.8`, matching the copied Sudoku reference and a concentration setting
reported in the [SDM paper](https://arxiv.org/html/2609.35553v1). The blurring
choice matches the original `blurring_diffusion` preset (`feature_rho=.5`,
`time_warp=linearized_crossings`).

The only generic adaptive wrapper swept is the binned CDCD-style fitter;
S-FLM spline-fit clocks are excluded, including for spherical flow. Simplex
retains its native binned loss-derivative time sampler. RePlaid retains its own
learned schedule, which is part of that method.

There is **no coefficient-rescaling axis**. `gaussian_rank`/`one_hot_rank` aliases
are not duplicated, and learned RePlaid schedules are not assigned to ordinary
Gaussian CE. Rank normalization, embedding dimensions/trainability and other
continuous hyperparameters retain their preset values. This is a finite catalogue
of named method choices, not every possible numerical hyperparameter combination.
Add choices to `method_variants`, `gaussian_variants` or `inference_jobs` to extend it.

Defaults are Sudoku hard with 200,000 training and 10,000 validation puzzles,
seed 1, 50,000 optimizer steps, effective batch 256,
evaluation batch 64, and 2,000 evaluation examples. Sampling steps are 32/64/128;
entropy bounds are .05/.1/.2. Other optimization, model, endpoint and precision
settings retain their method presets. Equal steps do not ensure equal training
compute; Heun and Euler at equal integration steps also have different model-call
counts.

## Self-conditioning and weights

Every variant requests self-conditioning off/on. On uses hidden feedback at
probability .9, except **RePlaid retains its reference posterior-mean feedback at
.25 with fixed-count selection** and its specialized gradient handling. RePlaid
inference keeps bootstrap feedback and final-transition noise; it evaluates raw
final weights (`evaluation.use_ema=false`). Other families retain their default
EMA evaluation setting. Every job selects `last.ckpt`.

Uniform plug-in supports hidden-state and expected-embedding self-conditioning.
Expected embeddings use the unshaped normalized plug-in weights as a heuristic;
they are not interpreted as a true posterior. The catalogue uses hidden feedback.
The default one-seed catalogue contains 68 training jobs and 372 potential inference
jobs, with no blocked combinations.

## Launch and restore

Run the supported catalogue on GPUs allocated to you:

```bash
CUDA_VISIBLE_DEVICES=2,3 uv run --extra tracking python examples/base_algorithm_sweep.py \
  --run --skip-unsupported --name base-algorithms
```

The script disables Ray's automatic uv project packaging for local workers, which
use the existing checkout/environment and data-cache links. Provision Sudoku data
as for normal training. The default combined concurrency limit is two, with one
GPU per job, four CPUs for training and two for inference. Adjust
`--max-concurrent`, `--train-cpus`, `--inference-cpus` and `--data-workers` as needed.

A smaller selected run can be configured independently:

```bash
CUDA_VISIBLE_DEVICES=2,3 uv run --extra tracking python examples/base_algorithm_sweep.py \
  --run --families mdlm replaid blurring --seeds 1 \
  --train-steps 2000 --sampling-steps 64 --max-examples 200 \
  --name initial-check
```

`--self-conditioning off` or `on` restricts that axis. The short-run example keeps
preset warmup lengths; it is an execution/learning pilot, not a convergence claim.

Outputs live under `core/outputs/ray/<name>` by default (`--storage` changes the
parent). `job-plan.json` records the exact configurations, blocked jobs and a plan
fingerprint. Existing directories require `--resume`; the current plan must match
its saved fingerprint. Repeat the original command with `--resume` to restore it.
Changed seeds, methods, budgets or inference settings require a new name. Native
Tune state owns the actual queue and execution status; `job-plan.json` is an audit
record, not a second scheduler.

See [programmatic sweeps](programmatic-sweeps.md) for queue persistence, terminal
failure behavior, shared-filesystem requirements and intermediate-reporting limits.

## W&B logging

Training jobs enable W&B by default: project `candi`, group equal to `--name`,
and a run name equal to the training job name. Install/use the tracking extra
with `uv run --extra tracking python examples/base_algorithm_sweep.py ... --run`.
Authenticate once with `uv run --extra tracking wandb login`.
`--wandb-project`, `--wandb-entity`, and `--wandb-mode online|offline|disabled`
control tracking. Previewing the plan does not create W&B runs.

W&B receives the resolved configuration, training loss, learning rate, validation
objective, and periodic task generation metrics. Each periodic evaluation also
logs a `<task>/samples` table and an evaluation artifact containing the complete
JSON records. Run summaries include `final_checkpoint` and
`completed_training_steps`. Standalone inference jobs currently report through
Tune and `results.json`, not W&B.

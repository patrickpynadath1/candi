# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

SDDC (Some DLM Design Choices; formerly CANDI v2.1) is a framework for controlled diffusion-language-model experiments, not a single algorithm. Recipes compose an **information path** (representation, corruption process, noise schedule), a **denoising task** (backbone, output parameterization, objective, training time sampler) and **inference settings** (initialization, inference time/noise grid, update rule, terminal decoding); see `docs/concepts/recipe.md`. Library code lives in `core/`, Hydra configs in `configs/`, docs in `docs/` (start at `docs/index.md`). Legacy implementations exist only in Git history before the `candiv2.1` cleanup. PRs target `candiv2.1`, not `main`.

## Server rules

Never stop clusters / `ray stop` / kill jobs, delete caches or overwrite results without explicit authorization. Check live state rather than trusting notes.

## Remote execution (laptop → `ruqi6000`)

This checkout is the laptop; it cannot run the CUDA environment. GPU work goes to the server `ssh ruqi6000` through Ray Jobs; see `docs/guides/remote-execution.md`.

```bash
alias sddc-remote="uv run -q --no-project --python 3.10 --with 'ray[default]==2.59.0' python scripts/cluster/sddc_remote.py"
sddc-remote tunnel &                                   # SSH-forwards the Jobs API on port 8275
sddc-remote submit --name NAME [--follow] -- python main.py ...   # also: list/status/logs/stop/gpus
```

- **Server disk:** `/` (incl. `/home/patrick`, `/tmp`) is nearly full. Everything persistent (datasets, checkpoints, logs, plans, envs, Ray temp, source snapshots) goes under `SDDC_STORAGE_ROOT` = `~/.cache/duo` (on `/data`). Never write to `/tmp` or elsewhere on `/`.
- `scripts/cluster/entrypoint.sh` runs every job from a persistent content-addressed snapshot (`sources/`) with a per-`uv.lock` env (`envs/`), and links `core/data_cache`/`core/outputs` to the storage root. Prepare plans on the server (via a submitted `core.sweeps.prepare` job) so manifests never contain laptop paths.
- Configs refer to server storage as `${oc.env:SDDC_STORAGE_ROOT,${oc.env:HOME}/.cache/duo}`; don't hard-code `/home/patrick/...`.
- Read-only inspection over `ssh ruqi6000` is fine. Starting/stopping clusters, submitting jobs or deleting server files needs the user's go-ahead.

## Commands

On the server, always use `uv run --locked` (Python 3.10, lock pinned to the PyTorch CUDA 12.1 index); never system Python. On the laptop the full env isn't available; tests that don't need torch can run with `uv run --no-project --python 3.10 --with pytest python -m pytest ...` (e.g. `core/tests/test_remote_submission.py`).

```bash
uv sync --locked --extra tuning --extra scoring --extra tracking --extra mazes
uv run --locked --extra tuning --extra scoring --extra tracking --extra mazes pytest -q core/tests
uv run --locked pytest -q core/tests/test_mdlm.py::test_name      # single test
uv run --locked ruff check core main.py frontier.py scripts experiments

uv run --locked python main.py +experiment=smoke               # CPU smoke run: 2 steps of tiny MDLM on Sudoku
uv run --locked python main.py +experiment=<name> --cfg job    # print composed config without running
uv run --locked python main.py mode=evaluate evaluation.checkpoint=/path/last.ckpt   # mode: train|evaluate|sample
uv run --locked python frontier.py evaluation.checkpoint=/path/last.ckpt            # pretrained sampling grid
```

Experiment workflow (same extras on install, Ray launcher, and driver):

```bash
# 1. Expand a sweep config (configs/sweep/) into a plan of cases; never launches training
uv run --locked --extra tuning --extra scoring --extra tracking \
  python -m core.sweeps.prepare -m +sweep=feedback preparation.output=/path/plan
uv run --locked --extra tuning --extra scoring --extra tracking \
  python -m core.sweeps.prepare collect /path/plan
# 2. Run the manifest with nested Optuna tuning on Ray
uv run --locked --extra tuning --extra scoring --extra tracking \
  python -m core.sweeps.nested_tuning MANIFEST --output OUTPUT \
  --address auto --case-concurrency 4 --concurrency 1
```

The default `main.py` (no overrides) is a full GPU MDLM Sudoku-hard training run; don't launch it casually. Outputs go to `core/outputs/`, dataset caches to `core/data_cache/`; both are Git-ignored and may be symlinks to other volumes. Tests start their own local Ray clusters and must not attach to live training clusters. Some sampler tests compare against snapshots in `core/tests/fixtures/`.

## Configuration model

- `main.py` → `core/training/runner.py:run`, composed from `configs/config.yaml`; `frontier.py` uses `configs/frontier.yaml`.
- Each component is a Hydra group with a `_target_`; algorithm configs reference `${model}`, `${process}`, `${parameterization}`, etc. Constructors receive explicit dependencies, never the whole root config.
- Switching mathematical families requires a full recipe in `configs/experiment/` (`# @package _global_`, `override /group: ...`). Changing only `algorithm=` does **not** swap the matching process/model/schedule/sampler. Copy a verified recipe and check with `--cfg job`. Paper-reproduction presets are documented in `docs/guides/paper-recipes.md`.
- Experiment-level config groups: `configs/sweep/` (Hydra multirun case expansion for `core.sweeps.prepare`), `configs/ablation_path/`, `configs/modifiers/feedback/`, `configs/information_path/time_map/` (e.g. CDCD loss-adaptive clock). `configs/task/battery/` and the `algorithm_definition` block in prepared configs are legacy names for "task suite" and "recipe definition".
- Resolved saved run configs have literal `output_dir` and no group defaults: when re-evaluating with `--config-path <run> --config-name config`, override both `output_dir` and `hydra.run.dir`.
- Shared modifiers are toggled via top-level blocks in `config.yaml`: `self_conditioning`, `block_diffusion`, `token_feedback`, `variable_length`.

## Architecture (ownership boundaries)

See `docs/concepts/architecture.md` and `docs/concepts/component-contracts.md` before changing a boundary.

- **Algorithm** (`core/algorithms/`): training-loss assembly and model→prediction endpoints. `loss(tokens, loss_mask, corruption_mask=...)` returns `{loss, loss_sum, num_tokens}`. Does *not* own the reverse-time loop; `generate` delegates to a sampler. Posterior methods share `core/algorithms/posterior.py` and the prediction coordinator in `core/modifiers/self_conditioning/prediction.py`; reuse these instead of adding new forward paths.
- **Sampler** (`core/samplers/`): state init, time progression, stopping, NFE accounting (NFE includes terminal calls; it's not the step count). Fixed-grid samplers share `loop.py`; `composed.py`, `policies.py` and `correctors.py` compose update rules and commitment policies. Entropy-bounded, adaptive ODE and AR keep specialized control flow.
- **Math components**: `processes/` (corruption + reverse transitions, incl. GIDD, vMF, Triton simplex), `noise_schedules/` (incl. rank-linear, Gumbel, adaptive), `time_sampling/` (incl. CDCD), `representations/`, `parameterizations/` (posterior vs plug-in vs score vs Gaussian-prior; same `[B,L,V]` shape does not mean interchangeable), `objectives/` (loss math, composed via `composition.py`; reductions may live in the algorithm).
- **Models** (`core/models/`): all share `TextDiT` execution with replaceable input/output/conditioning components; causal models keep their own KV-cache path. Public constructors keep parameter paths stable so existing checkpoints load.
- **Training** (`core/training/`): Lightning `TrainingModule` owns optimizer, EMA, checkpointing, logging, `post_optimizer_step` hooks, not diffusion equations. It forwards only `input_ids`, `loss_mask`, `corruption_mask` to `algorithm.loss` (not `prompt_mask`/`padding_mask`). Validation loss is logged as `val/nll` even when not a likelihood.
- **Tasks** (`core/tasks/`, data in `core/data/`): implement `core/tasks/base.py`; own datasets, collation, `prepare_generation` (prompt + fixed mask, never leaking answers), and metric accumulators that score final token sequences independent of the generator.
- **Modifiers** (`core/modifiers/`): self-conditioning (`posterior_mean` = expected embedding from previous logits; `hidden` = hidden-state feedback, a separate ablation), block diffusion (incl. block-size mixtures / BlockGen), hybrid multi-view (`DiffusionView` branches with per-branch process/schedule/clock, fused in `fusion.py`), token feedback, variable length (insertion during generation, deletion during forward corruption; birth state must match the process).
- **Sweeps** (`core/sweeps/`): thin layer over Ray tasks + Optuna with no custom scheduler. `prepare.py` exports Hydra multiruns into a plan/manifest; `nested_tuning.py` runs each case (recipe × task × training seed) as an outer Optuna study over training candidates plus inner per-sampler inference studies, selected on validation only, then evaluates only the winner on reserved final indices/seeds. Inference-only tuning is the one-fixed-checkpoint case of the same path. `configuration.py` composes presets and prepares each worker's run config (`prepare_run`, 0 or 1 GPU). The old Ray Tune queue (`run_trial`, `TrainingInferenceSearch`, live submissions) was removed; it exists only in Git history. Terminology (experiment, case, study, trial, run, job) is defined in `docs/guides/experiments.md`.

## Conventions

- Contracts are enforced by constructor arguments and runtime checks, not static typing (a strict-typing migration was reverted). Not all combinations are compatible or tested; see `docs/validation/coverage.md`.
- Mask semantics: `loss_mask` = contributes to loss, `corruption_mask` = may be corrupted, `prompt_mask` True = fixed during generation.
- Training and sampling prediction paths are separate so temperature/top-k/top-p never alter the training objective.
- Prefer Hydra composition and native Ray/Optuna machinery over campaign-specific execution scripts. Changing an existing experiment's protocol requires new case/output names; `nested_tuning` checks stored `protocol.json` compatibility on resume.
- New components: YAML in the relevant `configs/` group plus a doc page under `docs/` (see `docs/extending/`). Add a new algorithm class only when orchestration genuinely differs; most new methods are recipes.

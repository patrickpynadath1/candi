# Agent handoff: local development and remote Ray execution

## Purpose and working rules

CANDI v2.1 is a framework for controlled diffusion-language-model experiments,
not a single CANDI algorithm. Recipes compose information paths, denoising tasks,
and inference choices. Keep components reusable; prefer Hydra composition and
native Ray/Optuna machinery over campaign-specific execution scripts.

The current workstream is to move development and Codex to the user's local
computer while running training and evaluation on the existing GPU server via
Ray Jobs. Preserve all existing data, results, checkpoints, and provenance.
This document records the server state checked on **2026-10-29**; live status
must be checked again rather than inferred from this snapshot.

- Use `uv` and `uv.lock`; do not substitute system Python or an arbitrary venv.
- Do not stop clusters, kill training jobs, delete caches, regenerate datasets,
  or restart campaigns as part of setup without task-specific authorization.
- Do not overwrite completed results or silently change an existing experiment's
  protocol. Resume only compatible work; use new case/output names for changes.
- Never upload credentials, datasets, checkpoints, `.venv`, or large results as
  part of a Ray source bundle. Keep credentials server-side.
- Do not assume an apparently old worktree/environment is unused. In particular,
  the `54f2` environment below is still used by active jobs.

## Repository and verified environment

- Remote: `https://github.com/patrickpynadath1/candi.git`.
- Main development destination for this conversation: `candiv2.1`, not `main`.
- Migration branch requested by the user: `dev-local-to-cluster`.
- Active server checkout: `/tmp/candi-distillation-recipe`.
  Its original local branch was `codex/distillation-recipe`.
- Older checkout `/home/patrick/.codex/worktrees/54f2/duo` contains unrelated local
  work. Do not replace it with this checkout or sweep its changes into commits.
- Starting code revision for this handoff: `0cd4891` (pushed to `candiv2.1`).
  Preceding commits: `1888a2b` insertion policies/hazard weighting,
  `00a914d` feedback and insertion experiment configs.
- Python 3.10.9, Ray 2.59.0, PyTorch 2.5.1+cu121.
- Current shared environment: `/home/patrick/.codex/worktrees/54f2/duo/.venv`.
- uv executable: `/home/patrick/.local/bin/uv`.

Current server invocation pattern (historical environment dependency, not the
intended portable end state):

```bash
UV_PROJECT_ENVIRONMENT=/home/patrick/.codex/worktrees/54f2/duo/.venv \
  /home/patrick/.local/bin/uv run --locked \
  --extra tuning --extra scoring --extra tracking \
  python -m core.sweeps.nested_tuning MANIFEST --output OUTPUT \
  --address auto --case-concurrency 4 --concurrency 1
```

## Current campaign: feedback versus depth

Question: what does self-conditioning contribute compared with changing the
information path or doubling transformer depth? Here **self-conditioning means
expected embedding from previous token logits** (`posterior_mean`). Hidden-state
feedback (`hidden`) is a separate ablation, not interchangeable terminology.

Campaign: `feedback-depth-20k-20261028`, 42 cases = 14 recipes × 3 tasks × 1 seed.

- Pure Gaussian learned embeddings, masked, and uniform posterior CE:
  each with no feedback, expected-embedding SC, hidden feedback, or double depth
  without feedback (12 recipes).
- Gaussian learned embeddings + masked hybrid: no feedback and expected-
  embedding SC (2 recipes).
- Tasks: Sudoku hard, text8, TinyStories. No mazes/N-Queens in this campaign.
- Width 512, 8 heads, 8 layers (16 for double depth), batch 256, 20k updates,
  2,500 warmup steps, learning rate 3e-4, EMA .9999, BF16, torch.compile.
- Gaussian embeddings are trainable, unit-normalized, dimension 128. Continuous
  views use adaptive clocks. SC probability .9 training / 1 evaluation; detached
  first-pass states.
- Fixed 128 NFE. Separate validation tuning per sampler: native, native greedy,
  EB greedy. Select using validation only; final held-out indices/seeds differ.
- Metrics: Sudoku solve accuracy; text8 mean per-sample unique valid words of
  length at least 5; TinyStories reference BLEU (not a judge).
- Validation seeds 41/42; final seeds 1041/1042. These are generation seeds;
  there is only one training seed, so do not claim significance across seeds.
- W&B: https://wandb.ai/rz-text-dlp/feedback-depth-20k-20261028
- Configs: `configs/sweep/feedback.yaml`, `feedback_hybrid.yaml`,
  `configs/ablation_path/`, `configs/feedback/`.
- Details/preparation commands: `docs/guides/feedback-depth-ablation.md`.

At inventory: 35 new final results + 3 reused = **38/42**. Sudoku and text8
are complete (14 each); TinyStories has 10/14 final results. Unfinished names:

- `tinystories-uniform_posterior_ce-double_depth-seed-1`
- `tinystories-uniform_posterior_ce-expected_embedding-seed-1`
- `tinystories-uniform_posterior_ce-hidden-seed-1`
- `tinystories-uniform_posterior_ce-none-seed-1`

This identifies missing final artifacts, not whether a particular case is
training, tuning, or queued. Inspect logs and candidate state for that distinction.

## Storage and provenance: preserve these

Logical root `/home/patrick/.cache/duo` resolves to physical root
`/data/remote_cache/patrick/duo`. Existing JSON/configs contain **both absolute
path forms**. Preserve the alias and underlying mount; do not copy one over the
other or treat them as independent datasets.

| Contents | Location under `/home/patrick/.cache/duo` |
| --- | --- |
| Main campaign plans | `plans/feedback-depth-20k-20261028/` |
| Main campaign artifacts | `nested-tuning/feedback-depth-20k-20261028/` |
| Source of reused Sudoku runs | `nested-tuning/local-diffusion-20k-20261027/` |
| Dataset caches | `data_cache/` |
| Existing Sudoku dataset | `data_cache/sudoku_hard_train48000_valid2000_seed42/` |
| TinyStories cache | `data_cache/tinystories/` |
| Ray runtime files | `ray/` (main campaign uses `ray/fd28`) |
| Additional tracking artifacts | `wandb-data/`, plus per-run `wandb/` directories |
| Older experiments | `nested-tuning/`, `inference-tuning/`, `batteries/`, `worktrees/`, `new_runs/`, `new_eval_results/` |

Do not infer that the older directories are disposable. Text8 paths and any
external dataset/tokenizer caches must be inventoried from resolved run configs;
not every dataset dependency is guaranteed to live in `data_cache/`.

Main plan files: `manifest.json` (42 cases), `pending-manifest.json` (39 newly
submitted cases), `reused-results.json` (3 reuse references), `cases/`, `configs/`,
`hydra/`, `training-jobs.yaml`. Reuse references also exist in the result root.
The reused Sudoku cases are Gaussian expected-embedding SC and hybrid with/without
SC. **Do not resubmit all 42 as new training jobs.**

Within each result case, preserve the entire tree, including:

- `protocol.json`, pinned `config.yaml`, and `.driver.lock` handling.
- `training/candidates/<hash>/trained.json` and `training/config.yaml`,
  `training/evaluation-config.yaml`, `training/checkpoints/last.ckpt` and other
  checkpoints, metrics CSVs, W&B metadata.
- Inference candidate directories and `optuna.sqlite` databases (also any SQLite
  journal/WAL sidecars). Do not copy live databases without a consistent backup.
- `final/result.json` and per-seed generations/metrics under `final/seed-*`.
  `selection` records the checkpoint, chosen policy, parameters and validation
  score; top-level `score` is the final evaluation score.
- Root `launch.json`, `driver.log`, `driver.pid`, and archived attempt logs.

Other completed insertion experiments use the same `plans/<name>` and
`nested-tuning/<name>` layout:

- `insertion-before-after-text8-20261028`: two matched 2k-update training runs.
- `insertion-crossover-text8-20261028`: eight comparison cells, six new and two
  reused evaluations.
- `insertion-policies-text8-20261028`: ten cells, eight new and two reused.

For the latter two, `comparison.json` lives in the **plan** directory and links
reused results. Keep the before/after plan's frozen `source/` snapshot/provenance.
W&B project: `rz-text-dlp/insertion-before-after-text8-20261028` (separate groups).
Do not compare these 2k-update results as if they were the main 20k campaign.

Schedule plots/data from this conversation are outside the repo, under
`/home/patrick/.codex/visualizations/2026/10/11/01a12d34-6656-78b0-b299-814261a77f1e/`:
`gaussian-schedules/` and `hybrid-schedules/` contain PNG/PDF, curves CSV, and
checkpoint metadata. Preserve/export these separately if moving conversation assets.

## Running server state and unresolved research issue

The main driver was live at inventory: uv PID 944058, Python PID 944062. PIDs are
historical references only: verify command lines before acting on them.
Authoritative launch record is the campaign's `launch.json`.

- GPUs 0,1,2,5; 4 concurrent cases; inference concurrency 1 per case.
- Original driver used `--address local` to create its own cluster.
- `RAY_TMPDIR=/home/patrick/.cache/duo/ray/fd28`; longer paths previously hit the
  Unix socket pathname limit. Keep Ray temp paths short.
- `RAY_ENABLE_UV_RUN_RUNTIME_ENV=0` and the shared `UV_PROJECT_ENVIRONMENT` were
  used by that launch. These are legacy choices to deliberately replace and
  verify, not blindly carry into remote Jobs submissions.
- Object store cap `RAY_DEFAULT_OBJECT_STORE_MAX_MEMORY_BYTES=2147483648`,
  `OMP_NUM_THREADS=4`, `RAY_USAGE_STATS_ENABLED=0`.
- Host has 8 RTX A6000 GPUs (48 GiB class). Other users/processes use this server;
  several independent Ray clusters exist. Do not run blanket `ray stop`.

Unresolved: extracted hybrid Gaussian schedules jump to sigma about .97 almost
immediately on all three tasks, with/without SC. The user suspects loss feedback
concentrated at one endpoint. This is **not diagnosed or fixed**. The plotted
reference is the existing random-unit-embedding rank-linear approximation;
normalized pairwise rank differs from vocabulary-dependent outranking count.
Do not infer from the plots alone that the Gaussian view is useless. Incoming
CDCD work includes `core/time_sampling/cdcd.py`; do not silently migrate old
checkpoints or claim it fixes the older `AdaptiveTimeMap` runs.

## Desired local-to-cluster design

1. Source code and Codex live on the laptop. A local CLI/SDK submits a source
   snapshot and experiment config to the server's **Ray Jobs API**.
2. The sweep driver and all workers run on the server. Laptop disconnects do not
   end submitted jobs. Multiple submissions share a persistent cluster.
3. Reuse Hydra preparation, the existing nested Optuna workflow, and Ray resource
   scheduling. Inference-only tuning remains the existing-checkpoint case of the
   same search workflow. Preserve validation-only selection and final evaluation.
4. Keep datasets/checkpoints/results on persistent server storage, outside Ray's
   temporary uploaded working directory. Multi-node expansion would require
   shared accessible storage; current paths are single-host assumptions.
5. New uploads are snapshots; later local edits must not change queued/running
   jobs. Record source revision plus dirty-source identity, resolved configs,
   lockfile identity, job/submission ID, output root and W&B IDs.
6. Provide advisory GPU status: actual free VRAM/utilization/process ownership
   via NVIDIA tools, alongside Ray reservations. Flag externally busy GPUs;
   this is not an atomic reservation or an enforced memory partition.

Ray reservations coordinate jobs submitted through the same cluster. External
CUDA processes can still contend. `CUDA_VISIBLE_DEVICES=0,1,2,5` at cluster startup
limits its GPU pool; it does not reserve those GPUs against other users. GPU IDs
are remapped inside workers. Do not advertise fractional GPUs as memory isolation;
the current nested runner validates per-worker GPU requests as 0 or 1.

## Conversion work still to implement

- Install matching Ray Jobs/dashboard/client dependencies (current dependency is
  `ray[tune]`; ensure server has the dashboard extras). Pin compatible Ray and
  Python versions. The laptop needs only submission tools, not CUDA PyTorch.
- Run the server cluster as a persistent service with an explicit GPU pool,
  short temp directory, uv on PATH, and server-side credentials.
- Access dashboard/Jobs HTTP port 8265 through SSH forwarding/private networking;
  a typical tunnel is `ssh -N -L 8265:127.0.0.1:8265 patrick@SERVER`. Server hostname
  and laptop OS/path are not yet specified. Do not invent them.
- Upload source/configs/lockfile with Ray `working_dir` and explicit exclusions.
  Audit ignored generated plan files: a manifest is useless if its referenced
  configs were excluded or still point at a laptop path. Either generate plans
  remotely or bundle them with portable references through the existing tooling.
- Use `uv run --locked` for the submitted driver and verify worker inheritance of
  that environment. Replace the worktree-specific venv dependency for **new**
  jobs only after testing. Do not mutate the active jobs' environment.
- Use `--address auto` inside the remote driver, not the previous `local` launch
  flag. The HTTP Jobs address is distinct from the internal Ray cluster address.
- Add configurable server storage roots instead of scattered user-specific paths.
  Preserve old absolute paths/aliases for resume. Do not mutate stored protocols
  to bypass `_run_case`'s compatibility check.
- Keep the driver lightweight; reserve GPUs for worker tasks, not the coordinator.
- Support submit/status/logs/stop via native Ray Jobs APIs; a thin convenience
  wrapper is fine, a second scheduler or custom tarball engine is not.
- Distinguish **new Jobs submissions** from **appending to an existing queue**.
  `core/sweeps/nested_tuning.py` reads a fixed manifest. The separate
  `core/sweeps/submissions.py` durable SQLite queue assumes submitter and driver
  can access the same local disk; it is not a remote HTTP endpoint. Remote queue
  appends must execute on the server (e.g. a short submitted command), not write
  a laptop SQLite file or assume Ray Jobs appends to another job automatically.

Relevant code: `core/sweeps/{prepare,configuration,nested_tuning,submissions,queue,jobs}.py`,
`core/training/{runner,module,checkpoints}.py`, `configs/sweep/`.
Ray references:
- https://docs.ray.io/en/latest/cluster/running-applications/job-submission/index.html
- https://docs.ray.io/en/latest/cluster/running-applications/job-submission/sdk.html
- https://docs.ray.io/en/latest/ray-core/handling-dependencies.html

## Lossless rollout and acceptance checks

Before cutover, inventory active processes/jobs, source checkouts/environments,
plans and referenced files, checkpoint paths, Optuna/queue databases, dataset
caches and W&B links. Save checksums for critical immutable artifacts and take
consistent backups of mutable state. Keep the original server checkout,
environment, cache mount and path aliases until new submissions and resume work.
Do not require moving all results onto the laptop to move development there.

Start with a separate tiny smoke experiment/output root, using currently free
resources without interfering with the main campaign. Verify:

- Driver and worker source identities and Python/Ray/Torch versions agree.
- Training, checkpoint save/reload, validation-only tuning, final metrics,
  generated samples, and W&B logging all work remotely.
- Submitting a second job and disconnecting the laptop work without restarting
  Ray; logs/status remain retrievable after reconnecting.
- Configured server paths are writable; existing data is reused, not regenerated.
- Failure/retry does not duplicate completed work or change source/config silently.
- Old result/checkpoint references remain readable. Test resume against a copy
  of state, not an active campaign's output directory.
- Source exclusions omit secrets/large artifacts while including all required code.
- GPU advisory output distinguishes Ray reservations from external processes.

Only after these pass should new experiments use the new workflow. Existing
jobs can drain on their original cluster. A future migration/restart of those
jobs requires its own explicit plan; never submit duplicates during cutover.

## Testing and last verified state

From the repo, use `uv run --locked` with `--extra tuning --extra scoring
--extra tracking --extra mazes`, and `pytest -q`; lint with
`ruff check core main.py frontier.py`. On this server, use the shared environment
shown above until a replacement is verified. Do not run tests with system Python.

Latest merge validation covered 1,496 tests. The broad rerun had 1,487 passes and
nine historical snapshot failures in a process started before the isolation fix.
A fresh compiled-entrypoint + sampler run passed all 206 tests after that fix;
affected Ray/BlockGen/base/sampler checks passed 363 tests. Lint passed. Do not
misreport this as a single all-green full-suite run.

Tests now explicitly start local Ray clusters rather than attaching to live
training clusters; the BlockGen assertion checks the actual microbatch; sampler
snapshots control/restore denormal flushing because CPU compilation changes it.
All fixes were committed to `0cd4891`. No outstanding known test failures remain,
but rerun checks appropriate to migration changes.

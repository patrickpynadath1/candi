# Agent handoff: local development and remote Ray execution

## Performance and schedule artifact index (2026-11-01)

See [the performance and schedule handoff](docs/handoffs/performance-and-schedules.md) for campaign membership, final-score/checkpoint paths, per-sampler tuning records, schedule plots/CSVs, and intermediate-checkpoint caveats. Its adjacent CSV/JSON index records completed results and provenance at the stated snapshot time.

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

## Active rerun update (2026-10-30)

The original feedback-depth campaign below is now complete: 42/42 final results.
The user explicitly requested rerunning its adaptive cases with the **new
`core.time_sampling.cdcd.CDCDTimeSampler`**, not the old adaptive time-map.

New campaign: `feedback-depth-cdcd-20k-20261030`, 18 fresh cases (four pure
Gaussian variants plus two Gaussian+masked variants, across all three tasks).
Configs: `configs/path_schedule/cdcd.yaml`, `configs/sweep/feedback_cdcd.yaml`,
`configs/sweep/feedback_hybrid_cdcd.yaml`. All original training/data/inference
settings were compared and preserved. Only continuous view 0 uses the new learned
CDF; the discrete clock stays uniform, and no old AdaptiveTimeMap remains.

Plans and frozen source: `/home/patrick/.cache/duo/plans/feedback-depth-cdcd-20k-20261030/`.
Results: `/home/patrick/.cache/duo/nested-tuning/feedback-depth-cdcd-20k-20261030/`.
W&B: `rz-text-dlp/feedback-depth-cdcd-20k-20261030`.
The launch uses GPUs 0,1,2,5 with four concurrent cases and one inference trial
per case; 20k updates, BF16/compile, one seed, fixed 128 NFE. The source snapshot
is `source/` under the plan, with `source-provenance.json` and `source.patch`.
`launch.json`, `driver.log`, and `driver.pid` are in the result root. Initial
uv PID was 2317813; check live state rather than assuming this PID is still valid.
30 targeted CDCD/hybrid-clock/config tests and lint passed before launch. This
rerun does not itself establish that the earlier hybrid collapse is resolved;
inspect learned schedules and completed results before making that claim.

Deployment update (2026-10-30): the CDCD inverse-CDF needed a physical-time
clamp because float32 accumulation could produce values slightly above 1.
Six cases failed at the embedding schedule's domain check. The fix passed 16
targeted tests (including CPU/CUDA regression coverage). The original sweep
cluster was stopped and 12 unfinished cases restarted, preserving six completed
results. Active source is now `source-clamp-fix/` under the same plan root;
`clamp-fix-provenance.json` records its hashes, patch and checkpoint recovery.
Active manifest: `clamp-fix-pending.json`. Active launch/log/PID files in the
result root are `clamp-fix-launch.json`, `clamp-fix-driver.log`, and
`clamp-fix-driver.pid`; initial launcher PID 2515779. Ray temp root is
`/home/patrick/.cache/duo/ray/cd30fix`. GPUs remain 0,1,2,5.
For interrupted cases without `last.ckpt`, the existing full `best.ckpt` was
copied to `last.ckpt` to use the runner's existing resume support; the source,
step and checksum are recorded in the recovery provenance. This repeats work
since the saved checkpoint, not a resume from the final logged step.

Geometry extension queued on 2026-10-30: simplex, geodesic spherical, and vMF
spherical, each no-SC / expected-embedding SC, across the same three tasks:
18 additional cases. Hydra config `configs/sweep/feedback_geometry_cdcd.yaml`.
All use new CDCD, 20k steps, BF16/compile, one seed, native/native-greedy/EB
validation tuning at 128 NFE. Eight targeted training/generation tests passed.
Plans and frozen clamp-fixed source are under
`/home/patrick/.cache/duo/plans/feedback-geometry-cdcd-20k-20261030/`;
`launch.json` records source hashes and launch details. Results/W&B share the
active CDCD campaign above. Launcher PID 2595548 waits for PID 2515779 to exit
using GNU tail, then executes the existing nested runner on GPUs 0,1,2,5.
This is a sequential queued launch, not an append to the running fixed manifest.
Result-root log/PID files: `geometry-driver.log`, `geometry-driver.pid`.
The new Ray temp root will be `/home/patrick/.cache/duo/ray/geo30`.

Masked Sudoku architecture experiment launched 2026-10-30 on separate GPUs
3,4: `masked-attention-sudoku-20k-20261030`. Two matched no-SC models replace
the second block sublayer: standard MLP vs 2x-width attention (16 heads at
width 1024 internally; residual width 512). Both use eight blocks and the same
20k-step training and 128-NFE validation-only tuning protocol. Model parameter
counts are 28,559,628 and 28,539,148; this is not a FLOP-matched comparison.
Config: `configs/sweep/masked_attention_sudoku.yaml`; docs:
`docs/guides/attention-sublayer-ablation.md`. The component hook is
`DiT.sublayer_factory`, with a Hydra partial constructing independent sublayers.
20 targeted tests, lint, and three-step real-data BF16/compiled production-runner
smokes for both variants passed. The experiment's frozen source and provenance
live at `plans/masked-attention-sudoku-20k-20261030/` under the cache root;
results, launch record and driver log at `nested-tuning/` under the same name.
Initial launcher PID 2616778; Ray temp root `ray/ma30`; W&B project
`rz-text-dlp/masked-attention-sudoku-20k-20261030`. Existing sweeps were not stopped.

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

## Gated MLP centering experiment (2026-10-30)

The user stopped the masked-attention comparison; its driver/workers were
terminated and artifacts retained. Other campaigns were left running.
New experiment `masked-centered-mlp-sudoku-20k-20261030` compares fresh standard
MLP and `CenteredMLP` runs on GPUs 3,4. One unconstrained zero-initialized scalar
per block subtracts the sequence mean of up-projected features before GELU.
Full unpadded sequences only; masks and KV caching fail explicitly.
Config `configs/sweep/masked_centered_mlp_sudoku.yaml`, reusable choice
`configs/block_sublayer/centered_mlp.yaml`. Eight targeted tests and lint passed;
three real-data BF16/compiled updates passed with finite loss. Same eight-block,
512-wide, 20k-step, batch-256, one-seed masked/no-SC Sudoku protocol and 128-NFE
validation tuning as the stopped experiment. No completed result reused.
Plans/frozen source: `/home/patrick/.cache/duo/plans/masked-centered-mlp-sudoku-20k-20261030/`.
Results: `/home/patrick/.cache/duo/nested-tuning/masked-centered-mlp-sudoku-20k-20261030/`.
W&B project `rz-text-dlp/masked-centered-mlp-sudoku-20k-20261030`.
Initial launcher PID 2634189; Ray temp root `ray/mc30`; launch.json and
source-provenance.json record command/environment/source hashes.

## Forward-renoising inference sweep (2026-10-31)

User requested trying jump-to-x0 then forward-corrupt sampling on completed
feedback/depth models. New inference-only campaign:
`feedback-renoising-128nfe-20261031`, 48 checkpoints (24 masked/uniform from
original feedback campaign, 18 Gaussian/hybrid CDCD, six Sudoku geometry CDCD).
No retraining. Existing ComposedSampler + AllPositions + ForwardCorruption,
commitment=none; sample tokens (16 temperature trials in [0.5,1.5], initial 1)
and greedy tokens (two terminal-decoding choices). Original 128-NFE budget,
validation/test indices and generation seeds preserved. This compares point
clean-token estimates followed by fresh forward noise, not reverse bridges.
65 targeted sampler/resampling/budget tests passed. Sampler code matches frozen
source. Runner/source copied from the geometry campaign; unrelated newer
Sudoku Extreme/remote-shipping runner edits are not needed for this sweep.
Plans/configs/source/origins at `/home/patrick/.cache/duo/plans/feedback-renoising-128nfe-20261031/`.
Results at `/home/patrick/.cache/duo/nested-tuning/feedback-renoising-128nfe-20261031/`.
W&B project `rz-text-dlp/feedback-renoising-128nfe-20261031`.
GPU pool 3,4, two concurrent cases; initial launcher PID 3064546, Ray tmp `ray/re31`.
The 12 unfinished text8/TinyStories geometry cases are recorded in
`pending-unfinished.json` but NOT submitted; do not imply they are auto-queued.

## MDLM depth-mixing Sudoku comparison (2026-10-31)

User requested three agent implementations (DCA, MUDDFormer, content-based
per-position depth attention), shared architecture review, and fresh MDLM Sudoku
runs against a standard transformer. Implemented in `core/models/depth_mixing/`;
shared computation is `DDiTBlock.forward_streams`/`_forward`. DCA uses branch
contribution history plus final learned GRN; the other two use whole-state
history plus final R readout. See `docs/guides/depth-mixing-ablation.md` for the
paper ambiguities and explicit initialization choices. No commits requested.

Campaign `mdlm-depth-mixing-sudoku-20k-20261031`: four fresh runs, one seed,
20k steps, 8x512 backbone, MDLM hazard-weighted loss, no SC, BF16/compile,
batch256/warmup2500/EMA0.9999. MLP widths baseline2048/DCA2032/MUDD2032/depth1936
match counted training matrix FLOPs within0.3%. Native, native-greedy and EB-greedy
validation-only tuning at <=128NFE; fresh baseline required (older masked
posterior-CE runs are not the same objective). Caches/partitions are reused.

Plan, frozen source, manifest, FLOP/memory evidence:
`/home/patrick/.cache/duo/plans/mdlm-depth-mixing-sudoku-20k-20261031/`.
Results/checkpoints/driver.log/launch.json:
`/home/patrick/.cache/duo/nested-tuning/mdlm-depth-mixing-sudoku-20k-20261031/`.
W&B `rz-text-dlp/mdlm-depth-mixing-sudoku-20k-20261031`.
Launcher PID3302970; dedicated Ray local cluster GPU6,7, two concurrent cases;
Ray tmp `ray/dm31`. Other campaigns are untouched. Verify live state from logs.
All router/integration tests and affected backbone/sampler regressions passed;
all four real cached-data BF16/Inductor three-step training smokes passed, and
full microbatch64 GPU memory checks passed. The native discrete ancestral sampler
now accepts `greedy_clean_estimate`; default behavior is unchanged and tested.

## Geometry 16-layer follow-up (2026-11-01)

User authorized nine no-SC runs: simplex, spherical geodesic, spherical-vMF ×
Sudoku Hard, text8, TinyStories. Config: `configs/sweep/feedback_geometry_depth_cdcd.yaml`.
Plan: `/home/patrick/.cache/duo/plans/feedback-geometry-depth16-cdcd-20k-20261101/`.
Results: `/home/patrick/.cache/duo/nested-tuning/feedback-geometry-depth16-cdcd-20k-20261101/`.
W&B: `rz-text-dlp/feedback-geometry-depth16-cdcd-20k-20261101` (explicitly authorized).
Frozen source copies the previous geometry campaign, adding only the sweep YAML.
All nine resolved configs were compared to their eight-layer no-SC counterparts:
only depth (16 vs 8) and experiment metadata differ. Same 20k steps, BF16,
compile, CDCD, seed 1, and native/native-greedy/EB validation tuning at 128 NFE.
Launched using uv and Ray on GPUs 0,1,3,5; four concurrent cases, one trial per
case. Initial driver PID 4134446; Ray temp `ray/gd16`. Check `launch.json`,
`driver.pid`, `driver.log` in the results root rather than relying on this PID.
This follow-up is not included in the earlier dated performance index.

## Context-memory experiment retired (2026-11-01)

At the user's request, the experimental accumulator architectures, execution
machinery, configs, and dedicated tests were removed. No pilot training jobs
were launched. The provisional cache plan `plans/context-memory-pilot-20261101`
is retired and must not be submitted. Existing unrelated campaigns and earlier
architecture experiments are unchanged.

Historical measurements remain under `results/context-memory/`; their custom
architecture configs are archival and no longer runnable. The generic profiler
`scripts/benchmark_training_memory.py` retains normal/block diffusion support.
See `results/context-memory/preflight/normal-vs-block/` for matched normal versus
block training costs. Current work focuses on understanding block diffusion and
E2D2, not implementing another custom architecture.

## E2D2 architecture allocation experiment (in preparation)

User requested text8 + TinyStories to compare encoder versus decoder capacity,
then clarified that E2D2 should first be a separate composable wrapper.
`core/modifiers/encoder_decoder/EncoderDecoderModel` wraps the existing backbone;
block diffusion still supplies masks/layout and generation orchestration.
`configs/experiment/e2d2.yaml` composes the last-hidden-state design with local
DiT + MDLM. It is not an exact Qwen3 reproduction. See
`docs/guides/encoder-decoder.md` for cache scope and supported combinations.
Proposed matched allocations are encoder/decoder 10/2, 6/6, 2/10, one seed on
each task. Budget question (2k pilot versus 20k) is pending. No training sweep
has been launched. Do not confuse GPU smoke tests with campaign training.

Incremental cross-block caching is now implemented for both shared-transformer
BlockDiffusion and EncoderDecoderModel. A single cache persists per generation;
only new clean tokens are encoded, and E2D2 projects only new encoder outputs to
decoder K/V. Changed/truncated prefixes or incomplete bidirectional-block
extensions invalidate caches. Training compilation excludes mutable cache/layout
bookkeeping; full gradients remain enabled during training. Verified with
parallel/recompute parity, encoded-token call counts, and cache reset tests.
Latest validation: 57 affected regression tests passed, then 17 wrapper/cache
tests passed including two additional end-to-end sampler checks. Compiled BF16
GPU smokes passed for both architectures on both task configs; synthetic small
inputs only, saved at `results/e2d2/smoke.json`. No campaign launched.

## Real-data text8 pilots (2026-11-01, launched)

User requested actual text8 training/generation for both block diffusion and
E2D2. Plan/frozen source/resolved configs/driver log:
`/home/patrick/.cache/duo/plans/block-text8-pilot-20261101/`.
Current results: `/home/patrick/.cache/duo/ray-results/block-text8-pilot-20261101-attempt2/`.
Live worker CSVs/checkpoints/samples are initially under Ray artifacts in
`/home/patrick/.cache/duo/ray/btp01b/ray/session_*/artifacts/`; Tune persists them
to results on completion. Read launch.json/driver.pid for current driver identity.
W&B project: `rz-text-dlp/block-text8-pilot-20261101`; active BD3LM run `t52t4fnx`.
Two jobs, sequential on GPU7 through TrainingInferenceSearch. BD3LM has 8 shared
layers; E2D2 has 6 encoder + 2 decoder layers. Both exactly25,281,316 parameters,
width512, one seed1,2k optimizer steps, batch256/microbatch64, warmup200, BF16,
compile and checkpointing enabled. Existing text8 length256 cache reused,
351562 training windows. Blocksize32; ancestral15 updates+greedy terminal decode
=16 NFE/block and128 NFE/sequence. Validation16batches every400train batches;
generation16examples every100optimizer steps; final checkpoint evaluation128
examples automatically queued. Samples use the existing evaluation callback.
Attempt1 failed before training: Ray Tune log_to_file=True wraps stdout with a
Tee lacking isatty, which TorchDynamo requires. Attempt2 disables ONLY this
optional console Tee; driver/worker logs, CSV, W&B and checkpoint logging remain.
Failed attempt retained. No unrelated campaigns stopped or changed.
Real-data verification: BD3LM attempt2 reached step100 and successfully generated
16 text8 samples through the callback; training continued afterward. First
logged loss3.4852 at step9; later loss2.8843. Step100 generation correctly reports
128 total NFE. This is an execution check, not a final quality result. E2D2 remains
queued behind the BD3LM training/evaluation job on GPU7.
User subsequently requested stopping BD3LM and waiting for E2D2. BD3LM worker
250641 was sent SIGTERM and exited; Ray advanced to E2D2 worker262508 on GPU7.
E2D2 W&B run: `bgtgfzq2`. BD3LM saved artifacts copied to
`/home/patrick/.cache/duo/ray-results/block-text8-pilot-20261101-attempt2/stopped-bd3lm/`
(including step1300 samples and step1000 periodic checkpoint). The BD3LM Tune
trial is expected to show ERROR due user-requested termination, not a model bug;
its final-evaluation child is intentionally not queued. E2D2 and its automatic
final evaluation continue. Do not retry the stopped BD3LM trial.
User also stopped E2D2 after confirming functionality. Worker262508 received
SIGTERM and exited. Saved step500 checkpoint, best/last checkpoints, metrics,
and samples through500 are preserved under
`/home/patrick/.cache/duo/ray-results/block-text8-pilot-20261101-attempt2/stopped-e2d2/`.
Both pilot terminations are intentional; do not retry either. No final inference
children should be launched for these stopped training trials.

## E2D2 depth scaling prepared (2026-11-01, NOT launched)

User confirmed 20k optimizer steps. Source `configs/sweep/e2d2_depth_scaling.yaml`
uses native Hydra expansion: text8/TinyStories x encoder/decoder x1/2/4/8 layers,
other component fixed8. 16 runs, seed1, duplicate8/8 controls in both arms.
Shared-transformer block baseline would add two matched20k runs; stopped pilot
is not a quality baseline. See `docs/guides/e2d2-depth-scaling.md` for commands.
Manifest and resolved configs:
`/home/patrick/.cache/duo/plans/e2d2-depth-scaling-20k-20261101/`.
Planned results `/home/patrick/.cache/duo/nested-tuning/e2d2-depth-scaling-20k-20261101`.
W&B project `e2d2-depth-scaling-20k-20261101`; no runs submitted.
Latest GPU snapshot:5 and7 free;6 low memory but busy compute. Recheck at launch.
Batch256/micro8, BF16, compile, activation checkpointing, warmup2500, masked
ELBO-weighted CE. Native ancestral + native top1 tuning,128 total sequence NFE;
new shared sweep budget helper allocates across blocks conservatively. Recipe
identity includes scaling depth and encoder_decoder/block_diffusion settings.
Merged origin/candiv2.1 through8cfca5f (merge6ec829a), restoring all prior local
changes without conflicts. Pre-sync stash retained as backup. First65 affected
regressions passed; no changes pushed as part of preparing this sweep.

## E2D2 scaling submitted to existing cluster (2026-11-01)

User authorized submission. Driver277658 launched via uv and connected to the
existing active geometry-depth cluster at10.145.21.37:34819 (GPUs0,1,3,5),
without restart. 16 cases, case-concurrency4, inference-concurrency1, each
worker reserves1GPU. Existing geometry sweep continues; Ray shares slots.
Source snapshot/provenance and launch.json live in the prepared plan above;
results/driver.log/driver.pid/launch.json in the nested-tuning output above.
Use explicit RAY_AUTH_MODE=token and the existing local token path
/home/patrick/.ray/auth_token when attaching; never print/copy credentials.
The three clusters on6380 (0,1,2,5),6390/8275 (7),39873 (6) were stopped at
user request. The GPU6 PCA Sudoku training stopped at approximately12100/20000;
its existing best checkpoint and results remain. Do not automatically resume.
GPUs6/7 are outside the active cluster allocation, even when physically free.

## Raw-context controls queued (2026-11-01)

User requested8 decoder layers cross-attending to raw clean embeddings. Reused
EncoderDecoderModel with encoder_layers=0; no new execution branch. Named preset
configs/experiment/raw_context.yaml and inherited sweep configs/sweep/e2d2_raw_context.yaml.
Same joint decoder attention to clean K/V and noisy states, same block mask/cache;
not a separate cross-attention sublayer. Tests cover answer leakage, raw embedding
projection/gradient, incremental cache parity, and corruption-independent training.
Compiled BF16 GPU smoke with8 decoder layers/width64 passed on both task vocabularies.
Two20k-step controls queued on the existing10.145.21.37:34819 cluster (no restart).
Driver285925; plan/frozen source under ~/.cache/duo/plans/e2d2-raw-context-20k-20261101;
results/log/launch metadata under ~/.cache/duo/nested-tuning/e2d2-raw-context-20k-20261101.
Same W&B project e2d2-depth-scaling-20k-20261101; text8 and TinyStories, block32,
batch256, seed1, noSC, same validation-only native/native-top1 tuning. Existing
16 scaling jobs and geometry sweep remain untouched. These2 controls wait for
Ray GPU slots; GPUs6/7 are outside that cluster's allocation.

## Encoder attention K/V depth routing implemented (2026-11-01, not queued)

User chose actual encoder attention K/V instead of projected/raw intermediate
hidden states. Implemented as optional context_attention component on existing
EncoderDecoderModel, with EncoderDepthAttention in encoder_decoder/attention.py.
Preset configs/experiment/encoder_kv_depth.yaml; configurable encoder/decoder depth.
Per decoder query/head and context position: depth softmax over actual encoder
keys, weighted position score, then joint position softmax with noisy-state keys.
No extra context projections or rotary application; cache shared across decoder
layers. Corruption-independent through existing block input adapters.
PyTorch reference, query tiled, FP32 attention reductions; not a fused kernel.
Final encoder block attention-output/MLP branches don't affect exposed K/V (which
are derived from that block's inputs); document this when comparing depth/params.
55 focused tests passed; compiled BF16 synthetic training+cached generation smokes
passed for text8/TinyStories. Smoke results results/e2d2/encoder-kv-depth-smoke.json.
No new training jobs submitted for this architecture. Existing workers use frozen
source snapshots and remain unchanged. See docs/guides/encoder-decoder.md.

Terminal encoder cleanup: EncoderKeyValueReadout now replaces the final full
encoder block only for actual-K/V depth attention. It contains normalization and
K/V projection, no unused Q, attention-output, or MLP parameters/computation.
encoder_layers=L means L-1 full blocks plus one readout (L K/V levels). The same
training/checkpoint/cache loops execute it; regular E2D2 is unchanged. 39 targeted
tests pass, including full-block output/gradient parity at1 and2 levels and cache
checks. Compiled BF16 text8/TinyStories smokes also pass after this change. No jobs
for this variant have been launched; existing campaigns retain frozen code.

## Encoder-K/V depth sweep queued (2026-11-01)

User authorized levels2,4,8. Six cases: text8/TinyStories x encoder K/V levels2/4/8;
decoder8 fixed. Depth includes one terminal K/V readout (1/3/7 full encoder blocks).
Config configs/sweep/e2d2_encoder_kv_scaling.yaml inherits existing20k E2D2 settings:
block32, seed1, batch256, BF16, compile/checkpointing, noSC, masked ELBO, identical
validation-only native/native-top1 tuning. Same W&B project
rz-text-dlp/e2d2-depth-scaling-20k-20261101.
Plan, manifest, frozen source, provenance and launch metadata:
~/.cache/duo/plans/e2d2-encoder-kv-scaling-20k-20261101/.
Results/log/driver metadata:
~/.cache/duo/nested-tuning/e2d2-encoder-kv-scaling-20k-20261101/.
Driver294065 attached to existing10.145.21.37:34819; case-concurrency4, inner1.
No free Ray GPU slots at submission, so cases wait behind/alongside existing work.
No cluster restarted and no existing sweep changed.

## Contextual latent autoencoder diffusion implemented (2026-11-01, not queued)

User requested normalized encoder hidden states as diffusion states, latent MSE,
clean hidden -> token decoder CE, and decoded-denoiser CE as detached CDCD feedback.
Preset configs/experiment/latent_autoencoder.yaml; separate algorithm/model/objective/
sampler groups. Components in core/modifiers/latent_autoencoder/model.py reuse
EncoderDecoderModel for both denoiser and token decoder context attention, plus
existing GaussianProcess, LinearInterpolationSchedule, CDCD and FixedGridLoop.
No unused token embedding tables in latent prediction streams. Whole-block clean
encoding and context masks; arbitrary partially clean blocks/unaligned prompts
are rejected. Contiguous block-aligned text8/TinyStories prompts supported.
Default detach_encoder_for_denoising=true detaches noisy source, target and context;
encoder/decoder train via clean CE, denoiser via MSE. False allows MSE encoder grads.
Decoded denoising CE is no-grad CDCD signal only, never a training CE term.
Sampler uses linear physical-time ODE grid, then token decoding; counts denoiser
calls plus terminal token decode, excludes context encoding. Caches completed
context for encoder/denoiser/decoder; batch sampling currently serial per example.
No tuning manifest or jobs submitted. docs/guides/latent-autoencoder.md records
scope, configs, gradients, inference budget and limitations. 56 regressions passed
before strengthening visibility assertion; compiled BF16 synthetic training and
cached generation passed on text8/TinyStories vocabularies. Artifact:
results/latent-autoencoder/smoke.json (2 updates, width32, 1 layer per component,
length8, block2, batch1; execution check, not quality/anti-collapse evidence).

## GPU queue partition/restart (2026-11-01 evening)

User explicitly authorized discarding unfinished training progress and restarting
from scratch, then requested dedicated GPUs by campaign. Old 34819 cluster and
its four campaign drivers were stopped; unrelated GPU2/4/6/7 jobs untouched.
Inventory/manifests/SQLite backups and process records:
~/.cache/duo/plans/ray-gpu-reallocation-20261101/ (inventory.json,
restart-plans.json, clusters.json). Six completed geometry results remain in the
original output; only 27 unfinished cases were submitted anew. All original
results/checkpoints are preserved. Restart configs save recovery checkpoints every
1000 updates (latest periodic checkpoint + best); checkpoint.resume=null.
New plans/results retain campaign names with suffix -restart-gpu013. Despite that
shared suffix, GPU allocation is strictly partitioned:
- GPU0, Ray10.145.21.37:34920: geometry-depth16 (3 TinyStories runs).
- GPU1, Ray10.145.21.37:34921: E2D2 depth scaling16 + raw-context controls2.
- GPU3, Ray10.145.21.37:34923: encoder-K/V depth mixture6.
Each head exposes1GPU and12CPUs; drivers use case/inference concurrency1.
Separate worker port ranges prevent local cluster collisions. Use explicit Ray
addresses and token auth; NEVER global ray stop. Live launch.json/driver.pid
under each output records current drivers. Frozen source retained per campaign;
geometry snapshot only gained existing --working-dir CLI support to ship source.
GPU5 runs the new latent-autoencoder text8 pilot directly through main.py/shared
training lifecycle (not Ray). Plan/source/log/launch.json:
~/.cache/duo/plans/latent-autoencoder-text8-pilot-20261101/.
Outputs ~/.cache/duo/outputs/latent-autoencoder-text8-pilot-20261101/.
W&B rz-text-dlp/latent-autoencoder-text8-pilot-20261101, run xkwwenow.
Preset experiment=latent_autoencoder_text8_pilot,2k steps,batch256/micro8,
BF16,compile/checkpointing,encoder8/denoiser8/decoder8,width512 (89.4M params),
block32,text8 length256, generation16examples every100steps,16NFE/block,
checkpoint every500steps. Check live logs before claiming progress/quality.

### Superseding consolidation: one labeled Ray cluster

User requested native grouping for ALL jobs including text8. The isolated heads
34920/34921/34923 and direct text8 process are stopped. Current shared head is
10.145.21.37:34930, CPU-only, with four labeled one-GPU workers on this host:
queue=geometry GPU0; scaling GPU1; depth-mixture GPU3; latent-text8 GPU5.
A live Ray probe verified CUDA_VISIBLE_DEVICES=0/1/3/5 for these labels.
Startup must be sequential for same-host workers to avoid socket-name races.
All27 unfinished cases submitted via manifests with resources.label_selector;
six completed cases excluded. Outputs now have -restart-gpu013-labeled suffix.
Text8 uses core.training.submit with frozen resolved config, through Ray too:
plan/driver-labeled.log, launch-labeled.json, training-labeled.yaml;
output ~/.cache/duo/outputs/latent-autoencoder-text8-pilot-20261101-labeled.
The earlier brief attempts/logs remain preserved. No checkpoints existed for
those initial startup attempts. All final jobs start from scratch as authorized.
Queue inventory and authoritative current process/commands are in
~/.cache/duo/plans/ray-gpu-reallocation-20261101/shared-cluster.json and
shared-submissions.json. Never infer current PID from earlier sections.
Generic runner label routing added to core/sweeps/nested_tuning.py and frozen
snapshots (execution-only change); 15 nested runner tests passed including label
routing for training and evaluation. See docs/guides/ray-labeled-queues.md.

## Latent autoencoder 20k runs queued

User requested longer runs for text8 and TinyStories. Both submitted through
core.training.submit to shared Ray10.145.21.37:34930, queue=latent-text8 (GPU5),
behind the ongoing2k text8 pilot. One seed1 each, fresh20k training, same8/8/8
encoder/denoiser/decoder width512, block32, batch256/micro8, BF16,compile,
activation checkpointing. LR warmup2500; CDCD warmup1000. Recovery checkpoints
and generation16examples every1000steps; latest periodic + best checkpoints.
Task lengths256/1024, TinyStories prompt64;16NFE/block (14ODE updates+2readouts).
Config configs/experiment/latent_autoencoder_20k.yaml with task selected viaHydra.
Plan/frozen source/resolved task configs/submission PIDs/logs:
~/.cache/duo/plans/latent-autoencoder-20k-20261101/manifest.json.
Outputs ~/.cache/duo/outputs/latent-autoencoder-20k-20261101/{text8,tinystories}/.
W&B project rz-text-dlp/latent-autoencoder-20k-20261101. This adds standalone
training+periodic generation jobs, not sampler tuning studies. Other queues
unchanged. Both one-GPU requests are labeled so they cannot run concurrently
with each other or the pilot onGPU5; Ray determines order of pending jobs.

## November 2: latent jobs stopped, scaling moved to GPU 5

User cancelled latent-autoencoder training. Both 20k submissions are marked
stopped_by_user in their manifest; outputs/checkpoints remain intact. The latent
worker was removed. The scaling worker moved from GPU 1 to GPU 5 with the same
queue=scaling label. Both depth-scaling and raw-context drivers were restarted
against unchanged manifests/output directories. The early depth run had no
checkpoint yet (~300 steps); raw-context text8 retains its 20k checkpoint.
Geometry GPU 0 and depth-mixture GPU 3 remain untouched. GPU 1 was released.
Authoritative process records remain shared-cluster.json/shared-submissions.json
in ~/.cache/duo/plans/ray-gpu-reallocation-20261101/.

## November 2: simplex training-to-tuning recovery

The 16-layer TinyStories simplex run saved global_step=20000 but its outer
Optuna trial failed before trained.json/evaluation-config.yaml were recorded.
W&B emitted shutdown connection-reset errors; a separate subsequent Ray worker
had an MKL library-load failure and was retried. Do not conflate these errors.
Verified last.ckpt on CPU and recovered its handoff without training again.
Recovery plan: ~/.cache/duo/plans/simplex-handoff-recovery-20261102/ contains
manifest.json, frozen source, launch.json and driver.log. Driver475076 resumes
only the failed simplex case in its existing output/studies, queue=geometry.
It has entered inference search and waits behind the active spherical run.
Other active drivers were not restarted. Existing frozen runtime packages remain
unchanged; the recovery package contains the new recovery/finalization fixes.
Generic _train_worker now skips training for checkpoints at max_steps; incomplete
checkpoints resume normally. Tracking summary/finalization failures are logged
without invalidating successful training or masking training exceptions. Search
candidate failures now log immediately. 17 nested-tuning tests passed, including
completed-checkpoint recovery and incomplete-checkpoint resumption.

## TinyStories vocabulary audit (November 2)

Full pinned source parquet data is downloaded in the shared Hugging Face cache:
~/.cache/huggingface/hub/datasets--roneneldan--TinyStories/snapshots/
f54c09fd23315a6f9c86f9dc80f725de7d8f9c64/data/ (2,119,719 train; 21,990 validation).
CPU-only byte-BPE tokenizer audit uses all training stories, not the existing
100k/2k bounded caches. Artifacts: ~/.cache/duo/tokenizers/tinystories-bpe-20261102/.
Vocabularies 260/500/1000/5000/10000; smallest lossless byte alphabet needs 256
bytes + four specials, so 260 substitutes for user's approximate 100 target.
Largest BPE trained once; lower sizes keep prefixes of the same merge sequence.
All validation stories checked for lossless round-trip; no validation training.
Standalone tokenizers bpe-*.json; summary.csv/json; per-story lengths-*.npy;
manifest.json with source checksums, README.txt, run.log, frozen script.
Reusable tool scripts/analyze_bpe_lengths.py. Existing model training and task
configurations remain byte-level and unchanged; these new tokenizers are artifacts,
not yet wired into TinyStoriesTask (which currently requires vocab_size=260).

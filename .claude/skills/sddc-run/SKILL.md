---
name: sddc-run
description: Run, resume and monitor SDDC nested tuning (outer training study, inner per-sampler inference studies, one final test evaluation per case) on a local or existing Ray cluster. Use when launching a prepared manifest, checking how far a run got, resuming after a crash, or retrying failed candidates.
---

# Run and monitor nested tuning

Each case runs as an outer Optuna study over training candidates; every trained checkpoint
gets inner studies, one per inference policy, scored on validation only; then the overall
winner alone is scored on the test split. Details: `docs/guides/nested-tuning.md`.

Launching GPU work on a shared machine needs the user's go-ahead. Check the plan first with
`--check` (skill `sddc-sweep`).

## 1. Start Ray (if not already running)

`--address auto` attaches to a running cluster; start it with the same environment and
extras as the driver, and pin its GPUs:

```bash
CUDA_VISIBLE_DEVICES=0,1 uv run --locked --extra tuning --extra scoring \
  ray start --head --temp-dir $SDDC_STORAGE_ROOT/ray
```

For a small local run, `--address local` starts a private cluster for the driver instead.
On a shared server use the prepared head script and Ray Jobs (skill `sddc-remote`); never
attach to or stop someone else's cluster.

## 2. Run

```bash
uv run --locked --extra tuning --extra scoring python -m core.sweeps.nested_tuning \
  $SDDC_STORAGE_ROOT/plans/<name>/manifest.json \
  --output $SDDC_STORAGE_ROOT/runs/<name> \
  --address auto --case-concurrency 4 --concurrency 1
```

- `--case-concurrency`: cases driven at once; `--concurrency`: inference candidates per case.
  Each worker takes `resources_per_job` from the plan (0 or 1 GPU).
- Ray started under `uv run` ships the working directory to workers. Large untracked files in
  the checkout (archives, datasets) can exceed Ray's 512 MiB limit; keep them out of the
  repo, or for a local cluster set `RAY_ENABLE_UV_RUN_RUNTIME_ENV=0`.

## 3. Monitor

```bash
uv run --locked python -m core.analysis.status $SDDC_STORAGE_ROOT/runs/<name> [--json]
```

One row per case: `done` (final test result written), `running` (files changed in the last
`--active-minutes`, default 30), `failed` (a failed or skipped candidate, nothing running),
`partial`, or `not started`; plus trained candidates, selected policies and skipped trials.
It only reads files. Per-case details: `<case>/training/candidates/*/failure.json`,
`*/attempts.jsonl`, and each policy's `skipped.jsonl`.

## 4. Resume, retry

- **Resume** by rerunning the same command. Finished trials, candidates and final results are
  reused; a per-case `.driver.lock` stops two drivers from running one case.
- **Protocol changes** (config, evaluation spec, policies) make the case refuse to resume:
  "Protocol changed; use a new case name". Prepare under a new name instead of editing a
  plan in place. Resources and W&B settings may change freely.
- **Failed candidates** are skipped (`on_failure: skip`) and logged; `--retry-skipped`
  retries them once.
- A missing final file is not proof a run died: check `status` and the newest log first.

## Done when

`status` shows every case `done` (or the failures are understood), and the next step is
analysis (skill `sddc-analyze`) or archiving (skill `sddc-archive`).

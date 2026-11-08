---
name: sddc-remote
description: Drive a GPU server from a laptop with SDDC's Ray Jobs tooling (scripts/cluster/sddc_remote.py) - tunnel, submit jobs from a frozen source snapshot, follow logs, stop your own jobs, prepare plans on the server. Use whenever work must run on a remote machine or the laptop cannot install the CUDA lock.
---

# Run on a GPU server through Ray Jobs

Setup and background: `docs/guides/remote-execution.md`. Machine-specific values (host,
storage root) belong in environment variables or an untracked `CLAUDE.local.md`, never in
committed files.

## Safety first

- Read-only inspection (`gpus`, `list`, `status`, `logs`, `nvidia-smi`, `df`) is fine.
- Submitting jobs, starting or stopping a cluster, and deleting server files each need the
  user's go-ahead, every time.
- Stop only jobs you submitted, by id. Never run a bare `ray stop`, `pkill`, or touch other
  users' clusters.
- Everything persistent goes under the server's `SDDC_STORAGE_ROOT`, on a large volume, never
  `/tmp` or a nearly full root disk.

## 1. Configure the laptop

```bash
export SDDC_REMOTE_HOST=<ssh host>                 # as in ~/.ssh/config
export SDDC_REMOTE_STORAGE_ROOT=<absolute path on the server>
alias sddc-remote="uv run -q --no-project --python 3.10 --with 'ray[default]==2.59.0' python scripts/cluster/sddc_remote.py"
sddc-remote fetch-token                            # Ray auth token to ~/.config/sddc/ray_token
sddc-remote tunnel &                               # forwards the Jobs API (port 8275)
sddc-remote list && sddc-remote gpus
```

The server runs a Ray head started by `scripts/cluster/start_head.sh` (needs `GPUS=...`,
token auth, ports 6390/8275). Starting one is a cluster change: ask first.

## 2. Submit

```bash
sddc-remote submit --name <job> --follow -- \
  python -m core.sweeps.nested_tuning $SDDC_REMOTE_STORAGE_ROOT/plans/<name>/manifest.json \
  --output $SDDC_REMOTE_STORAGE_ROOT/runs/<name> --address auto --case-concurrency 4
```

`submit` uploads the checkout (at most 100 MiB; checkpoints, archives and caches are
excluded) and records its git revision and a hash of uncommitted changes.
`scripts/cluster/entrypoint.sh` copies it into a content-addressed snapshot under
`sources/`, builds one environment per `uv.lock` under `envs/`, links `core/data_cache` and
`core/outputs` to the storage root, and runs the command with `uv run --locked`. Later edits
on the laptop never change a running job.

Prepare plans **on the server**, as a submitted job, so manifests hold server paths:

```bash
sddc-remote submit --name prepare-<name> -- python -m core.sweeps.prepare -m +sweep=<name> \
  preparation.output=$SDDC_REMOTE_STORAGE_ROOT/plans/<name>
```

## 3. Follow and stop

`sddc-remote list`, `status JOB`, `info JOB`, `logs JOB [--follow]`, `stop JOB` (your own job
only).
Read run progress with `python -m core.analysis.status` submitted as a short job, or over
read-only `ssh` (skill `sddc-run`).

## Done when

The job shows in `sddc-remote list` with your name, its logs stream, and the run tree
appears under the server's storage root.

# Remote execution on a GPU server

Develop on a laptop; train and evaluate on a GPU server through the native
[Ray Jobs API](https://docs.ray.io/en/latest/cluster/running-applications/job-submission/index.html).
A persistent Ray head on the server receives submissions. Each submission uploads a
source snapshot, so later local edits never change queued or running jobs, and jobs
keep running when the laptop disconnects. Ray schedules the work; there is no second
scheduler. Single-host only: the storage paths assume all workers share one disk.

## Configuration

| Variable | Where | Meaning |
| --- | --- | --- |
| `CANDI_STORAGE_ROOT` | server | Persistent storage root (default `~/.cache/candi`). Put it on a large disk. |
| `CANDI_REMOTE_HOST` | laptop | ssh host or alias of the server. |
| `CANDI_REMOTE_STORAGE_ROOT` | laptop | The server's `CANDI_STORAGE_ROOT`, as an absolute path. |
| `CANDI_REMOTE_DASHBOARD_PORT` | laptop | Jobs API port (default 8275). |
| `CANDI_REMOTE_TOKEN` | laptop | Local copy of the cluster token (default `~/.config/candi/ray_token`). |

## Storage layout

| Path under the root | Contents |
| --- | --- |
| `data_cache/` | Datasets. Linked as `core/data_cache` in every snapshot; reused, never regenerated. |
| `outputs/` | Default run outputs. Linked as `core/outputs`. |
| `plans/<experiment>/` | Prepared manifests and case configs. |
| `nested-tuning/<experiment>/` | Nested-tuning results, checkpoints, Optuna databases. |
| `sources/<rev>-<content hash>/` | Persistent copy of each uploaded snapshot. Plans reference these paths. |
| `envs/<uv.lock hash>/` | One locked environment per lockfile. |
| `ray/rj/`, `ray/rj.token` | Temp dir and auth token of the persistent cluster. |

Configs name storage as `${oc.env:CANDI_STORAGE_ROOT,${oc.env:HOME}/.cache/candi}`.

## One-time server setup

Copy a checkout to the server (anywhere under the storage root) and start the head in
tmux. Choose GPUs that are actually free; the pool does not reserve GPUs against other
users of the machine.

```bash
git ls-files -co --exclude-standard | rsync -a --files-from=- . "$CANDI_REMOTE_HOST:candi-head/"
ssh "$CANDI_REMOTE_HOST"
tmux new -s candi-head
cd ~/candi-head && CANDI_STORAGE_ROOT=/big/disk/candi GPUS=0,1 scripts/cluster/start_head.sh
```

`start_head.sh` builds the locked environment, refuses ports already in use (defaults:
GCS 6390, dashboard/Jobs 8275), binds the dashboard to loopback, enables token
authentication and blocks in the foreground. Stop **only this** cluster with
`ray stop --address=127.0.0.1:6390` from that environment; on a shared machine never run
a bare `ray stop`.

## Laptop usage

The laptop needs only Ray's client, not the project environment:

```bash
export CANDI_REMOTE_HOST=my-gpu-server CANDI_REMOTE_STORAGE_ROOT=/big/disk/candi
alias candi-remote="uv run -q --no-project --python 3.10 --with 'ray[default]==2.59.0' \
  python scripts/cluster/candi_remote.py"
candi-remote fetch-token          # once: copies the token to ~/.config/candi/ray_token
candi-remote tunnel &             # ssh -N -L 8275:127.0.0.1:8275 $CANDI_REMOTE_HOST
candi-remote gpus                 # advisory: nvidia-smi alongside Ray reservations
```

Submit a command; it runs on the server from the snapshot with the locked environment
(`scripts/cluster/entrypoint.sh`):

```bash
candi-remote submit --name smoke --follow -- python main.py +experiment=smoke trainer.accelerator=cpu

R=$CANDI_REMOTE_STORAGE_ROOT
candi-remote submit --name prep-X -- python -m core.sweeps.prepare -m +sweep=X preparation.output=$R/plans/X
candi-remote submit --name prep-X-collect -- python -m core.sweeps.prepare collect $R/plans/X
candi-remote submit --name run-X -- python -m core.sweeps.nested_tuning $R/plans/X/manifest.json \
  --output $R/nested-tuning/X --address auto --case-concurrency 2 --concurrency 1

candi-remote list | status JOB | logs [--follow] JOB | info JOB | stop JOB
```

Prepare plans on the server, as above, so manifests never contain laptop paths. Each job
records its git revision, upload content hash, dirty flag and lock hash in its Ray
metadata and in `CANDI_SOURCE_*` environment variables. The driver reserves no CPUs or
GPUs; worker tasks hold the GPUs and use the same locked environment through
`runtime_env.py_executable`.

## Rules

- Use new experiment/output names for protocol changes; resume only compatible work.
- Do not touch Ray clusters, jobs or environments started outside this workflow.
- `nested_tuning` validates per-worker GPU requests as 0 or 1; fractional GPUs would not
  isolate memory.

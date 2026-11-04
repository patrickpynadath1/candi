# Remote execution on the GPU server

Development happens on a laptop; training and evaluation run on the GPU server
(`ruqi6000`) through the native [Ray Jobs API](https://docs.ray.io/en/latest/cluster/running-applications/job-submission/index.html).
A persistent Ray head on the server receives submissions. Each submission uploads
a source snapshot, so later local edits never change queued or running jobs, and
jobs keep running when the laptop disconnects. Ray schedules the work; there is
no second scheduler.

## Storage

Everything persistent lives under one server root, `CANDI_STORAGE_ROOT`
(default `~/.cache/duo`, which resolves to `/data/remote_cache/patrick/duo`).
The server's root filesystem (`/home`, `/tmp`) is nearly full: never write
checkpoints, datasets, logs, environments or Ray temp files there.

| Path under the root | Contents |
| --- | --- |
| `data_cache/` | Datasets. Linked as `core/data_cache` in every snapshot; reused, never regenerated. |
| `outputs/` | Default run outputs. Linked as `core/outputs`. |
| `plans/<experiment>/` | Prepared manifests and case configs. |
| `nested-tuning/<experiment>/` | Nested-tuning results, checkpoints, Optuna databases. |
| `sources/<rev>-<content hash>/` | Persistent copy of each uploaded snapshot. Plans reference these paths. |
| `envs/<uv.lock hash>/` | One locked environment per lockfile. |
| `ray/rj/`, `ray/rj.token` | Temp dir and auth token of the persistent cluster. |

Configs that name server storage use
`${oc.env:CANDI_STORAGE_ROOT,${oc.env:HOME}/.cache/duo}`, which resolves to the
same paths as before when the variable is unset.

## One-time server setup

Copy a checkout to the server (any location under the storage root) and start the
head inside tmux. Choose GPUs that are actually free; other users and other Ray
clusters share this machine, and the pool does not reserve GPUs against them.

```bash
git ls-files -co --exclude-standard | \
  rsync -a --files-from=- . ruqi6000:.cache/duo/head/
ssh ruqi6000
tmux new -s candi-head
cd ~/.cache/duo/head && GPUS=7 scripts/cluster/start_head.sh
```

`start_head.sh` builds the locked environment, refuses ports already in use
(defaults: GCS 6390, dashboard/Jobs 8275), binds the dashboard to loopback,
enables token authentication and blocks in the foreground. Stop **only this**
cluster with `ray stop --address=127.0.0.1:6390` from that environment; never run
a bare `ray stop` on the server.

## Laptop usage

The laptop needs only Ray's client, not the project environment:

```bash
alias candi-remote="uv run -q --no-project --python 3.10 --with 'ray[default]==2.59.0' \
  python scripts/cluster/candi_remote.py"
candi-remote fetch-token          # once: copies the token to ~/.config/candi/ray_token
candi-remote tunnel &             # ssh -N -L 8275:127.0.0.1:8275 ruqi6000
candi-remote gpus                 # advisory: nvidia-smi alongside Ray reservations
```

Submit a command; it runs on the server from the snapshot with the locked
environment (`scripts/cluster/entrypoint.sh`):

```bash
candi-remote submit --name smoke --follow -- python main.py +experiment=smoke trainer.accelerator=cpu

candi-remote submit --name prep-X -- python -m core.sweeps.prepare -m +sweep=X \
  preparation.output=/home/patrick/.cache/duo/plans/X
candi-remote submit --name prep-X-collect -- python -m core.sweeps.prepare collect \
  /home/patrick/.cache/duo/plans/X
candi-remote submit --name run-X -- python -m core.sweeps.nested_tuning \
  /home/patrick/.cache/duo/plans/X/manifest.json \
  --output /home/patrick/.cache/duo/nested-tuning/X --address auto \
  --case-concurrency 2 --concurrency 1

candi-remote list | status JOB | logs [--follow] JOB | info JOB | stop JOB
```

Prepare plans on the server, as above, so manifests never contain laptop paths.
Appending to a `core/sweeps/submissions.py` queue is likewise a submitted command
that runs on the server against a queue file under the storage root.

Each job records its git revision, upload content hash, dirty flag and lock hash
in its Ray metadata and in `CANDI_SOURCE_*` environment variables. The driver
reserves no CPUs or GPUs; worker tasks hold the GPUs. Workers use the same locked
environment through `runtime_env.py_executable`.

## Rules

- Use new experiment/output names for protocol changes; resume only compatible work.
- Do not touch Ray clusters, jobs or environments started outside this workflow.
- `nested_tuning` validates per-worker GPU requests as 0 or 1; fractional GPUs
  would not isolate memory.
- Single-host only: the storage paths assume workers share this machine's disk.

#!/usr/bin/env bash
#SBATCH --job-name=tinygsm-ray
#SBATCH --partition=ai
#SBATCH --account=ruqiz
#SBATCH --qos=normal
#SBATCH --requeue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=112
#SBATCH --time=7-00:00:00
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
# Keep the Slurm allocation even when setup, Ray, or the training driver fails.
# Only an explicit signal/cancellation or the wall-time limit ends this shell.
hold_allocation() {
  echo "Allocation retained for inspection/recovery until cancellation or wall-time limit."
  while true; do sleep 60 >/dev/null 2>&1 & wait "$!" || true; done
}
trap hold_allocation EXIT
trap 'trap - EXIT; ray stop --force >/dev/null 2>&1 || true; exit 0' TERM INT
repo_root=${CANDI_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}}
cd "$repo_root"
export CANDI_PYTHON=${CANDI_PYTHON:-$HOME/candi-1/.venv/bin/python}
export PATH="$(dirname "$CANDI_PYTHON"):$PATH"
export PYTHONPATH="$repo_root${PYTHONPATH:+:$PYTHONPATH}"
queue_root=${1:?Pass an absolute persistent queue directory}
export OMP_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export RAY_ENABLE_UV_RUN_RUNTIME_ENV=0
export RAY_gcs_server_rpc_server_thread_num=2
export RAY_gcs_server_rpc_client_thread_num=2
export RAY_num_server_call_thread=2
export RAY_TMPDIR="/tmp/candi-ray-${USER}-${SLURM_JOB_ID}"
mkdir -p "$RAY_TMPDIR" "$queue_root"
export RAY_AUTH_MODE=token
export RAY_AUTH_TOKEN_PATH="$RAY_TMPDIR/auth_token"
unset RAY_AUTH_TOKEN
"$CANDI_PYTHON" - <<'PY'
import os
from pathlib import Path
import secrets
path = Path(os.environ['RAY_AUTH_TOKEN_PATH'])
with open(path, 'w', opener=lambda p, flags: os.open(p, flags, 0o600)) as stream:
    stream.write(secrets.token_hex(32))
path.chmod(0o600)
PY
# Bind the jobs/dashboard endpoint to loopback; access it via SSH to this node.
ray start --head --node-ip-address=127.0.0.1 --port=6379 --num-cpus=112 --num-gpus=8 \
  --include-dashboard=true --dashboard-host=127.0.0.1 --dashboard-port=8265 \
  --temp-dir="$RAY_TMPDIR" --disable-usage-stats
printf '%s\n' "$(hostname)" > "$queue_root/node.txt"
printf '%s\n' "$SLURM_JOB_ID" > "$queue_root/allocation.txt"
ray job submit --address=http://127.0.0.1:8265 \
  --submission-id="tinygsm-queue-${SLURM_JOB_ID}-${SLURM_RESTART_COUNT:-0}" \
  --entrypoint-num-cpus=0 -- \
  "$CANDI_PYTHON" "$repo_root/scripts/tinygsm_ray_queue.py" --root "$queue_root" \
  || echo "Training driver exited with an error; Ray remains available for inspection/retry." >&2
# Keep the head available after the batch drains, for further Ray Jobs submissions.
hold_allocation

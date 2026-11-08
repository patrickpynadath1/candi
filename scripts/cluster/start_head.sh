#!/usr/bin/env bash
# Start the persistent Ray head that receives laptop submissions. Run on the server, in tmux:
#
#   GPUS=7 scripts/cluster/start_head.sh
#
# Other Ray clusters on this host are never touched. Stop only this one with
#   ray stop --address=127.0.0.1:$PORT   (from the same environment)
set -euo pipefail

root=${CANDI_STORAGE_ROOT:-$HOME/.cache/candi}
gpus=${GPUS:?Set GPUS to the comma-separated GPU pool, e.g. GPUS=7}
port=${PORT:-6390}
dashboard_port=${DASHBOARD_PORT:-8275}
cpus=${CPUS:-16}
# Short path: Unix socket names under the temp dir are limited to ~107 characters.
temp_dir=$root/ray/rj
uv=$(command -v uv || echo "$HOME/.local/bin/uv")
repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)

for p in "$port" "$dashboard_port"; do
  if ss -ltn | awk '{print $4}' | grep -qE ":$p\$"; then
    echo "Port $p is already in use; choose another with PORT/DASHBOARD_PORT." >&2
    exit 1
  fi
done

env_dir=$root/envs/$(sha256sum "$repo/uv.lock" | cut -c1-12)
mkdir -p "$temp_dir" "$root/envs"
(cd "$repo" && UV_PROJECT_ENVIRONMENT=$env_dir "$uv" sync --locked \
  --extra tuning --extra scoring --extra tracking --extra mazes)

export CANDI_STORAGE_ROOT=$root
export PATH="$(dirname "$uv"):$PATH"
export CUDA_VISIBLE_DEVICES=$gpus
export OMP_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export RAY_USAGE_STATS_ENABLED=0
export RAY_ENABLE_UV_RUN_RUNTIME_ENV=0
export RAY_DEFAULT_OBJECT_STORE_MAX_MEMORY_BYTES=2147483648
export RAY_AUTH_MODE=token
export RAY_AUTH_TOKEN_PATH=$root/ray/rj.token
unset RAY_AUTH_TOKEN
if [ ! -s "$RAY_AUTH_TOKEN_PATH" ]; then
  (umask 077 && head -c 32 /dev/urandom | od -An -tx1 | tr -d ' \n' > "$RAY_AUTH_TOKEN_PATH")
fi

num_gpus=$(awk -F, '{print NF}' <<< "$gpus")
# Loopback-only dashboard/Jobs API; the laptop reaches it through an SSH tunnel.
"$env_dir/bin/ray" start --head --node-ip-address=127.0.0.1 --port="$port" \
  --num-cpus="$cpus" --num-gpus="$num_gpus" \
  --include-dashboard=true --dashboard-host=127.0.0.1 --dashboard-port="$dashboard_port" \
  --temp-dir="$temp_dir" --disable-usage-stats --block

#!/usr/bin/env bash
# Server-side wrapper for every remote Ray Jobs submission (see scripts/cluster/sddc_remote.py).
#
# Ray unpacks each upload into a temporary directory that it may garbage-collect, but
# `core.sweeps.prepare` writes absolute source paths into plans. The command therefore runs
# from a persistent, content-addressed copy under $SDDC_STORAGE_ROOT/sources/. Datasets and
# outputs are symlinked to persistent storage so nothing is regenerated or written into the upload.
set -euo pipefail

root=${SDDC_STORAGE_ROOT:?SDDC_STORAGE_ROOT must name persistent storage}
source_id=${SDDC_SOURCE_ID:?SDDC_SOURCE_ID is set by sddc_remote submit}
uv=$(command -v uv || echo "$HOME/.local/bin/uv")
extras=(--extra tuning --extra scoring --extra tracking --extra mazes)

env_dir=$root/envs/$(sha256sum uv.lock | cut -c1-12)
source_dir=$root/sources/$source_id
mkdir -p "$root/envs" "$root/sources" "$root/data_cache" "$root/outputs"

link_storage() {
  ln -sfn "$root/data_cache" "$1/core/data_cache"
  ln -sfn "$root/outputs" "$1/core/outputs"
}
# Ray workers of this job import from the upload; give it the same storage links.
link_storage .

(
  flock 9
  if [ ! -d "$source_dir" ]; then
    staging=$(mktemp -d "$root/sources/.staging.XXXXXX")
    cp -a ./. "$staging/"
    mv "$staging" "$source_dir"
  fi
  # One environment per lockfile, so a lock change never alters an environment in use.
  cd "$source_dir"
  UV_PROJECT_ENVIRONMENT=$env_dir "$uv" sync --locked "${extras[@]}"
) 9>"$root/sources/.lock"

cd "$source_dir"
export UV_PROJECT_ENVIRONMENT=$env_dir
echo "sddc: source=$source_dir env=$env_dir host=$(hostname)" >&2
exec "$uv" run --locked --no-sync "${extras[@]}" "$@"

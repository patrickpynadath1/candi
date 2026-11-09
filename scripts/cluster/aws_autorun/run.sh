#!/usr/bin/env bash
# Boot-time runner of one self-healing spot machine; started by the launch template's user
# data as `run.sh BUCKET RUN [SLICE]`. Everything durable lives in s3://BUCKET:
#   autorun/RUN/{code.tar.gz,run.env,sets.txt}  inputs written by setup.sh
#   autorun/RUN/slices/SLICE/{run.env,cases.txt} a slice: this machine owns only these
#                                               PLAN/CASE names (several machines, one run)
#   data_cache/, hf/                            prepared datasets (scripts/build_data_cache.py)
#   plans/RUN/, runs/RUN/                       plans and nested-tuning run trees (synced both ways)
#   logs/RUN/<instance>.log                     this script's log
# A replacement machine restores plans and runs from S3 and resumes. When every case has a
# final result it uploads everything and sets its Auto Scaling group to 0 (which terminates it).
set -uo pipefail
bucket=$1 run=$2 slice=${3:-}
s3=s3://$bucket
export AWS_REGION=${AWS_REGION:-us-east-1}
export SDDC_STORAGE_ROOT=/data/sddc
export TINYSTORIES_DATA_DIR=$SDDC_STORAGE_ROOT/hf/TinyStories/data
export OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export RAY_USAGE_STATS_ENABLED=0
root=$SDDC_STORAGE_ROOT
log() { echo "[$(date -u +%FT%TZ)] $*"; }
imds() {
  local token
  token=$(curl -s -X PUT http://169.254.169.254/latest/api/token -H "X-aws-ec2-metadata-token-ttl-seconds: 300")
  curl -s -H "X-aws-ec2-metadata-token: $token" "http://169.254.169.254/latest/meta-data/$1"
}
instance=$(imds instance-id)
logfile=/var/log/sddc-autorun/run.log
upload_log() { aws s3 cp --only-show-errors $logfile $s3/logs/$run/$instance.log || true; }

# Setup steps must all succeed; on the first failure, upload the log and stop.
set -e
trap 'log "setup failed at line $LINENO"; upload_log' ERR

# 1. Storage: the instance-store NVMe (DLAMI mounts it at /opt/dlami/nvme) or the root disk.
sudo mkdir -p /data && sudo chown ubuntu:ubuntu /data
if mountpoint -q /opt/dlami/nvme; then
  sudo mkdir -p /opt/dlami/nvme/sddc && sudo chown ubuntu:ubuntu /opt/dlami/nvme/sddc
  ln -sfn /opt/dlami/nvme/sddc $root
else
  mkdir -p $root
fi
mkdir -p $root/plans/$run $root/runs/$run

sync_up() {  # Serialized; a spot notice waits for a running periodic sync.
  (
    flock 9
    aws s3 sync --only-show-errors $root/plans/$run $s3/plans/$run
    # --delete mirrors checkpoint cleanup (only best, last, newest step); the machine restored
    # everything it owns from S3 at boot, so its tree always contains what S3 had. A slice
    # syncs only its own case folders, so machines never delete each other's files.
    if [ -n "$slice" ]; then
      while read -r case; do
        [ -d $root/runs/$run/$case ] && aws s3 sync --only-show-errors --delete \
          $root/runs/$run/$case $s3/runs/$run/$case
      done < $root/cases.txt
    else
      aws s3 sync --only-show-errors --delete $root/runs/$run $s3/runs/$run
    fi
    upload_log
  ) 9>/tmp/sddc-sync.lock
}

# 2. Inputs and state from S3.
log "instance $instance ($(imds instance-type), $(imds placement/availability-zone)) starting run $run"
aws s3 cp --only-show-errors $s3/autorun/$run/run.env $root/run.env
aws s3 cp --only-show-errors $s3/autorun/$run/sets.txt $root/sets.txt
# shellcheck disable=SC1090
source $root/run.env   # ASG, SOURCE_REV, SOURCE_DIRTY
if [ -n "$slice" ]; then
  aws s3 cp --only-show-errors $s3/autorun/$run/slices/$slice/run.env $root/slice.env
  aws s3 cp --only-show-errors $s3/autorun/$run/slices/$slice/cases.txt $root/cases.txt
  # shellcheck disable=SC1090
  source $root/slice.env   # this slice's ASG
  log "slice $slice owns $(wc -l < $root/cases.txt) cases"
fi
export SDDC_SOURCE_REV=$SOURCE_REV SDDC_SOURCE_DIRTY=$SOURCE_DIRTY SDDC_SOURCE_ID=$run
rm -rf $root/sddc-head && mkdir -p $root/sddc-head
aws s3 cp --only-show-errors $s3/autorun/$run/code.tar.gz - | tar xz -C $root/sddc-head
aws s3 sync --only-show-errors $s3/data_cache $root/data_cache
aws s3 sync --only-show-errors $s3/hf $root/hf
aws s3 sync --only-show-errors $s3/plans/$run $root/plans/$run
if [ -n "$slice" ]; then
  while read -r case; do
    aws s3 sync --only-show-errors $s3/runs/$run/$case $root/runs/$run/$case
  done < $root/cases.txt
else
  aws s3 sync --only-show-errors $s3/runs/$run $root/runs/$run
fi
log "restored $(find $root/runs/$run -name result.json -path '*final*' | wc -l) finished cases from S3"
cd $root/sddc-head
ln -sfn $root/data_cache core/data_cache

# A checkpoint uploaded while it was being written is truncated; drop it so that candidate
# restarts instead of failing to load.
find $root/runs/$run -name '*.ckpt' -print0 | while IFS= read -r -d '' ckpt; do
  python3 -c "import sys, zipfile; zipfile.ZipFile(sys.argv[1]).namelist()" "$ckpt" 2>/dev/null \
    || { log "removing unreadable checkpoint $ckpt"; rm -f "$ckpt"; }
done

# S3 does not keep mtimes, but nested tuning fingerprints checkpoints by (size, mtime):
# put back the recorded mtimes so finished training is reused after a restore.
aws s3 cp --only-show-errors $s3/autorun/$run/restore_mtimes.py $root/restore_mtimes.py
log "$(python3 $root/restore_mtimes.py $root/runs/$run | tail -1)"

# 3. Environment.
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1
export PATH=$HOME/.local/bin:$PATH
uv sync --locked --extra tuning --extra scoring --extra tracking --extra mazes
python=$root/sddc-head/.venv/bin/python

# From here on failures are handled case by case.
set +e
trap - ERR

# 4. Background: periodic sync and the spot interruption notice (2 minutes' warning).
( while sleep 300; do sync_up; done ) &
( while sleep 5; do
    code=$(curl -s -o /dev/null -w '%{http_code}' -H "X-aws-ec2-metadata-token: $(curl -s -X PUT http://169.254.169.254/latest/api/token -H 'X-aws-ec2-metadata-token-ttl-seconds: 60')" http://169.254.169.254/latest/meta-data/spot/instance-action)
    if [ "$code" = 200 ]; then log "spot interruption notice: final sync"; sync_up; break; fi
  done ) &

# 5. Plans (prepared once, then reused from S3) and the run.
plans=()
while read -r name sweep overrides; do
  [ -z "${name:-}" ] || [ "${name:0:1}" = "#" ] && continue
  plan=$root/plans/$run/$name
  if [ ! -f $plan/manifest.json ]; then
    log "preparing $name"
    set -f  # Overrides such as seeds=[41] must not glob.
    # shellcheck disable=SC2086
    $python -m core.sweeps.prepare -m +sweep=$sweep preparation.output=$plan $overrides >> $root/prepare.log 2>&1 \
      && $python -m core.sweeps.prepare collect $plan >> $root/prepare.log 2>&1 \
      || { log "prepare $name failed; see prepare.log"; tail -30 $root/prepare.log; }
    set +f
  fi
  [ -f $plan/manifest.json ] && plans+=("$plan")
done < $root/sets.txt
sync_up

for attempt in 1 2 3; do
  log "run attempt $attempt over ${#plans[@]} plans"
  owned=(); [ -n "$slice" ] && owned=(--cases $root/cases.txt)
  if $python scripts/cluster/aws_autorun/run_all.py --runs $root/runs/$run "${owned[@]}" "${plans[@]}"; then
    log "all cases finished"; sync_up
    if aws autoscaling set-desired-capacity --auto-scaling-group-name "$ASG" --desired-capacity 0; then
      log "set $ASG to 0; this machine will be terminated"
    else
      log "could not set $ASG to 0 (role lacks autoscaling:SetDesiredCapacity); set it from the laptop"
    fi
    upload_log
    exit 0
  fi
  sync_up; sleep 60
done
# Do not leave an idle GPU machine running: everything is in S3 for inspection.
log "cases still unfinished after 3 attempts; stopping (state is in $s3/runs/$run)"
sync_up
aws autoscaling set-desired-capacity --auto-scaling-group-name "$ASG" --desired-capacity 0
upload_log

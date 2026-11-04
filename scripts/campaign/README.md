# Persistent Ray cluster inside Slurm

These scripts bootstrap Ray and submit the existing nested-tuning runner. They
do not allocate GPUs outside Slurm, regenerate data, or build an experiment
manifest. Use a reviewed, exported manifest from `core.sweeps.prepare`.

## 1. Shared environment and source

Choose paths visible at the same locations from all compute nodes. Use a frozen
source checkout: editing it during a campaign can change subsequently imported
workers. Prepare datasets/tokenizers and frozen split IDs before submission.

```bash
export CANDI_ROOT=/shared/candi-source
export UV_PROJECT_ENVIRONMENT=/shared/candi-env
bash "$CANDI_ROOT/scripts/campaign/setup_env.sh"
export CANDI_PYTHON="$UV_PROJECT_ENVIRONMENT/bin/python"
export CANDI_CLUSTER_DIR=/shared/candi-clusters
export CANDI_GPUS_PER_NODE=1
export CANDI_CPUS_PER_NODE=16
```

The setup script uses `uv sync --locked` with tuning, scoring and tracking extras.
Run it once for a shared environment. CUDA availability on a CPU login node is
not a GPU test. Workers need compatible drivers, this Python environment, and
read/write access to campaign storage. Configure W&B credentials separately on
workers if using online tracking; credentials are never embedded in manifests.

## 2. Request the cluster

```bash
sbatch --account=YOUR_ACCOUNT --partition=YOUR_GPU_PARTITION \
  --nodes=30 --gres=gpu:1 --cpus-per-task=16 --mem=64G \
  --time=4-00:00:00 --output=/shared/logs/candi-ray-%j.log \
  "$CANDI_ROOT/scripts/campaign/start_ray.slurm"
```

Create the log directory before submission. Resource/account flags are examples;
match site policy and set the two `CANDI_*_PER_NODE` values to the resources
actually requested. Homogeneous nodes are required by this bootstrap. With eight
GPUs per node, request `--gres=gpu:8` and set `CANDI_GPUS_PER_NODE=8`.

The batch starts one Ray process per node using allocation-scoped `srun` steps.
The first node hosts the head and also contributes GPUs. All nodes must reach
each other on Ray's ports; defaults are GCS 6379/dashboard 8265 and Ray's dynamic
worker ports. Optional script arguments `--port`, `--dashboard-port`, and
`--head-ip` handle port/network selection. Site firewall policy must allow the
cluster traffic; keep the dashboard on the trusted compute network.

A private auth token, per-node logs and `cluster.json` appear under
`$CANDI_CLUSTER_DIR/JOBID-RESTART_COUNT/`. Wait until its state is `ready`.
This verifies all requested nodes registered their GPU resources. The token
file is mode 0600 and the allocation directory is mode 0700. Never include the
token in Git or a results archive.

The allocation remains alive after submitted campaigns finish or fail. A Ray
node-process failure marks it degraded; startup readiness failures retain it
for inspection. The scripts never call global `ray stop`. Slurm cancellation,
wall-time expiration and cluster failures still end the allocation. Requeue
creates a new connection record and does **not** automatically resubmit campaigns.

## 3. Submit a smoke job

```bash
bash "$CANDI_ROOT/scripts/campaign/submit_campaign.sh" \
  --cluster "$CANDI_CLUSTER_DIR/JOBID-0/cluster.json" \
  --manifest /shared/plan/manifest.json \
  --output /shared/results/smoke-001 --id campaign-smoke-001 \
  --smoke --train-steps 1 --sampler-steps 1 --examples 1
```

Smoke uses the ordinary Ray training and evaluation workers, sequentially across
cases, and checks **every policy present in each case**. It trains a fresh model,
saves/loads its checkpoint, and generates with each policy. It uses one tuning
example/seed by default and never reads the final-test partition. Each policy
uses its first initial point, with deterministic fallback values for other
search dimensions. This tests execution, not search quality or convergence.

One sampler step means one solver transition **per block**, including any
corrector/bootstrap calls and terminal prediction in the NFE budget. AR generates
one new token and pads the remainder. The defaults preserve architecture,
precision, compilation, data and tokenizer, but reduce batch/accumulation to one
small batch and disable periodic callbacks/W&B. Add `--no-compile` for a faster
eager-only check; that does not validate compilation. Data preparation can still
be expensive if the required caches were not prepared beforehand.

Outputs include `smoke-report.json`, original case specs, smoke configs,
checkpoints, per-policy results and errors. A policy failure does not skip later
policies. Training failure is recorded and later cases are still checked. The
Ray Job fails if any case/policy fails. Use a **new output directory** for each
smoke attempt: smoke output never resumes or overwrites production results.

## 4. Submit the full campaign

```bash
bash "$CANDI_ROOT/scripts/campaign/submit_campaign.sh" \
  --cluster "$CANDI_CLUSTER_DIR/JOBID-0/cluster.json" \
  --manifest /shared/plan/manifest.json \
  --output /shared/results/production --id campaign-production-001 \
  --case-concurrency 30
```

Submission returns immediately with a Ray Jobs ID. The driver reserves no GPU;
training and inference tasks request resources from each manifest case. Case
concurrency counts complete training/tuning workflows. Choose it from available
GPUs rather than node count. Inference concurrency is one per case. No smoke
overrides are used in production.

Run submission on the login node if it can reach the dashboard. Otherwise run
the same command in an overlapping Slurm step on the head, for example
`srun --jobid JOBID --overlap --nodes=1 --ntasks=1 --cpus-per-task=1
--nodelist HEAD bash /shared/candi-source/scripts/campaign/submit_campaign.sh ...`.
The `CANDI_ROOT` and `CANDI_PYTHON` environment must be exported there too.

## Status and recovery

```bash
bash "$CANDI_ROOT/scripts/campaign/status.sh" \
  --cluster "$CANDI_CLUSTER_DIR/JOBID-0/cluster.json"
```

This reads Slurm status and Ray Jobs status. The cluster JSON is a bootstrap
record, so Slurm/Ray are authoritative after abrupt outages. To recover a failed
campaign, preserve its source/configs and output tree and submit the same manifest
and output with a new Ray Jobs ID. The nested runner reuses completed stages and
resumes saved checkpoints. A new protocol requires new case identities/output.

This implementation is CPU-tested for train/checkpoint/evaluation and command
construction. A real multi-node Slurm/GPU smoke remains required on the target
cluster before the full campaign is launched.

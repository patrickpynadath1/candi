# Hybrid CANDI in embedding space

From the repository root:

```bash
mkdir -p logs
sbatch job_candi_hybrid_embed_geometric_rank.sh
sbatch job_candi_hybrid_embed_linear_rank.sh
sbatch job_candi_hybrid_embed_adaptive_linear_rank.sh
sbatch job_candi_hybrid_embed_linear.sh
```

These use `algo=candi_embedding`, Gaussian interpolation in unit embeddings,
hybrid clean-token reveals, and rank bounds 0.01–0.49. The adaptive variant
uses linear rank coupling with the existing `rank-adaptive` S-FLM spline
clock (physical coordinate). It retains the default 1,000-step adaptive warmup.
The fourth variant, `linear`, uses plain interpolation `u = 1 - alpha`, with
no rank inversion and no adaptive clock. Rank bounds do not apply to it.
All four use the branch's unweighted hybrid cross-entropy objective. Rank variants use
finite-rank endpoints (`rank_endpoint_width=0`).

All four explicitly disable `legacy_architecture`: normalized embedding
lookup → hybrid corruption in embedding space → learned input projection
→ transformer. The input projection is 768 → 768 for these jobs, with a
learned bias; equal dimensions do not bypass the projection.

Defaults: TinyGSM/SmolLM-135M tokenizer (the existing `tinygsm_candi`
experiment), small model with 768-dimensional embeddings, sequence length
512, global batch 512, gradient accumulation 4,
250,000 optimizer steps, eight H100s on Gautschi's `ai` partition/account
`ruqiz`, and a 48-hour allocation. Checkpoints are saved every 5,000 optimizer
steps; validation runs every 5,000 batches. The GSM8K code-scoring callback evaluates 128 examples with 32
sampling steps every 5,000 training steps; full evaluation at the end is
disabled. Unconditional validation generation is disabled; the configured
sampler is `candi_embedding`. As in the existing TinyGSM CANDI experiment,
prompt tokens are excluded from the loss and padding tokens are trained.

The shared launcher uses `uv run --no-sync`. It prefers the repository's
`.venv`, then the existing `$HOME/candi-1/.venv`; set
`UV_PROJECT_ENVIRONMENT` to select another installed environment. It does
not synchronize dependencies on compute nodes. `UV_BIN` can select uv's
executable.

Data defaults to `$RCAC_SCRATCH/candi-1-data/tinygsm`, reusing the existing
512-token cache. Set `DATA_CACHE_DIR` to override. Checkpoints and Hydra
output are separated by schedule and Slurm job ID under
`${SCRATCH_DIR:-$RCAC_SCRATCH/cont_diff}/candi-v2/`. New submissions
start fresh. To resume, pass explicit checkpointing overrides.

Inspect a resolved configuration without allocating GPUs:

```bash
bash job_candi_hybrid_embed_adaptive_linear_rank.sh --cfg job --resolve
```

Extra arguments are Hydra overrides. For fewer GPUs, change both the Slurm
request and `NPROC_PER_NODE`, for example:

```bash
NPROC_PER_NODE=1 sbatch --gres=gpu:1 --cpus-per-task=14 \
  job_candi_hybrid_embed_linear_rank.sh batch_size=32
```

Submit from the repository root, or export `CANDI_ROOT` with its absolute
path. Create the `logs` directory before submission. W&B uses the existing
authentication and the `candi-tinygsm` project; set `WANDB_MODE=offline` for
offline logging.

DataLoader workers use the `spawn` multiprocessing context and persist for
both training and validation. This avoids forking a process with an active
CUDA runtime when a new validation iterator starts. The original failures
occurred in worker garbage collection: clearing an inherited exception
traceback destroyed a CUDA tensor and aborted the worker. Keep
`loader.multiprocessing_context=spawn` when using multiple workers.

The Slurm wrappers request `--qos=preemptible --requeue`. On a Slurm restart,
the launcher inspects the run's saved checkpoints and resumes the readable
checkpoint with the highest optimizer step, including optimizer state.
It falls back to older readable files if a checkpoint write was interrupted.
Progress since the latest completed checkpoint can be lost on preemption;
the scripts do not promise a checkpoint at the instant of preemption.
The currently queued resumed runs retain their explicit W&B run IDs.

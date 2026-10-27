# Gautschi RePlaid text training

Submit from the repository root:

```bash
sbatch scripts/jobs/job_tinygsm_replaid.sh
sbatch scripts/jobs/job_tinygsm_replaid_hybrid.sh
sbatch scripts/jobs/job_openwebtext_replaid.sh
sbatch scripts/jobs/job_openwebtext_replaid_hybrid.sh
```

These run `replaid` and the new full-ELBO `replaid_hybrid`, using
`scripts/run_replaid_text.sh` and the installed uv environment. Both use the
small 12-layer, 768-hidden backbone, 16-dimensional unit-norm embeddings,
learned noise schedules, bidirectional attention, and 250,000 optimizer steps.
The learning-rate warmup is 2,500 steps; there is no noise curriculum.

Each allocation requests eight GPUs, 112 CPUs, normal QoS, 48 hours, and
Slurm requeue on account ruqiz / partition ai. Global batch size is 512.
TinyGSM uses the existing 512-token SmolLM cache, accumulation 8, and preserves
its prompt-excluded/padding-included loss mask. OpenWebText uses the existing
1,024-token GPT-2 cache and accumulation 16. Validation runs every 5,000
optimizer steps (eight batches); checkpoints are saved every 5,000 steps.
TinyGSM also evaluates 128 GSM8K examples every 5,000 steps with 32 sampling
steps. OpenWebText has no GSM8K callback.

W&B projects are `candi-tinygsm` and `candi-openwebtext`, entity `rz-text-dlp`,
group `replaid-comparison`. Each job gets a separate run ID and checkpoint
directory. Requeued jobs resume their newest readable training checkpoint;
a requeue before the first checkpoint starts fresh. Slurm requeue handles
preemption; it does not automatically extend a job past its wall-time limit.

Override Hydra settings by appending arguments to sbatch. `DATA_CACHE_DIR`,
`SCRATCH_DIR`, `RUN_DIR`, and `UV_PROJECT_ENVIRONMENT` override storage or the
environment. Configuration can be inspected without a GPU using
`bash scripts/jobs/job_tinygsm_replaid.sh --cfg job --resolve`.

# Start here: portable Slurm campaign

## Prompt for the receiving agent

> Work from the `candiv2.1` branch. Read AGENTS.md and CAMPAIGN_HANDOFF.md.
> Prepare the portable Sudoku Hard, TinyStories 10k BPE, Text8 and vocabulary
> ladder campaign. Inspect the existing implementation before changing it.
> Complete the executable manifest and data preparation, validate every case and
> sampler, then run the one-step smoke on a small multi-node Slurm allocation.
> Report the source commit, manifest hash, case count, smoke results and remaining
> blockers before production launch. Do not substitute tokenizers, partitions,
> methods or training budgets silently. Ask for site-specific values that cannot
> be discovered. Preserve existing jobs and results.

## What exists, and what remains

- Implemented: locked environment setup, a persistent Ray cluster inside a Slurm
  allocation, campaign submission/status scripts, configurable train/checkpoint/
  sampler smoke execution. See [commands](scripts/campaign/README.md).
- Draft: [protocol and delivery requirements](docs/experiments/tinystories-bpe/portable-campaign/README.md)
  and [method inventory](docs/experiments/tinystories-bpe/portable-campaign/inventory.json).
  The inventory is NOT an executable manifest; do not pass it to submission.
- Still required: export the full executable case matrix; freeze all data splits;
  validate renoising compatibility; perform real multi-node GPU smoke; implement
  and validate results packaging. There is no finished one-command data or
  manifest builder for this expanded campaign yet.
- Existing CPU checks cover training, checkpoint reload, sampling and launcher
  construction. They do not establish that the target site's Slurm/network/GPU
  setup works.

## Scope

115 training cases per seed: 33 methods each for Sudoku Hard, TinyStories 10k
BPE and Text8, plus 16 additional TinyStories vocabulary-ladder cases. The 10k
ladder endpoints reuse the main-task cases. Exclude mazes and Sudoku-Extreme.
Use the inventory and protocol for method details. Deliver a results archive,
not an arXiv paper. Do not infer that the inventory already enumerates all 115
resolved cases.

## Setup and validation order

1. Record the checkout commit and any local changes. Use a frozen source snapshot
   and a shared locked uv environment for workers. Follow the campaign scripts
   README. Historical paths and Ray addresses elsewhere in AGENTS.md describe
   the original server, not this deployment.
2. Obtain the site's Slurm account, GPU partition, node/GPU type and count,
   CPUs/RAM per node, wall-time limits, shared storage paths and Ray network
   reachability. Thirty nodes does not necessarily mean thirty GPUs. Keep
   credentials outside Git and manifests.
3. Use the five committed tokenizer JSONs in
   `artifacts/tokenizers/tinystories-bpe-20261102/`; do not retrain.
   Run `sha256sum -c SHA256SUMS` from that directory and set each case's
   `bpe.tokenizer_path` to the matching absolute shared-checkout path. Fetch the pinned
   TinyStories parquet revision documented in the protocol. Transfer original
   cohort/split IDs when reproducing that protocol. If unavailable, report the
   blocker; a new versioned split is a protocol change, not a reproduction.
4. Resolve and record Sudoku Hard and Text8 dataset versions, preparation,
   train/tuning/test partitions, sequence lengths and primary metrics. Prepare
   caches once, then verify identical paths and readability on all workers.
5. Expand the task/method matrix using existing Hydra preparation and
   `core.sweeps.prepare collect`; use existing nested tuning, not a new scheduler.
   Record resolved configs, absolute paths, data/tokenizer/split hashes, unique
   case IDs, training budget and inference policies in the exported manifest.
   Verify exactly 115 unique training cases and no tuning/test overlap.
6. Audit model/objective/SC settings and global batch size. AdaLN must be enabled
   for block/context models. The portable draft also specifies time conditioning
   enabled, while some existing recipes deliberately disable time conditioning;
   surface this difference and resolve it explicitly before freezing configs.
   Clean-context encoders remain unconditioned. Do not equate AdaLN with time
   conditioning or inherit conflicting defaults silently.
7. Validate every intended sampler on its actual algorithm/task interface.
   Include ForwardResampling (renoising) where supported; report unsupported
   combinations rather than silently omitting them. Smoke only evaluates policies
   actually present in the manifest. Preserve sequence-level NFE accounting.
8. Start a small TWO-node allocation using the documented Slurm bootstrap.
   Verify both nodes register GPUs and each can access source, environment and
   artifacts. Submit smoke with `--smoke --train-steps 1 --sampler-steps 1
   --examples 1`. Keep compilation enabled for the final validation attempt.
   Require Ray Job SUCCEEDED and `smoke-report.json` with `passed: true`; inspect
   per-case/per-policy coverage, checkpoint reloads and finite metrics. A single
   training step is a wiring check, not a convergence or full-batch memory test.
9. Run a representative production-batch check and sampler checks at the intended
   production NFE. Record GPU memory and throughput before choosing concurrency.
   Report readiness evidence and obtain production launch authorization if it
   has not already been given by the colleague. Submit the full manifest using
   the same scripts, sizing concurrency from GPUs and per-case requirements.
10. Preserve source/configs and output trees during recovery. Do not stop the
    allocation when an individual campaign fails. Requeue creates a new cluster
    record and does not automatically resubmit the campaign. Never run global
    `ray stop` on a shared server.
11. Package the completed campaign according to the protocol's archive checklist.
    Include failed/incomplete cases in the index. Verify archive extraction and
    checksums; exclude secrets, raw datasets, environments and Ray sessions.

## Progress and communication

Keep a status record with: commit, manifest hash, Slurm job/cluster record, Ray
submission IDs, counts completed/running/failed/pending, latest smoke result,
artifact locations and blockers. Do not treat historical status as live evidence.

A shared GitHub issue can serve as an asynchronous request/status channel if both
people have repository access. The colleague must explicitly authorize their
agent to read that issue and post replies. Record which people may request
changes; quote the request being acknowledged and report what was done. Reading
an issue once does not create automatic monitoring. Polling or webhook delivery
requires a separately configured integration. Do not post credentials, private
node access details or checkpoint payloads. Do not share account credentials to
share an agent session.

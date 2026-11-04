# Agent handoff: CANDI v2.1

For a **new Slurm deployment / colleague handoff**, start with
[CAMPAIGN_HANDOFF.md](CAMPAIGN_HANDOFF.md) and follow its readiness gates.
Server paths, allocations and campaign status below are historical context for
the original deployment; do not copy them as defaults on another cluster.

## Purpose and working rules

This is a framework for controlled diffusion-language-model component ablations,
not one preferred algorithm. Reuse the information-path, denoising-task and
sampler interfaces. Use Hydra's existing sweep expansion and Ray/Optuna execution;
do not add campaign-specific schedulers.

- Use `uv` and the locked project environment.
- Do not restart or stop jobs, regenerate caches, or delete checkpoints without
  task-specific authorization. Preserve existing results and source snapshots.
- Never silently change a pinned experiment protocol or reuse results after
  changing data, tokenizer, objective, architecture or training budget.
- Never upload credentials, datasets, checkpoints, `.venv`, or runtime caches.
- Do not run global `ray stop` on the shared server. Other users have jobs here.
- Check live process ownership and commands; historical PIDs are not authority.

## Repository and environment

Active checkout: `/tmp/candi-distillation-recipe`, branch `dev-local-to-cluster`.
Development destination: `candiv2.1`, not `main`. Remote:
`https://github.com/patrickpynadath1/candi.git`. Other worktrees contain independent
work; do not reset or sweep their changes into this checkout.

Current server environment (used by active workers):
`/home/patrick/.codex/worktrees/54f2/duo/.venv`. Do not remove it.
Executable: `/home/patrick/.local/bin/uv`. Example:

```bash
UV_PROJECT_ENVIRONMENT=/home/patrick/.codex/worktrees/54f2/duo/.venv \
  /home/patrick/.local/bin/uv run --locked \
  --extra tuning --extra scoring --extra tracking pytest core/tests -q
```

A fresh machine should use `uv sync --locked` with the relevant extras.
Run tests without allocating active experiment GPUs unless explicitly authorized.

## Current campaigns and storage

Logical data root `/home/patrick/.cache/duo` resolves to
`/data/remote_cache/patrick/duo`. They are aliases, not duplicate datasets.
Plans/source snapshots: `plans/`; training/tuning/results: `nested-tuning/`;
standalone outputs: `outputs/`; datasets: `data_cache/`; Ray state: `ray/`.
Preserve the entire case tree: protocol, configs, checkpoints, Optuna SQLite,
selected parameters, final metrics, samples and tracking metadata.

The shared Ray address is `10.145.21.37:34930`. Latest recorded allocation:

| Label | GPU | Work |
| --- | --- | --- |
| geometry | 0 | Remaining 16-layer TinyStories continuous runs |
| depth-mixture | 3 | Encoder-K/V mixture sweep |
| scaling | 5 | E2D2 depth scaling and raw context |

GPU 1 was released; latent-autoencoder jobs were stopped at user request.
Verify live availability rather than assuming it is still free. Authoritative
launch records: `plans/ray-gpu-reallocation-20261101/shared-cluster.json` and
`shared-submissions.json`. Additional simplex recovery driver:
`plans/simplex-handoff-recovery-20261102/launch.json`.

The simplex recovery reuses a verified 20k checkpoint and its existing Optuna
studies. Its patched source snapshot is separate from ongoing workers' snapshots.
Do not assume code edits here change an already-running Ray worker.

## TinyStories BPE: handed off to another agent

See [the campaign README](docs/experiments/tinystories-bpe/README.md). It lists
49 proposed training cases, data controls, objectives and setup. The user handed
this campaign to another agent on a different server; do not launch it here. It is not launched
and its final manifest is not prepared. Shared cohorts and new validation/test
partitions must be frozen before launch; historical byte-cache indices cannot be
reused. No hidden-state SC or latent-autoencoder runs are included.

Tokenizer artifacts/audit: `~/.cache/duo/tokenizers/tinystories-bpe-20261102/`.
Source: all TinyStories parquet shards at revision
`f54c09fd23315a6f9c86f9dc80f725de7d8f9c64` in the shared Hugging Face cache.
Vocab sizes 260/500/1000/5000/10000; 260 is the minimum lossless byte alphabet plus
four specials. Full-corpus statistics, hashes and tokenizers are saved.
`task=battery/tinystories_bpe` now supports these artifacts and local parquet
inputs. Existing model jobs retain their original byte configs.

## Further context

- [Performance and schedule artifact index](docs/handoffs/performance-and-schedules.md)
- [Historical server/workstream notes](docs/handoffs/server-history-20261102.md)
- [Ray queue setup](docs/guides/ray-labeled-queues.md)
- [Nested training/inference tuning](docs/guides/nested-tuning.md)

The historical notes preserve the local-to-cluster migration goals, older
campaigns, stopped pilots, and diagnostic plots. Their process IDs and status
reports are snapshots. Always check the launch records and current process state.

## November 2: encoder-K/V recovery

The text8 encoder-K/V 2-layer / decoder 8-layer checkpoint reached 20k.
Recovered its missing training handoff and verified one-sample CPU generation
with the original native sampler and 128 sequence NFE. Recovery launch record:
`plans/encoder-kv-handoff-recovery-20261102/launch.json`. It reuses the existing
case, checkpoint, partitions and Optuna study on the depth-mixture queue.

Raw-context and E2D2 decoder-1 recovery drivers are already tuning successfully;
do not submit duplicates. Historical outer FAIL rows remain while retry work
runs: the outer objective includes both training and inference search. Check
inner trials, driver locks and live workers before declaring a case failed.

## November 2: AdaLN-corrected block/E2D2 restart (current)

At the user's request, stopped the old depth-scaling, raw-context, encoder-K/V
and K/V recovery drivers. Their original outputs remain preserved. Replaced them
with 24 fresh cases (16 depth scaling, 2 raw context, 6 encoder-K/V), using
`model.adaLN=true` inherited from the corrected MDLM recipe. MDLM's
`algorithm.time_conditioning=false` remains intentional: zero time input does
not remove the learned AdaLN modulation and gates. Clean encoders remain
unconditioned. Other training and evaluation settings are unchanged.

Authoritative launch index:
`plans/adaln-block-restart-20261102/submissions.json`.
New plan/output names: `e2d2-{depth-scaling,raw-context,encoder-kv-scaling}-adaln-20k-20261102`.
W&B project: `rz-text-dlp/e2d2-adaln-20k-20261102`.
Same Ray queues: scaling on GPU 5, depth-mixture on GPU 3. Geometry on GPU 0
and simplex recovery remain untouched. BPE is owned by another agent/server.
70 focused tests passed, including AdaLN-active cache parity and context gradients.

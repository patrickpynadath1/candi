# TinyStories BPE: corruption, feedback, depth, and context

Status: **planned campaign; not launched**. The full source dataset and five
trained tokenizer artifacts exist on the current server. The BPE task is now
configurable, but the 49-case campaign manifest has not been generated. Cohort
selection and validation/test partitions must be frozen first. Do not relaunch
historical manifests with only a vocabulary override: their data indices and
checkpoint identities refer to the old byte task.

## Questions

1. Does expected-embedding self-conditioning help each corruption form? How does
   an eight-layer model with SC compare with a sixteen-layer model without it?
2. Does a masked discrete view provide the same benefit as self-conditioning,
   or do the two effects complement each other?
3. In block generation, is additional encoder or decoder depth more useful?
   How much context can raw embeddings or encoder-depth K/V mixtures provide?
4. How does token granularity change Gaussian, masked, and uniform diffusion,
   relative to autoregressive generation?

These are component ablations, not a leaderboard sweep. Keep task, data, training
budget and evaluation fixed within each comparison. “SC” below always means the
expected embedding from the preceding prediction (`source: posterior_mean`).
Hidden-state SC, insertion, distillation, latent-autoencoder, GIDD and blurring
runs are outside this campaign.

## Exact run matrix: 49 training jobs per seed

Use one seed initially. All main comparisons use vocabulary 10,000. Base depth
is eight layers; double depth is sixteen. Gaussian means **learned embedding**
corruption, uniform means **posterior cross entropy**, and continuous paths use
the corrected CDCD adaptive time sampler. Do not also repeat the obsolete CDCD
implementation as another baseline.

| Family at vocabulary 10,000 | Configurations | Jobs |
| --- | --- | ---: |
| Gaussian / masked / uniform | Each: eight layers without SC, eight layers with expected-embedding SC, sixteen layers without SC | 9 |
| Gaussian + masked hybrid | Eight layers, SC off/on | 2 |
| Simplex / spherical geodesic / spherical vMF | Each: eight layers without SC, eight layers with expected-embedding SC, sixteen layers without SC | 9 |
| E2D2 encoder scaling | Encoder 1/2/4/8, decoder 8 | 4 |
| E2D2 decoder scaling | Encoder 8, decoder 1/2/4 (8/8 already counted) | 3 |
| Encoder-depth K/V mixture | Encoder levels 2/4/8, decoder 8 | 3 |
| Raw embedding context | Encoder 0, decoder 8 | 1 |
| Shared-transformer block diffusion | Eight layers | 1 |
| **Main suite subtotal** | | **32** |
| Tokenizer ladder: Gaussian / masked / uniform | Expected-embedding SC only, at 260/500/1,000/5,000; the three 10k endpoints above are reused | 12 |
| AR baseline | Eight layers at 260/500/1,000/5,000/10,000 | 5 |
| **Total** | | **49** |

The K/V encoder depth counts L−1 full transformer blocks plus a terminal K/V
readout. It deliberately excludes the unused final attention/MLP update; document
actual trainable parameter counts rather than implying equal parameter budgets.
Expected-embedding SC shares backbone weights but adds compute; report that too.

49 counts **training jobs**, not Ray tasks. Sampler tuning and final evaluation
create additional GPU tasks. More training seeds multiply this count. If the
ladder also needs no-SC controls, add 12 jobs (61 total). If it uses a different
corpus than the main suite, its three 10k endpoints cannot be reused (52 total).

## Data and tokenizer protocol

Pinned source: `roneneldan/TinyStories`, revision
`f54c09fd23315a6f9c86f9dc80f725de7d8f9c64`; 2,119,719 training and 21,990
validation stories. The old experiments considered only the first 100,000/2,000
source examples. Train all new models, including the 260 baseline, on the same
newly selected source population; old byte checkpoints are not matched controls.

Byte-level BPE uses a complete 256-byte alphabet and four specials
PAD/BOS/EOS/MASK at IDs 0/1/2/3. Thus the requested approximate 100 vocabulary is
replaced by the minimum lossless size, **260**. Train the largest vocabulary on
the full training split only; smaller variants retain prefixes of the learned
merge sequence. Preserve case and Unicode. Validation never trains the tokenizer.

Full-corpus audit (content lengths, excluding BOS/EOS):

| Vocabulary | Train mean | Validation mean | Train fitting 512 incl. BOS/EOS | Train fitting 1024 incl. BOS/EOS |
| ---: | ---: | ---: | ---: | ---: |
| 260 | 897.7 | 874.0 | 3.74% | 79.70% |
| 500 | 393.1 | 381.6 | 84.84% | 98.49% |
| 1,000 | 296.1 | 287.2 | 92.00% | 99.67% |
| 5,000 | 224.2 | 217.2 | 96.66% | 99.93% |
| 10,000 | 218.4 | 211.6 | 96.96% | 99.95% |

Use **512 tokens for the 10k suite**, not the old 1024-byte canvas. Proposed ladder
capacities are 1024/768/512/512/512 for 260/500/1000/5000/10000. Before preparing
cases, select one common cohort fitting *every* tokenizer's capacity and freeze
its source-hash IDs. This intentionally selects shorter stories. Use the same
cohort for the main suite if its 10k endpoints are to be reused by the ladder.

The task accepts `data.allowed_ids_path`, a JSON list of stable IDs such as
`train:<sha256-of-stripped-UTF8-story>` and `validation:<sha256>`. The allowlist
filters source stories; it does not truncate them or override capacity checks.
Verify equal retained ID sets after preparing every tokenizer. A shared allowlist
alone is insufficient if different token capacities still reject different IDs.

Keep the prompt at the same **64 UTF-8 bytes** across variants, backing up to a
character boundary. Encode prompt and continuation separately so no BPE merge
crosses the conditioning boundary. This can add a few tokens compared with the
whole-story audit above; use task-prepared lengths for final cohort membership.
The task cache fingerprints tokenizer contents, source files and any cohort IDs.
Existing byte caches retain their original identities.

Freeze disjoint tuning and final-test subsets by stable IDs from the validation
source split, then map them to each task's dataset indices. Never reuse old cache
row indices. The remaining source validation examples can be held in reserve.

## Reproduce tokenizers and lengths

Use the same locked uv environment on the submission machine and workers:

```bash
uv sync --locked --extra tuning --extra scoring --extra tracking
export DUO_CACHE="${HOME}/.cache/duo"
export TINYSTORIES_REVISION=f54c09fd23315a6f9c86f9dc80f725de7d8f9c64
uv run --locked hf download roneneldan/TinyStories --type dataset \
  --revision "$TINYSTORIES_REVISION" --include 'data/*.parquet' --include README.md
# Set this to the returned snapshot directory followed by /data:
export TINYSTORIES_DATA_DIR="${HOME}/.cache/huggingface/hub/datasets--roneneldan--TinyStories/snapshots/${TINYSTORIES_REVISION}/data"
RAYON_NUM_THREADS=8 uv run --locked python scripts/analyze_bpe_lengths.py \
  --data "$TINYSTORIES_DATA_DIR" \
  --output "$DUO_CACHE/tokenizers/tinystories-bpe-20261102"
```

Reuse existing audit artifacts if present. The script refuses to overwrite saved
tokenizers; choose a fresh output directory when changing the protocol.

The audit writes standalone `bpe-*.json`, checksums/provenance in `manifest.json`,
`summary.csv/json`, and full per-story length arrays. It is CPU-only and does not
start model training. Existing server artifacts are under that output directory.
The original source parquet files remain in the HF cache; do not duplicate them.

The configured task consumes those artifacts:

```bash
# Inspect a resolved config without training or accessing the dataset:
uv run --locked python main.py +experiment=mdlm task=battery/tinystories_bpe \
  bpe.vocabulary_size=10000 task.length=512 --cfg job --resolve
# Once data/cohort/partitions are frozen, use the same task override with any
# compatible recipe. A custom artifact is selected with bpe.tokenizer_path=...
```

The BPE preset reads all parquet shards from `TINYSTORIES_DATA_DIR`; it defaults
to the full source split sizes. Alternatively supply separate local JSONL
`data.train_path` and `data.validation_path` and set `data.parquet_dir=null`.
`data.download=true` explicitly permits the existing streaming Hub source.
Default `download=false` requires local source files or an existing matching cache.

## Training and sampler settings

Proposed shared budget: 20,000 optimizer updates; global batch 256; width 512;
eight heads; LR 3e-4; 2,500-update warmup; BF16; compilation. Use gradient
accumulation to fit the device. Checkpoint at least every 1,000 updates and retain
the latest complete checkpoint. Do not allocate GPUs by hard-coded machine IDs
inside recipe definitions; use the existing Ray cluster's resources/queue labels.

Block size is 32 tokens. Keep model-specific losses explicit. Historically,
masked full-sequence runs used unweighted CE while block runs used ELBO-weighted
CE. **Align these for the new architecture comparison before launch** (proposed:
masked ELBO weighting throughout); otherwise architecture and objective effects
are confounded. Keep uniform posterior CE and the continuous recipe objectives.

Tune inference on validation, never on final-test scores. Reuse the existing
nested Optuna workflow; a fixed training recipe is one outer candidate. Record
per-policy selections and report the validation-selected overall winner separately.
Use native stochastic and native greedy policies; include EB/renoising only where
supported and with the same search budget for comparisons using those policies.

Fix the diffusion budget at 128 NFE per sequence. The block budget utility divides
that across blocks, including a terminal call. For 512 tokens/block32 this is
8 NFE per block. Prompt blocks may reduce actual usage. SC can add model forwards;
record actual calls and latency, not only solver-step labels. AR uses its cached
sequential decoder: set `budget_parameter: null` in its tuning policy, omit sampler
steps, and use sequence capacity as its NFE upper bound. AR is a quality/compute
reference, not a claim of matching diffusion's 128-call generation budget.

Measure corpus BLEU on the same decoded continuations, plus word count, byte or
character count, repetition, termination and diversity. Token counts alone are
not comparable across vocabularies. The default task primary metric remains a
termination diagnostic; **explicitly select `corpus_bleu` for campaign tuning**.

## Preparing and running the campaign

Reuse Hydra multiruns through `core.sweeps.prepare`; do not write 49 independent
configs or a special campaign expansion engine. Existing families are:

- `sweep=feedback` / `feedback_cdcd` / `feedback_hybrid_cdcd` for pure/hybrid SC.
- `feedback_geometry_cdcd` / `feedback_geometry_depth_cdcd` for geometry.
- `e2d2_depth_scaling`, `e2d2_encoder_kv_scaling`, `e2d2_raw_context` for context.
- `experiment=bd3lm` and `experiment=autoregressive` for the controls.

Override the task, budget and output locations in a new preparation directory.
Do not carry `preparation.reference_manifest` from the historical presets into
this campaign. Supply the newly frozen evaluation partitions directly. Deduplicate
8/8 E2D2 and the three 10k SC ladder endpoints before exporting the manifest.
AR preparation uses `preparation.sampler_budget_parameter=null` as well as
`budget_parameter: null` in its policies.

After preparing the cases, the ordinary runner is sufficient:

```bash
uv run --locked --extra tuning --extra scoring --extra tracking \
  python -m core.sweeps.prepare collect "$PLAN_DIR"
uv run --locked --extra tuning --extra scoring --extra tracking \
  python -m core.sweeps.nested_tuning "$PLAN_DIR/manifest.json" \
  --output "$RESULT_DIR" --address "$RAY_ADDRESS" \
  --working-dir "$SOURCE_SNAPSHOT" --case-concurrency 8 --concurrency 1
```

These are launch instructions, not a record that jobs have been submitted.
Freeze the exact source snapshot and manifest before launch. Data/tokenizer paths
must be accessible on workers. See [Ray queues](../../guides/ray-labeled-queues.md)
and [nested tuning](../../guides/nested-tuning.md). Reports/checkpoints belong in
`$DUO_CACHE`, not Git. Each case keeps its resolved config, tokenizer fingerprint,
checkpoint, tuning studies, selections, final metrics and generated samples.

The previous estimate of 1–3 days on eight H100s is **not a benchmark**. Before
reserving a campaign-sized budget, measure representative 10k-vocabulary standard,
SC, 16-layer and block runs. High-dimensional continuous states and vocabulary
projections may scale differently from attention; 49 jobs do not have equal cost.

# TinyStories

`TinyStoriesTask` trains prompt-conditioned generation from complete short stories.
The presets are `+experiment=tinystories` (masked fixed canvas) and
`+experiment=tinystories_insertion` (Gaussian/masked hybrid plus insertion).
These presets demonstrate supported assemblies; changing between them also changes
corruption, so they are not by themselves a controlled insertion ablation.

## Data and reproducibility

```bash
uv run python main.py +experiment=tinystories data.download=true
uv run python main.py +experiment=tinystories_insertion data.download=true
```

Downloading is opt-in. The source is `roneneldan/TinyStories`, pinned to revision
`f54c09fd23315a6f9c86f9dc80f725de7d8f9c64`. Preparation streams at most
100,000 training and 2,000 official validation source examples by default. It
retains only complete stories fitting the byte budget, removes duplicates within
each split, and refuses duplicate stories across train and validation. It never
labels an arbitrarily truncated story as a complete ending. Filtering means the
retained population is **short TinyStories**, not the entire dataset. The cache
manifest and evaluation `data_manifest` include source/retained counts and
retention rate; limits count source examples, not accepted stories.

Local offline input uses one `{"text":"A complete story ..."}` JSON object per
line, with separate train and validation files:

```bash
uv run python main.py +experiment=tinystories \
  data.train_path=/path/to/train.jsonl data.validation_path=/path/to/validation.jsonl
```

Cache identity includes source revision (or local file SHA256), split, source
limit, sequence length, and prompt length. Keep the pinned revision for comparisons.
An existing matching cache is reused without source access. Validation-only
standalone evaluation never prepares training data.

## Tokens and masks

The dependency-free baseline uses **UTF-8 bytes**, not subword tokens: 256 byte
values plus PAD=0, BOS=1, EOS=2, MASK=3. `task.length=1024` includes BOS/EOS;
`task.prompt_bytes=64` is a fixed prefix budget. Prefixes back up to a valid UTF-8
boundary. A 1,024-byte budget is not equivalent to 1,024 BPE tokens. This small
vocabulary is useful for cheap corruption comparisons, but is not a reproduction
of published subword TinyStories models. There is currently no subword preset.

The prompt is fixed and excluded from loss/corruption. Fixed-canvas training
learns continuation, EOS **and PAD**, rather than leaking the correct answer
length through clean padding. Insertion training uses an explicit valid mask
through EOS and excludes PAD. Generation provides only the prefix and a fixed
maximum capacity; neither the true target length nor target EOS position is
provided. Samples are grouped by prefix length for insertion compatibility.

## Evaluation

The presets train 10k optimizer steps, with generation callbacks and checkpoints
every 1k. Native callbacks use the preset's selected sampler. Evaluation uses a
deterministic held-out prompt subset (default 128, seed 23). Cheap metrics cover
continuations only, stopping at the first EOS:

- `average_length`: number of generated token positions before EOS (bytes and any
  malformed special positions); `average_word_count` counts decoded words.
- `termination_rate`, `nonempty_rate`, `invalid_special_rate`, `invalid_utf8_rate`.
- `duplicate_rate`, corpus `distinct_1`/`distinct_2`, and average repeated-trigram
  fraction. N-grams never cross example boundaries.
- Optional `corpus_bleu`, using sacreBLEU's default corpus metric with its signature
  saved. Use `uv run --extra scoring` and set `task.calculate_bleu=true`. This compares
  a continuation against the one observed ending, so valid alternative stories
  can score poorly. It is a secondary overlap diagnostic, not coherence accuracy.

`termination_rate` is a placeholder primary **diagnostic**, not a quality score.
Do not select the best generator using EOS rate alone. Judge quality and diversity
together, and report identical generation budgets/sample counts.

Each saved result includes stable source-hash IDs, prompt, generated continuation,
reference continuation, reconstructed story, lengths and degeneration diagnostics.
Invalid UTF-8 is decoded with replacement and explicitly counted, not silently
classified as fluent output. EOS-free outputs are evaluated through capacity and
have termination=false; insertion truncation metadata is also retained by the
shared evaluator.

```bash
uv run python main.py +experiment=tinystories mode=evaluate \
  evaluation.checkpoint=/path/to/checkpoints/last.ckpt \
  data.train_path=/path/to/train.jsonl data.validation_path=/path/to/validation.jsonl \
  output_dir=/path/to/story-evaluation
```

Score the saved `results.json` with the separate optional judge tool (see
[story judge documentation](../guides/story-judging.md)). Judging is intentionally outside
training callbacks to avoid loading a 7B model alongside each training worker.
The task itself never calls an external service. References are examples of valid
continuations, not the only allowable stories.

## Offline verification

```bash
OMP_NUM_THREADS=1 uv run pytest core/tests/test_tinystories.py -q
```

This tests exact token/mask semantics, no answer leakage, filtering, metric EOS
handling, and actual CPU train → callback → checkpoint → standalone evaluation
for fixed and insertion presets. Fixtures do not establish real-corpus quality or
training convergence. No full corpus, judge model, or GPU training is downloaded
or launched by these tests.

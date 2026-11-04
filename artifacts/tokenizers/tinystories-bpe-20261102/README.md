# Frozen TinyStories BPE tokenizers

These five standalone Hugging Face tokenizer JSONs are copied byte-for-byte from
`tinystories-bpe-20261102`, not retrained. They are intentionally versioned in Git
so a clone supplies the tokenizer artifacts. Total JSON size is about 1.1 MB.

Verify from this directory:

```bash
sha256sum -c SHA256SUMS
```

SHA256SUMS was computed from the transferred files at vendoring time; the original
source manifest contains parquet hashes, not independent tokenizer hashes.
`source-manifest.json` preserves that original audit, including its historical
source path. It is provenance, not an executable campaign manifest.

Training source: roneneldan/TinyStories revision
`f54c09fd23315a6f9c86f9dc80f725de7d8f9c64`, all training shards. Byte-level BPE,
nested prefixes of one learned merge sequence, preserving case and Unicode.
PAD/BOS/EOS/MASK IDs are 0/1/2/3. BOS/EOS are not inserted automatically.

Configure the existing task with explicit absolute paths (the checkout must be
visible at the same path on all Ray workers):

```bash
# Append these overrides to the campaign preparation command:
# task=battery/tinystories_bpe bpe.vocabulary_size=10000
# bpe.tokenizer_path=/shared/candi/artifacts/tokenizers/tinystories-bpe-20261102/bpe-10000.json
```

For the vocabulary ladder, select the matching file and vocabulary size together.
Existing cache-path defaults are preserved for historical runs. Datasets, length
arrays, caches and frozen cohort/tuning/test split IDs are not included here;
prepare or transfer those separately according to the campaign handoff.

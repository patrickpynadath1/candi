# TinyStories BPE tokenizers

The five byte-level BPE tokenizers used by the tokenizer-ladder and BPE-10k experiments
(vocabularies 260, 500, 1000, 5000, 10000). They are versioned here so a clone has them;
`task=battery/tinystories_bpe` loads `bpe-<vocabulary_size>.json` from this directory by
default.

- Trained on all four training shards of `roneneldan/TinyStories` at revision
  `f54c09fd23315a6f9c86f9dc80f725de7d8f9c64` (`source-manifest.json` lists the shard hashes).
- Nested prefixes of one learned merge sequence; case and Unicode preserved; 260 is the
  minimum lossless byte alphabet plus four specials.
- PAD/BOS/EOS/MASK ids are 0/1/2/3; BOS/EOS are not inserted automatically.

Verify the files with `shasum -a 256 -c SHA256SUMS` (or `sha256sum -c SHA256SUMS`).

They were built with [`scripts/build_bpe_tokenizers.py`](../../../scripts/build_bpe_tokenizers.py), which also
audits full-story lengths at each vocabulary size.

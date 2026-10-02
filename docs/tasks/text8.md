# Text8

[Text8Task](../../core/tasks/text8.py) is unconditional character generation
with word-count metrics, not model likelihood. The task default length is 256;
the [Text8 experiment](../../configs/experiment/text8.yaml) selects length 128.

```bash
uv run python main.py +experiment=text8 data.raw_dir=/path/to/text8
uv run python main.py +experiment=text8 data.download=true
```

## Data and tokenization

Provide `text8.train.txt`, `text8.validation.txt`, and `text8.test.txt` under
`data.raw_dir`, or a `text8.zip` containing the standard 100M-character `text8`
file. Explicit `data.download=true` permits downloading the archive. The split
is 90M/5M/5M characters; input must contain lowercase a–z and spaces.

[Preparation](../../core/data/text8.py) caches arrays by source hash and sequence
length. Packing splits the source into length-L chunks, appends EOS, groups
eight chunks, repacks bodies of length L−2, then wraps BOS/EOS. Incomplete tails
are dropped. It uses one deterministic preprocessing shard; arbitrary legacy
multi-worker packing boundaries are not guaranteed identical.

The 35-token vocabulary has eight special IDs followed by a–z and space.
Mask ID is 4. All positions participate in training loss/corruption. Held-out
windows provide comparison samples only, never conditioning. Evaluation selects
up to `task.num_eval_samples` windows (default 128) without replacement, using
`task.evaluation_seed=23`. `task.evaluation_split` selects validation or test.

## Word counts and rates

A valid word belongs to the vocabulary extracted **only from the training
split**, with its source SHA256 and vocabulary size stored in results.
Special tokens split text into segments; they never join neighboring words.
Potentially truncated boundary words are discarded from each segment. Generated
and held-out windows use identical extraction and scoring.

For each sample and for the complete corpus, the metric stores total word
occurrences `total`, valid occurrences `valid`, distinct words `unique`, and
distinct valid words `unique_valid`. Corpus uniqueness uses a union across all
samples. Mean per-sample uniqueness is also stored and can differ substantially.

The aggregate rates are:

| Field | Formula |
| --- | --- |
| `rates.valid` | valid occurrences / total occurrences |
| `rates.unique` | distinct words / total occurrences |
| `rates.unique_valid` | distinct valid words / distinct words |

Zero denominators produce `null`, not zero. The same counts/rates are computed
for inclusive minimum lengths in `task.min_word_lengths`, default `[3,5,7]`.
The top-level `valid_word_rate` is the generated unfiltered valid rate used as
the primary metric. Records also store decoded text, character count, and
special-token count for generated and held-out samples.

Implementation: [text8_metrics.py](../../core/tasks/text8_metrics.py) and shared
[word utilities](../../core/tasks/metrics/text.py). Use both valid-word counts
and diversity when comparing models: repeating a single valid word can yield
a high validity rate. Frontier comparisons need equal sample counts and lengths.
See [frontier sweeps](../guides/frontier-sweeps.md) and the
[permuted-AR pilot](../validation/reproduction-results.md).

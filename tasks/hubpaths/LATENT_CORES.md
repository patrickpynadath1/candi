# Latent core tokens: unmarked graph language

This mode generates node sequences with hidden core occurrences. The model sees
only a conventional BOS, the finished node stream, and trailing PAD. There is no
waypoint prefix, no internal separator, no role embedding, no core-position loss
mask, and no revealed topic. The data loader returns only `input_ids` and
`attention_mask`; sidecar arrays are reserved for diagnostics.

A node type has no permanent core/connector role. Even a hub can be selected as a
core, and a low-traffic node can appear inside a connecting bridge. Repeated cores
are allowed after at least one intervening core. A token matching a later core
may also occur incidentally in an earlier bridge: the hidden **occurrence**, not
its first appearance in the document, defines the anchor. This differs from the
first-visit ordering rule of the older prompted navigation task.

## Generative rule

Generate one graph and calibrate its bridge traffic. The pilot chooses uniform
goals and uniform starts in the configured distance band, samples first-hit RSP
bridges at the configured temperature, and counts interior visits only. Endpoints
are excluded, so bridge frequency measures connector traffic rather than endpoint
selection. Add a positive pseudocount and freeze the resulting frequencies `f_j`.
This is an empirical calibration under a neutral endpoint prior, not exact
betweenness or a claim that the final corpus has that same connector distribution.

Draw and freeze independent per-node baseline log frequencies, source/target
rank-r factors, and topic loadings. For a preceding core i, candidate core j, and
current topic z, the next-core law is proportional to:

```text
exp(b_j) * (1 + N*f_j)^(-beta)
  * exp(lambda * dot(u_i, v_j)/sqrt(rank))
  * exp(tau * dot(a_j, z)/sqrt(topic_dim))
  * exp(-0.5 * ((distance(i,j) - distance_target)/distance_width)^2)
  * 1[min_distance <= distance(i,j) <= max_distance]
```

The baseline `b_j ~ Normal(0, baseline_log_std)` gives unequal token frequencies.
The bridge-frequency factor is finite and positive: central nodes are discouraged
as cores, not categorically excluded. `u` and `v` are frozen N-by-r Gaussian
matrices. Their dot products create recurring asymmetric pair preferences without
an independent random score for every pair. No N-by-N random matrix is created
or saved. Graph-derived distance vectors use a bounded cache.

The first core omits pair and distance terms. Every document starts with
`z_0 ~ Normal(0,I)`. Between cores, update:

```text
z_next = rho*z + sqrt(1-rho^2)*Normal(0,I)
```

The stationary AR(1) topic changes slowly when rho is near one. Its loadings stay
fixed across every document and split. Topics reset between documents; increasing
`n_cores` extends within-document coherence. Set `topic_strength=0` for a no-topic
control, or `correlation_strength=0` for a no-pair-factor control. Distance alone
can still induce consecutive-core dependence in either control.

After drawing cores, independently sample one RSP bridge per consecutive pair.
Each bridge has its own `bridge_max_steps` horizon and stops at the first arrival
at its endpoint. Concatenate bridges with their shared endpoint emitted just
once. Every sequence edge is a graph edge. The maximum stored length, including
BOS, is `2 + (n_cores-1)*bridge_max_steps`; defaults give 72 slots (8 cores, up to
10 steps per bridge). Shorter documents use ordinary PAD. All non-BOS slots,
including PAD, contribute to loss. These bridge budgets replace the old single
whole-route horizon in this mode.

The exact bridge entropy is conditional on an endpoint pair and its bridge
horizon. Summing these entropies gives the entropy of the **segmented bridge
tuple given the latent cores**. Marginalizing hidden cores, topics, and segment
boundaries to score the visible sequence is a different problem. The code does
not label the summed entropy as visible-sequence entropy or subtract it from a
language-model NLL to claim an exact KL.

## Generate and analyze

From the repository root:

```bash
.venv/bin/python -m tasks.hubpaths generate --mode latent_core \
  --preset tiny_debug --output data_cache/hubpaths/latent_core_debug \
  --set n_train=2000 --set n_val=128 --set n_test=128 \
  --latent-set pilot_paths=1024

.venv/bin/python -m tasks.hubpaths diagnose data_cache/hubpaths/latent_core_debug \
  --output analysis_results/hubpaths/latent_core_debug --ngrams 1,2,3,4,5
```

Output directories must be new. Use `--set` or `--config` for graph parameters and
split sizes; use `--latent-set` or `--latent-config` for the latent generator.
All resolved parameters and frozen fields are persisted. For the 10,000-node
graph, use `--preset language_like` and set desired split sizes explicitly.
Calibrating more pilot paths improves frequency estimates at greater cost.

This mode uses train/validation/test document splits. Identical finished node
sequences cannot occur in different splits; the generator retries a conflicting
document. Core combinations intentionally recur. Pair-holdout OOD and coverage
top-ups belong to the prompted task and are not applied here, because they would
change this language prior. Metadata lists those unused conditional-task knobs
and counts retries for both infeasible chains and duplicate documents. A node
without any feasible successor can force a whole-chain retry; that can further
change the accepted prior.

The reusable interface is:

```python
from tasks.hubpaths import LatentCoreConfig, build_latent_dataset, preset

build_latent_dataset(
    preset("tiny_debug"),
    LatentCoreConfig(correlation_rank=8, topic_persistence=0.95),
    "data_cache/hubpaths/my_latent_corpus",
)
```

`core_prior.npz` stores baseline logits, the two low-rank factors, topic loadings,
pilot counts, and softened base probabilities. Per-split `*_latent_*.npy` files
record cores (node IDs), core positions (excluding BOS), topic states, normalized
core-choice log probabilities conditional on topics, shortest bridge distances,
and exact bridge entropies/log probabilities. None are returned to the model.
Core positions are essential: matching core token IDs against the sequence alone
would misclassify incidental appearances as core occurrences.

## Measurements

The ordinary `report.json`, token-frequency TSV, n-gram SQLite database, and plots
remain available. The default n-gram scope counts the entire unmarked node stream,
without BOS or PAD and without crossing document boundaries. `--scope prompt` is
rejected because the model has no waypoint prompt.

Additional outputs:

- `latent_core_report.json`: realized core unigram entropy and vocabulary,
  consecutive-core mutual information, per-position MI, graph-distance histograms,
  exact bridge entropy histograms and means by distance, topic persistence,
  and the core-versus-connector hub fractions.
- `core_frequencies.tsv`: every node's realized core count/frequency, connector
  count, pilot bridge count, intended baseline and softened base probabilities,
  and expected probabilities before/after the distance filter.
- `consecutive_core_counts.tsv`: sparse observed ordered-pair counts. These are
  measurements, not a generated random pair-score table.
- `core_prior_vs_realized.svg`, `bridge_visitation_vs_core_frequency.svg`,
  `core_distance.svg`, and `bridge_entropy_bits.svg`.

Expected before/after frequencies average each distribution over the **realized
core/topic contexts**. They expose the filtering shift and remaining empirical
sampling deviations, not a closed-form stationary marginal. Additional per-context
filter TV and surviving prior mass are evaluated on up to 256 evenly spaced
documents, with the context count reported. This catches shifts that can cancel
when averaging over different contexts.

Consecutive-core MI uses empirical directed pair counts. Sparse finite samples
bias this estimate upward. The report includes eight shuffled controls which
preserve each core position's marginals, plus the observed-minus-shuffled value.
The difference is a diagnostic, not an unbiased or universally nonnegative MI
estimator. Distance restrictions, topics, and low-rank factors can all contribute
to observed dependence; the report does not attribute all MI to pair factors.

## Parameters

| Latent field | Default | Meaning |
|---|---:|---|
| `n_cores` | 8 | Hidden core occurrences per document |
| `bridge_max_steps` | 10 | Per-bridge first-hit walk horizon |
| `min_distance`, `max_distance` | 2, 8 | Hard candidate distance band |
| `distance_target`, `distance_width` | 4, 2 | Gaussian distance preference within the band |
| `baseline_log_std` | 1 | Spread of frozen baseline log frequencies |
| `bridge_avoidance` | 1.5 | Strength of the soft traffic penalty |
| `bridge_pseudocount` | 1 | Smoothing for calibration counts |
| `pilot_paths` | 2048 | Number of neutral calibration bridges |
| `pilot_starts_per_goal` | 32 | Pilot starts sharing a goal DP table |
| `correlation_rank` | 8 | Rank of frozen pair compatibility factors |
| `correlation_strength` | 1 | Scale of pair compatibility logits |
| `topic_dim` | 4 | Latent topic dimension |
| `topic_strength` | 0.75 | Scale of topic preference logits |
| `topic_persistence` | 0.95 | Stationary AR(1) correlation per core transition |
| `distance_cache_size` | 32 | Maximum cached distance vectors |
| `max_attempts_per_example` | 1000 | Bound on infeasible/duplicate-document retries |
| `seed` | 0 | Independent seed for calibration, frozen fields, and document RNGs |

## Training

```bash
CUDA_VISIBLE_DEVICES=2 .venv/bin/python main.py +experiment=hubpaths_latent_ar \
  strategy=single_gpu trainer.devices=1
```

This uses the same 512-wide, eight-block AR architecture, and default latent data
length 72. Set `data.prepared_dir` and `model.length` to match another generated
corpus. Normal validation NLL is computed on the observed stream. The conditional
waypoint evaluation callback is intentionally excluded and rejects latent-mode
data rather than exposing its hidden cores to the model.

The old conditional-waypoint run remains a different task and cannot be changed
into this data distribution by relabeling its existing sequences or checkpoints.

## Generated example and verification

`data_cache/hubpaths/latent_core_debug` contains 2,000 training documents and 128
each for validation and test, on the 200-node `tiny_debug` graph. It uses the
defaults above with `pilot_paths=1024`. Reproduce it with the generation command
above; reports are in `analysis_results/hubpaths/latent_core_debug`.

Across 16,000 training core occurrences, 198 nodes appear as cores. Hubs account
for 1.43% of core occurrences and 73.81% of connector occurrences. Consecutive
cores have mean distance 4.68 edges. Their plug-in mutual information is 1.546
bits versus 1.081 bits under within-position shuffled controls (0.465 bits
excess); the raw estimator has substantial finite-sample bias. Mean conditional
bridge entropy is 8.017 bits. Mean per-context total variation caused by distance
filtering/weighting is 0.120, measured over 1,792 contexts. These are small-corpus
measurements, not guarantees for other graph sizes or parameters.

Exact unique 1–5-gram counts on the unmarked training node streams are 200,
1,444, 8,453, 24,312, and 46,067 respectively. N-grams do not cross documents.

Validation: 58 HubPaths, latent-core, AR, and Sudoku unit tests passed, along with
Ruff checks. A two-step host-GPU training check of the 512-wide configuration
completed with validation NLL 5.30823. This checks integration, not convergence;
the small example vocabulary gives 25.4M parameters. The GPU check wrote to
`/tmp/hubpaths_latent_preflight` with W&B disabled.

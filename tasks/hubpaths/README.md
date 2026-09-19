# HubPaths: ordered-waypoint graph navigation

HubPaths trains a model to remember a fixed hidden graph and generate a continuous
walk through a supplied list of nodes, in order. The graph is never included in
the prompt. Multiple routes can be correct, with known probabilities under a
finite-horizon Gibbs distribution. Existing tasks keep their existing behavior.

## Prompt and response

With three waypoints, the stored sequence is:

```text
[w0] [w1] [w2] [SEP] [w0] ... [w1] ... [w2] [PAD] ...
```

`n_waypoints` defaults to **3**, and is fixed within a generated dataset. Set it to
2 for the original start/goal task, or raise it for longer waypoint lists. Nodes
in the prompt must be distinct. First visits to the requested nodes must occur
in prompt order: a future waypoint cannot be visited early. Already satisfied
waypoints and ordinary nodes can be revisited. The walk ends at its first arrival
at the final waypoint. `max_steps` covers the **entire** response, not each leg.

Node `i` has token ID `i + 3`; PAD=0, SEP=1, MASK=2. The fixed sequence length is
`max_steps + n_waypoints + 2`. Thus `tiny_debug` has length 21, and the default
77-step, three-waypoint task has length **82**. Use `max_steps=76` for length 81
with three waypoints. No extra BOS/EOS tokens occupy sequence slots.

Infilling training follows the repository convention: `input_ids` contains the
complete target, and `attention_mask` is a **loss/corruption mask**, zero on the
prompt and any revealed hints, one on all other response slots, including PAD.
`HubPathsDataset.infilling_input(i)` exposes the explicitly masked view. Models
perform their own training-time corruption. AR training uses the same full target
and prefix loss exclusion. AR datasets must use `n_hints=0`, since scattered future
hints would require a different causal prompt layout.

## Generate datasets and compute n-gram statistics

Run from the repository root, with the existing virtual environment:

```bash
.venv/bin/python -m tasks.hubpaths generate \
  --preset tiny_debug --output data_cache/hubpaths/tiny_debug

.venv/bin/python -m tasks.hubpaths diagnose data_cache/hubpaths/tiny_debug \
  --ngrams 1,2,3,4,5 --entropy-prompts 128
```

Use a new output directory for each run. Generation refuses to overwrite existing
data. Diagnostics also refuses to reuse an existing counts database, preventing
accidental double counting. Select another `--output` for repeat analyses.

A larger example:

```bash
.venv/bin/python -m tasks.hubpaths generate --preset language_like \
  --output data_cache/hubpaths/language_like \
  --set n_waypoints=4 --set n_train=100000 --set n_val=1000 --set n_test=1000

.venv/bin/python -m tasks.hubpaths diagnose data_cache/hubpaths/language_like \
  --output analysis_results/hubpaths/my_language_like --ngrams 1,2,3,4,5,6
```

`--set KEY=JSON_VALUE` can override every config field; quote string values, e.g.
`--set 'backbone="chain"'`. `--config file.json` supplies a JSON override object.
The full resolved configuration is saved in both `graph.npz` and `metadata.json`.
The production preset requests two million training examples; generation is
explicit and never starts implicitly when a model is constructed.

Diagnostics writes:

- `report.json`: unique n-grams, total occurrences, singleton counts,
  type/token ratio, repetition fraction, empirical n-gram entropy, and the top
  20 n-grams for every requested order. It also contains every token's frequency,
  unique response counts, unigram entropy, redundancy, fitted rank-frequency
  slope, graph degree statistics, train coverage, and route histograms.
- `token_frequencies.tsv`: token ID, node ID, count, normalized frequency.
- `ngram_counts.sqlite`: **all exact counts**, using integer token tuples encoded
  as little-endian int32 blobs. Query table `counts(n, gram, count)`; table
  `sequences(tokens, count)` stores complete response counts.
- SVG plots: token rank-frequency, degree distribution, path length, shortest
  distance, stretch, hubs visited, and conditional entropy. No plotting package
  is needed. `powerlaw`, if already installed, adds optional fits.

The default scope is the response path, with PAD excluded. N-grams **never cross
example boundaries**. `--scope prompt` counts the waypoint lists, while
`--scope full` includes the prompt and SEP and permits windows across the
prompt/response boundary within an example. `--split` selects train, validation,
test, or test_ood. `--max-examples` caps counting and is recorded in the report.
N-gram counting uses SQLite with bounded Python buffers; it does not approximate
unique counts with hashes. Higher n-gram orders can require substantial disk space.

`--entropy-prompts` limits the expensive exact entropy and shortest-path-count
calculations to an evenly spaced sample of examples. The default is 128; it does
**not** limit n-gram or token counts. Reports state this sample size. Conditional
entropy averaging follows the sampled example distribution, which can contain
repeated prompts. The count of shortest routes ignores hints; entropy honors them.

## Query the known distribution

The following takes **node IDs**, not offset token IDs, and returns actual paths:

```bash
.venv/bin/python -m tasks.hubpaths sample data_cache/hubpaths/tiny_debug \
  --waypoints 4 9 2 --count 4 --seed 7
```

The command fails clearly when the supplied route is impossible within the saved
horizon or under waypoint-order constraints. This is the ground-truth sampler;
trained-model sampling runs through the repository's `generate_samples` API.

Python interfaces:

```python
from tasks.hubpaths import Graph, RSP, RouteDistribution, verify_route

graph = Graph.load("data_cache/hubpaths/tiny_debug/graph.npz")
route = RouteDistribution(RSP(graph), [4, 9, 2])
# route.sample(np.random.default_rng(0)) returns node IDs.
# route.true_logprob(path_token_ids) returns nats, or -inf when invalid.
# route.cond_entropy_bits(), .first_step_marginal(), .length_distribution()
result = verify_route(graph, [4, 9, 2], path_token_ids, max_steps=16)
```

`verify_path(graph, start, goal, path, ...)` and `verify_batch` preserve the original
two-endpoint interfaces. `verify_route` and `verify_route_batch` handle arbitrary
waypoint lists. All verifier paths are **token IDs**; scalar waypoint and hint
values are **node IDs**. Batch `prompts` are **token IDs**, matching model inputs.
Batch edge membership uses vectorized search over sorted directed edge keys.
The verifier does not import or invoke the sampler.

## Probability, splits, and coverage

The reference neighbor probabilities are proportional to destination
`weight**alpha`. An edge has weight `P_ref * exp(-cost/temperature)`, where
`cost = edge_cost + hub_cost * destination_is_hub`. The backward DP runs over
(position, waypoint progress, node). Intermediate arrivals advance waypoint
progress without resetting the remaining budget. The terminal state is absorbing.
Every partition function is computed in log space; entropy and marginals derive
from those same transitions. `length_mode=exact` requires exactly `max_steps`
edges; `at_most` permits earlier arrival and pads the remaining slots.

Generation amortizes a table across `starts_per_goal` starts sharing an ordered
waypoint suffix. `goal_batch_size` suffixes are computed as dense columns in each
sparse update. Memory scales approximately as
`8*(max_steps+1)*(n_waypoints-1)*n_nodes*goal_batch_size` bytes for tables, plus
sparse-update temporaries. Decrease the batch size if needed. A bounded number of
suffix tables are persisted for evaluation; the rest regenerate on demand.

The distance band applies to the shortest **whole ordered route**, including
restrictions against prematurely visiting a future waypoint. Final goals are
drawn according to `weights**endpoint_alpha`; feasibility and split quotas then
condition the accepted prompt population. Intermediate waypoint IDs are drawn
without replacement. Starts are uniform among eligible nodes for that suffix.

Unordered endpoint-pair hashing reserves train/validation/test partitions before
sampling, so even reversed endpoints cannot leak across splits. Full waypoint
lists are also unique up to reversal within each evaluation split. Selected
unordered leaf-cluster pairs define OOD endpoint prompts. Train/validation/test
prompts containing a held-out cluster pair anywhere in the waypoint list are
excluded. Router nodes have cluster ID -1. OOD paths may of course traverse edges
seen during training; the holdout concerns prompts, not graph connectivity.

Training counts include coverage top-ups. These use short waypoint prompts but
sample from the same full-horizon path law; they do not force a sampled edge or
change the conditional likelihood. Top-ups can fall outside the normal distance
band. Coverage counts are undirected (the graph is undirected). Split reservations,
ordering constraints, or an insufficient attempt budget can prevent complete
coverage. The builder warns and records `coverage.edges_below_target` rather than
silently violating the splits. Inspect that field before treating a corpus as
fully covered. `min_edge_count=0` disables top-ups for quick exploratory runs.

`require_simple=true` uses rejection and verification to prohibit repeated nodes.
The walk partition function then no longer normalizes the accepted distribution;
reported walk log probabilities and entropies are explicitly **not** exact for the
simple-path law. A warning and metadata flag accompany that setting.

Hints reveal random interior positions from a sampled target and leave the joint
training samples intact. The constrained DP computes the walk law conditioned on
those **fixed position/value constraints**. It does not incorporate the additional
length-dependent probability of the random hint-selection procedure. Thus a KL
interpretation against the full observed-hint data distribution is exact only in
the no-hints case. Prompt suffix cache entries are unhinted; hinted recursions are
computed separately.

## Training and evaluation

```bash
CUDA_VISIBLE_DEVICES=2 .venv/bin/python main.py +experiment=hubpaths_ar \
  strategy=single_gpu trainer.devices=1

CUDA_VISIBLE_DEVICES=2 .venv/bin/python main.py +experiment=hubpaths_mdlm \
  strategy=single_gpu trainer.devices=1
```

Both configs use `data_cache/hubpaths/tiny_debug` and length 21. Change
`data.prepared_dir` and `model.length` together for another dataset. The loader
checks the length against metadata. Other infilling algorithms can reuse the data
interface when their sampler supports arbitrary prompt masks.

The callback evaluates test and test_ood separately and groups metrics by shortest
ordered-route distance. It reports validity, failure counts, shortest rate,
valid-path stretch and length, distinct valid paths per prompt (`unique_valid@K`),
true sample NLL in **nats**, and first-step/length total variation. TVs are computed
from valid samples only; prompts with no valid samples contribute null values and
are counted separately. Shortest rate uses all samples as its denominator.

AR reports path-total `test_nll_bits`; MDLM reports a stochastic likelihood-bound
estimate, also in path-total bits. `kl_bits` subtracts the exact conditional walk
entropy, so it is an expected KL estimate for AR, or a bound-derived quantity for
MDLM, rather than a guaranteed nonnegative finite-sample statistic. PAD prediction
is included in model NLL; prompt and revealed hint slots are excluded.

`subset_size` controls the number of prompts per split, `samples_per_prompt`
defaults to 16, and `num_steps` controls infilling generation. `full_eval_at_end`
means the callback runs at training end using that configured subset; it does not
silently evaluate all 5,000 prompts. Increase `subset_size` to cover a whole split.

## Presets and measured interactions

- `tiny_debug`: 200 nodes, 8 clusters, one hierarchy level, horizon 16, three
  waypoints, 2,000 train / 100 validation / 100 test / 100 OOD before top-ups.
- `language_like`: 10,000 nodes, gamma 2.1, three hierarchy levels, ring backbone,
  temperature 1, horizon 77. Three waypoints yield length 82.
- `sudoku_like`: the language-like graph at temperature 0.05 with up to eight
  interior hints. This is a starting point; it does not promise a unique route.
- `flat_control`: gamma 6, no hierarchy, complete cluster-hub backbone. Hubs and
  route constraints still affect observed frequencies; measure flatness.

Lower gamma can create shortcuts and shorten routes. Counter with backbone,
hub cost, and the distance band. Graph size alone need not increase distances in
hub-heavy graphs. Temperature affects both route lengths and token skew, so choose
it before interpreting skew sweeps. Chung–Lu clips edge probabilities; the
realized exponent can differ substantially from the nominal gamma, especially
below 3. A third hierarchy level can merge all 50 clusters into a single top-level
unit, making the chosen backbone irrelevant for that particular configuration.

Finite nonzero temperature retains some probability on non-shortest walks.
Least-cost routes need not minimize hop count when hub_cost is nonzero. Also,
Zipf slope need not change monotonically with gamma on a finite graph, and its
negative sign must not be confused with the positive power-law exponent.

```bash
.venv/bin/python -m tasks.hubpaths sweep --output analysis_results/hubpaths/sweep \
  --knob gamma --values '[2.1,2.5,3.0,4.0]'
```

The same command accepts `temperature`, `backbone`, `max_dist`, or `n_hints`.
Each setting writes its own data and diagnostics; `sweep.csv` and `sweep.json`
tabulate Zipf slope, mean path length, vocabulary used, and mean conditional
entropy. These are measured outcomes, not assertions of a target exponent.

## Configuration reference

All fields below are persisted. Defaults refer to the base config, before presets.

| Field | Default | Meaning |
|---|---|---|
| `n_waypoints` | `3` | Number of distinct ordered nodes in each prompt. |
| `n_nodes` | `10000` | Exact vocabulary node count, including newly created router nodes. |
| `max_steps` | `77` | Global edge budget; sequence length is max_steps + n_waypoints + 2. |
| `gamma` | `2.5` | Exponent of the truncated leaf-weight power law. |
| `mean_degree` | `8` | Target mean of rescaled leaf weights. |
| `min_degree` | `2` | Lower bound before rescaling weights. |
| `max_degree` | `None` | Upper bound before rescaling; null uses cluster size minus one. |
| `alpha` | `0.0` | Destination popularity bias in the reference walk. |
| `endpoint_alpha` | `0.0` | Popularity bias for final goals and intermediate waypoint draws. |
| `n_clusters` | `50` | Number of leaf clusters. |
| `cluster_size_dist` | `equal` | equal or powerlaw leaf-cluster allocation. |
| `hubs_per_cluster` | `2` | Highest-weight leaf nodes selected as exits. |
| `n_levels` | `2` | Number of router hierarchy levels. |
| `branching` | `5` | Child units per parent unit. |
| `hubs_per_group` | `2` | New routers per parent unit; sibling links guarantee connectivity. |
| `hub_link_prob` | `1.0` | Parent/child hub edge probability, with a guaranteed link per child. |
| `backbone_link_prob` | `1.0` | Hub link probability between adjacent top-level units. |
| `backbone` | `ring` | ring, chain, tree, grid, random, or complete. |
| `backbone_shortcuts` | `0` | Extra distinct random top-level unit edges, capped by availability. |
| `bridge_rate` | `0.0` | Number of attempted non-hub cross-cluster edges per graph node. |
| `temperature` | `1.0` | Gibbs temperature; strictly positive. |
| `edge_cost` | `1.0` | Base per-edge cost; strictly positive. |
| `hub_cost` | `0.0` | Additional cost on entering a hub; total cost must stay positive. |
| `length_mode` | `at_most` | at_most or exact global horizon. |
| `require_simple` | `False` | Reject repeats; invalidates closed-form walk-law normalization for accepted paths. |
| `min_dist` | `4` | Minimum shortest admissible whole-route distance. |
| `max_dist` | `24` | Maximum shortest admissible whole-route distance; at most max_steps. |
| `n_hints` | `0` | Maximum random interior response positions revealed for infilling. |
| `n_train` | `2000000` | Base training example count before coverage top-ups. |
| `n_val` | `5000` | Unique validation waypoint-list count. |
| `n_test` | `5000` | Unique test count, also used for OOD when enabled. |
| `starts_per_goal` | `256` | Start draws sharing each ordered waypoint suffix. |
| `min_edge_count` | `5` | Target undirected training-edge coverage; zero disables top-ups. |
| `holdout_cluster_pairs` | `20` | Number of unordered leaf-cluster pairs held out for OOD. |
| `seed` | `0` | Graph seed; fixed derived seeds govern splits and path generation. |
| `goal_batch_size` | `4` | Suffix columns evaluated together in sparse log-space recursion. |
| `cache_goals` | `4` | Maximum saved suffix tables and in-memory two-endpoint tables. |
| `max_sampling_attempts` | `100000` | Global coverage top-up budget and per-path simple rejection cap. |

## Validation and example report

Run `.venv/bin/python -m pytest tests/unit/tasks/test_hubpaths.py -q`.
Tests enumerate tiny finite walk spaces, compare all probabilities and entropies,
check fixed-seed sample frequencies, exercise 128-step low-temperature numerics,
compare scalar and batched verifiers on 10,000 sequences, and check split and
byte-level reproducibility. Additional Sudoku and AR tests cover existing behavior.

The generated 10,000-node report and GPU smoke-check results are summarized in
`docs/hubpaths_report.md`. No heavy dependency was added.

## Sudoku-sized AR run with W&B

`hubpaths_ar_sudoku_size` matches the Sudoku AR backbone (8 blocks, width 512,
8 heads, conditioning width 128, dropout 0.1) and its 20,000-step, batch-256
schedule with 2,500 warmup steps. With 10,003 HubPaths tokens it has 35,448,083
parameters; the same backbone with Sudoku's 12 tokens has 25,207,308 parameters.
The difference is the untied input/output vocabulary layers.

For a durable run with a frozen source snapshot, persistent logs and checkpoints:

```bash
.venv/bin/python new_scripts/launch_hubpaths_ar.py \
  --dataset data_cache/hubpaths/language_like_100k_seed0 \
  --output outputs/hubpaths_ar_language_like_20k_seed1_20260919 \
  --gpu 2 --entity rz-text-dlp --detach
```

The dataset must already be generated. `run.json` records the command, dataset
metadata, source path, and W&B URL. `status.json` records the training PID and
completion/exit status, and `train.log` contains live training output. W&B uploads
training/evaluation metrics and configuration; model checkpoints stay in the
local run directory. The launch refuses to replace an existing run.

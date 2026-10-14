# Mazes

The four presets use `maze-dataset==1.4.2` to generate supplied-map path tasks:

| Preset | Target | Sequence model |
| --- | --- | --- |
| `maze` | Start-to-goal path | Fixed canvas |
| `maze_insertion` | Start-to-goal path | Insertion/deletion |
| `maze_waypoints` | Path through two ordered waypoints | Fixed canvas |
| `maze_waypoints_insertion` | Path through two ordered waypoints | Insertion/deletion |

Both sequence models use the same masked hybrid assembly, so the insertion
modifier can be compared without changing corruption. These are task adaptations,
not a claim to reproduce a published maze benchmark exactly.

## Run

```bash
uv sync --locked --extra mazes
uv run --extra mazes python main.py +experiment=maze output_dir=core/outputs/maze
uv run --extra mazes python main.py +experiment=maze_waypoints_insertion \
  output_dir=core/outputs/maze-waypoints-insertion

uv run --extra mazes python main.py +experiment=maze mode=evaluate \
  evaluation.checkpoint=core/outputs/maze/checkpoints/last.ckpt \
  output_dir=core/outputs/maze-eval
```

The optional dependencies pin `muutils` and `zanj` because newer releases remove
an import required by this generator version. An alternative requirements file is
`core/tasks/requirements-maze.txt`. No maze corpus download is required.

Defaults are 10×10 grids, mixed DFS trees and cyclic DFS/percolation maps,
4,000 training maps and 256 each for validation and test. Training runs 5k
optimizer steps with generation evaluation and checkpoints every 1k; the default
callback uses 64 validation maps. The native sampler uses 127 transitions plus
one greedy final prediction. These budgets are runnable starting points, not
measured convergence recommendations.

## Representation and leakage checks

The prompt contains the connectivity bit mask of every cell in row-major order,
then start, optional waypoints and goal. The answer is a sequence of cell IDs and
EOS. Generation receives a fixed capacity of 256 tokens including the prompt,
never the reference route length. Fixed-canvas training supervises response
padding; insertion training uses the actual valid-token mask through EOS.

Insertion starts with the map/anchor prompt and grows the answer. It does not
start with waypoint tokens embedded in an editable solution canvas. To change grid
size, update `task.vocab_size` to `32 + grid_size**2` and increase
`task.length` to fit the prompt and maximum route.

An independent BFS constructs labels and checks optimal path length. Waypoints
must occur as an **ordered subsequence**: walks and revisits are allowed, including
passing a later waypoint early and visiting it again in the required order.

## Splits and metrics

Generation is seeded and cached by configuration and generator version. Train,
validation and test have distinct map topologies, including equivalence under
rotation/reflection; changing only endpoints cannot move a map to another split.
The cache is reused on subsequent runs. The task accepts validation or test for
standalone evaluation via `+evaluator.evaluation_name=test`; reserve test for final reporting.

Success requires an intact prompt, EOS within the generated length, valid cell
IDs, legal adjacent steps, correct endpoints and ordered waypoints. Any valid
alternative route counts. Reports include success, optimal-solution rate,
endpoint/waypoint validity, termination, route length and successful-route excess
length, with tree/cyclic breakdowns. Exact reference matching is not required.
Set `task.eval_repeats=4` to measure unique valid routes per maze. Cyclic maps offer
alternative paths; on trees, valid walks can still differ through backtracking.

## Verification

```bash
OMP_NUM_THREADS=1 uv run --extra mazes pytest core/tests/test_maze.py -q
```

Tests exercise real generation/cache reuse, symmetry-aware split separation,
answer-length isolation, padding, alternative solutions, illegal jumps, waypoint
order and EOS boundaries. All four presets also run a two-step CPU training,
callback, checkpoint and standalone evaluation round trip. These checks establish
execution and metric semantics, not trained-model quality.

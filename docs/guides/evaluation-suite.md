# Small-task evaluation suite

Use [algorithm definitions and task batteries](task-batteries.md) to run the suite
with one shared method definition and linked result provenance.

| Task | What it measures | Presets / documentation |
| --- | --- | --- |
| Sudoku hard | Clue-conditioned solution validity and clue-removal diversity | [Existing Sudoku task](../tasks/sudoku.md) |
| N-Queens | Constraint validity, clue consistency, valid-board diversity | `nqueens`, `nqueens10`; [guide](../tasks/nqueens.md) |
| Text8 | Character generation and lexical diagnostics | [Existing Text8 task](../tasks/text8.md) |
| TinyStories | Length, termination, repetition, diversity, optional reference BLEU and judged quality | `tinystories`, `tinystories_insertion`; [guide](../tasks/tinystories.md) |
| Mazes | Supplied-map route validity, optimality, waypoint constraints, variable-length generation | `maze`, `maze_insertion`, `maze_waypoints`, `maze_waypoints_insertion`; [guide](../tasks/maze.md) |

Tasks own data, tokenization, conditioning masks and correctness/quality metrics.
They do not select a corruption process, noise schedule, objective or sampler.
The experiment presets in the table combine a task with one example recipe;
those recipe choices are replaceable through the normal component configuration.
Fixed and variable-length tasks expose the different padding/generation contracts
needed by those modifiers, without implementing their diffusion mathematics.

Install optional maze and BLEU dependencies in the environment used for these tasks:

```bash
uv sync --locked --extra mazes --extra scoring
```

Include `--extra tracking` for W&B. These options do not themselves enable logging;
use the normal `wandb` configuration. Each task exports metrics through the shared
periodic evaluation callback and saves standalone evaluation records in
`results.json`. [Story judging](story-judging.md) is a separate pass over generated
records; it need not load a 7B judge into every training worker.

For hypothesis tests, keep data splits/seeds and model/training budgets matched.
Tune sampler/temperature on validation and report held-out results. Task presets
are executable starting points, not a single controlled cross-task ablation:
in particular the TinyStories insertion preset changes its corruption assembly.
Vocabulary, representation and length also differ across tasks. Do not interpret
word validity or BLEU alone as coherence, or N-Queens correctness as evidence of
unseen-solution discovery. Maze test maps are held out; N-Queens and TinyStories
currently expose training and validation splits, requiring an additional reserved
set before repeated tuning and final significance testing.

# Flow Reasoning Models on Sudoku-Extreme

This implementation follows arXiv:2606.29150 while keeping the repository's
prepared-data and evaluation formats intact. On disk, Sudoku remains a
180-token puzzle/solution record. `FRM` extracts the two grids and the learned
network sees exactly 81 positions with vocabulary `0..9`, matching Table 3 of
the paper. Generated grids are serialized back to 180 tokens for the existing
exact-solve and constraint checks.

## Components

- `models/frm_dit.py`: 256-wide, seven-block, eight-head non-causal DiT with
  separate projections for the continuous state, fixed clues, and recurrent
  probability carry. The shipping configuration has 7,031,050 parameters.
- `algos/frm.py`: categorical clean-prediction loss, conventional detached
  self-conditioning, and detached Fixed-Point Forcing rollouts.
- `samplers/frm.py`: recurrent Euler integration and optional held-time
  refinement. Every denoiser invocation, including the final readout, counts
  toward NFE.
- `configs/experiment/sudoku_extreme_frm_stage_{a,b}.yaml`: the paper's two
  100k-step stages and optimization hyperparameters.

The loss-bearing state is always the canonical interpolant
`x_t = (1-t) noise + t onehot(solution)`. Clue clamping affects only generated
rollout/inference states. FPF uses the same Gaussian noise at `t_start` and the
supervision time, and gradients never traverse its 16-call rollout.

## Train

Prepare the pinned EqR dataset as described in `docs/reasoning_tasks.md`, then:

```bash
new_scripts/train_sudoku_extreme_frm.sh 0 outputs/frm_sudoku_extreme_seed1 1
```

The launcher is resumable through each stage's `checkpoints/last.ckpt`. Run
three output roots with final arguments `1`, `2`, and `3` for the paper's
three-seed protocol.

To run the stages manually:

```bash
.venv/bin/python main.py +experiment=sudoku_extreme_frm_stage_a \
  checkpointing.save_dir=outputs/frm/stage_a hydra.run.dir=outputs/frm/stage_a

.venv/bin/python main.py +experiment=sudoku_extreme_frm_stage_b \
  training.finetune_path=outputs/frm/stage_a/checkpoints/last.ckpt \
  checkpointing.save_dir=outputs/frm/stage_b hydra.run.dir=outputs/frm/stage_b
```

Stage B loads only Stage-A model weights. Its optimizer and EMA state are
fresh, as required by the paper.

## Evaluate

```bash
.venv/bin/python main.py +experiment=sudoku_extreme_frm_stage_b \
  mode=reasoning_eval \
  eval.checkpoint_path=outputs/frm/stage_b/checkpoints/last.ckpt \
  reasoning.max_examples=1000 sampling.steps=64 \
  reasoning.output_dir=outputs/frm/eval_nfe64
```

Sweep `sampling.steps=16,32,64,128,256`. For held-time refinement, set
`sampling.predictor=frm_held` and add `+sampling.frm_flow_steps=16`; the
remaining NFE updates the carry while holding the terminal flow state fixed.

The paper does not disclose the exact sampler grids, clue-conditioning fusion,
validation IDs, or the operational meaning of every Table 3 field. These are
explicit choices here rather than claims of bitwise reproduction. The key
acceptance ladder is approximately 13% for the base flow, 33% after Stage A,
and 99% after FPF, measured as exact-grid solve rate.

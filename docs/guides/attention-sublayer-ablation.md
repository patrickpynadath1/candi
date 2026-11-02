# Replacing the position-wise MLP with expanded attention

`DiT.sublayer_factory` creates an independent second sublayer in every
`DDiTBlock`. The default is the existing 4x-width GELU MLP. Its parameter names
remain unchanged, so existing model checkpoints still load strictly.

The expanded-attention option is configured with a Hydra partial:

```yaml
model:
  backbone:
    sublayer_factory:
      _target_: core.models.sublayers.ExpandedAttention
      _partial_: true
      expansion: 2
```

The block supplies `dim` and `n_heads`. At residual width 512 and eight base
heads, the new sublayer projects Q/K/V to width 1024, uses 16 heads of size 64,
then projects back to 512. It receives the same RoPE positions and attention
mask as the first attention sublayer. Layer normalization, time modulation,
residual gating and dropout stay on the enclosing block. No corruption or
objective is implemented in the component.

The replacement has 2,097,152 projection parameters, versus 2,099,712 in the
MLP (the difference is biases). The matched eight-layer masked models have
28,539,148 and 28,559,628 parameters respectively. This matches parameter
budget approximately, not FLOPs: attention includes sequence-length-dependent
mixing. Both variants still predict token distributions separately.

The DiT wiring supports full-sequence attention masks, including causal masks,
but does not implement a second KV cache. Requests to use cached block
execution with a non-position-wise replacement fail explicitly.

## Sudoku experiment

`configs/sweep/masked_attention_sudoku.yaml` compares the MLP and expanded
attention with masked corruption and no self-conditioning. Both use Sudoku
Hard, eight layers, width 512, one seed, 20k optimizer steps, global batch 256,
BF16, torch.compile, learning rate 3e-4 and 2,500 warmup steps. Native,
native-greedy and EB-greedy sampling are tuned on validation at 128 NFE, using
the same held-out final evaluation protocol as the feedback campaign.

```bash
uv run --locked --extra tuning --extra scoring --extra tracking \
  python -m core.sweeps.prepare -m +sweep=masked_attention_sudoku
uv run --locked --extra tuning --extra scoring --extra tracking \
  python -m core.sweeps.prepare collect PLAN_DIRECTORY
```

Submit the collected manifest to `core.sweeps.nested_tuning`, as for other
experiments. The `block_sublayer` Hydra group is reusable across recipes with
a DiT backbone; the sweep itself only selects the masked Sudoku comparison.

## Gated sequence centering inside the MLP

`block_sublayer=centered_mlp` retains the standard 4x-width MLP and computes
`GELU(a - lambda * a.mean(dim=-2, keepdim=True))`, where `a` is the up-projection.
Each block has one unconstrained learnable scalar `centering_strength`, initialized
at zero. Thus initial computation and projection initialization match the ordinary
MLP exactly. Lambda may learn negative values or exceed one; it is not a sigmoid.
The reduction is within each example, across positions, before GELU. This is
centering, not projection orthogonal to the mean direction.

The component is independent of corruption and uses the same factory hook.
Currently it accepts full unpadded sequences only, and explicitly rejects attention
masks and KV caching rather than leaking information across masked boundaries.
This is suitable for fixed-length Sudoku, including its visible puzzle clues.

`+sweep=masked_centered_mlp_sudoku` prepares a fresh standard-MLP versus centered-MLP
comparison with the same 20k-step protocol above. The centered model adds eight
parameters (one per block). Learned scalars are stored in model checkpoints.

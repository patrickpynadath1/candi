# Blurring diffusion

Blurring diffusion corrupts a learned, unit-row-normalized input table in a
local DCT basis. Positional and feature attenuation are separable. The model's
larger output table is untied from this input geometry. The input table remains
differentiable through corruption by default; `detach_forward_embedding` only
cuts that corruption path. Block clean-stream encoding and optional clean AR
loss can still create their own representation gradient paths.

`SequenceLayout` supplies generic independent boundaries. Effective blur spans
are intersections of those boundaries with contiguous true corruption-mask
runs, excluding prompts and padding. Block diffusion supplies block boundaries
during training; active-block sampling naturally uses the current tail length.
Every span uses its own DCT and length-calibrated schedule, with equal lengths
batched and structural tensors cached.

Length one uses an identity DCT and the length-two shifted-geometric clock
calibration applied to the singleton DC mode. This provides a finite scalar
tail-block limit. For every length of at least two, the schedule is unchanged
from the source implementation. Stable exponential derivatives avoid the
source's `0 * inf` NaNs after high-frequency attenuation underflow.

`BlurAncestralSampler` is a stochastic exact Gaussian bridge. Each reverse
transition re-blurs the predicted clean modes to the target time and draws
noise except at the clean endpoint. The final decode performs one additional
network evaluation, so NFE is `steps + 1` (plus bootstrap evaluations when
that self-conditioning mode is selected).

Compose the standalone Sudoku recipe with:

```bash
python main.py +experiment=sudoku_blurring_hard
```

Enable independent block geometry through the existing modifier:

```bash
python main.py +experiment=sudoku_blurring_hard \
  block_diffusion.enabled=true block_diffusion.block_size=32
```

The tail block uses its actual length. Override schedule fields through
`noise_schedule`, for example `noise_schedule.feature_rho=1.0` for the exact
positional-only path.

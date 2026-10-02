# Blurring diffusion

Blurring diffusion has its own algorithm and spectral process. It corrupts a learned unit-row-normalized embedding table in a local discrete cosine transform (DCT) basis, with separable positional and feature attenuation. The larger model output table is untied from this input geometry.

For a local span with DCT matrix $B$, forward corruption takes the form

$$z_t=B\,[A_t\odot(B^T e(x_0))+\sigma_t B^T\epsilon].$$

The schedule owns attenuation and noise coefficients; the process owns the transform and Gaussian bridge; the sampler owns reverse iteration. The algorithm trains a posterior with CE and supplies the predicted representation mean. It requires raw-time conditioning without scalar Gaussian input preconditioning or Gaussian-specific objective weighting.

```bash
uv run python main.py +experiment=sudoku_blurring_hard
uv run python main.py +experiment=sudoku_blurring_hard block_diffusion.enabled=true block_diffusion.block_size=32
```

## Independent sequence spans

The generic `SequenceLayout` interface supplies independent boundaries. Effective spans intersect those boundaries with contiguous corruption-mask runs, excluding prompts and padding. The [block modifier](../modifiers/block-diffusion.md) supplies boundaries during training; active-block sampling uses the current tail length. Blurring therefore does not mix information across block boundaries.

Each span has its own DCT and length-calibrated schedule. Equal lengths are batched and structural tensors cached. Length one uses an identity DCT with the length-two shifted-geometric clock calibration applied to its DC mode, yielding a finite tail-block limit. Lengths at least two retain the reference schedule. Stable derivatives avoid the source's underflow-induced `0 * inf` NaNs.

## Sampling and gradients

`BlurAncestralSampler` draws from the exact Gaussian bridge conditional on its predicted clean modes. It re-blurs that estimate to the target time, adds transition noise except at the clean endpoint, and performs a final prediction. NFE is `steps + 1`, plus any explicit bootstrap calls.

Input embeddings remain differentiable through corruption. `algorithm.detach_forward_embedding=true` cuts that path only; block clean-stream encoding and optional clean AR loss can still update the representation. `noise_schedule.feature_rho=1.0` selects the positional-only path.

This assembly has implementation and integration checks, including independent block geometry. Do not interpret component parity as a matched trained-quality benchmark; consult [validation coverage](../validation/coverage.md).

Sources: [algorithm](../../core/algorithms/blurring_diffusion.py), [process](../../core/processes/blurring.py), [schedule](../../core/noise_schedules/blurring.py), [sequence layout](../../core/sequences.py).

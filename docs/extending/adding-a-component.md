# Adding a component

A component should own one mathematical or operational responsibility. Start by
reading [architecture](../concepts/architecture.md),
[contracts](../concepts/component-contracts.md), and the closest existing
implementation. A new class does not necessarily require a new algorithm.

## Choose the boundary

| You want to change… | Start in… |
| --- | --- |
| Time-to-corruption mapping | `core/noise_schedules/` |
| Training time distribution | `core/time_sampling/` |
| Forward/reverse transition equations | `core/processes/` |
| Mathematical meaning of network outputs or inference shaping | `core/parameterizations/` |
| Token-to-continuous encoding/decoding | `core/representations/` |
| Training loss/weighting | `core/objectives/` |
| Reverse-time integration, stopping, or commitment | `core/samplers/` |
| Neural architecture/conditioning | `core/models/` |
| Reusable architecture/sequence transformation | `core/modifiers/` |

Keep sampling loops out of algorithms and inference temperature/filtering out of
loss calculations. Do not make a predictor reach into a representation to discover
an embedding table implicitly; pass the tensors or conversion it needs through
the existing prediction orchestration boundary.

## Implement the contract

State the input domain, tensor shapes, time direction, output meaning, supported
limits, and constraints. Reuse the family-specific interface rather than inventing
a universal signature. For example:

- A keep-rate schedule returns keep rate and its derivative.
- Gaussian schedules return the path coefficients/derivatives consumed by Gaussian
  processes and samplers; a scalar sigma alone is not the whole contract.
- An objective must agree with whether its input is posterior log probabilities,
  plug-in weights, scores, or a continuous estimate.
- A sampler must maintain fixed prompts and provide `nfe_per_sample` metadata.

If a component has learned state, register it as parameters/buffers in an `nn.Module`
owned by the algorithm/module tree. Verify that optimizers, EMA where appropriate,
checkpoints, and device movement include it. Adaptive feedback also needs a clear
owner and distributed synchronization behavior. A `post_optimizer_step` constraint
hook runs after EMA updates and currently excludes that method from the restricted
FSDP path.

## Add configuration

Place a YAML option in the corresponding root `configs/` group. This is an
illustrative schedule registration, not an existing selectable implementation:

```yaml
# configs/noise_schedule/my_schedule.yaml
_target_: core.noise_schedules.my_schedule.MySchedule
scale: 1.0
```

Use explicit constructor arguments and interpolations for dependencies. There is
no central registration dictionary to edit: Hydra imports the selected `_target_`.
If multiple coordinated selections are needed, add an
[experiment recipe](adding-a-method.md). Reject unsupported combinations at the
closest useful construction boundary with an actionable error.

## Verify what changed

Choose tests that can catch mathematical or ownership errors:

- Analytic derivatives against autograd/finite differences, including endpoints.
- Transition normalization, support, limiting behavior, and prompt preservation.
- A controlled loss/gradient comparison when replacing existing mathematics.
- Checkpoint round trips for new learned state.
- One small train/backward/generate integration for affected compositions.
- DDP collectives and parameter handling when the new behavior is distributed.

Do not infer correctness from reproducing the implementation formula in a test.
Use an independent identity, reference, or invariant. Document supported selectors
and limitations in the relevant component page, then update validation coverage
with the actual evidence level.

Examples to inspect: [Gaussian paths](../../core/noise_schedules/gaussian_path.py),
[posterior uniform process](../../core/processes/uniform_posterior.py), and
[training lifecycle](../../core/training/module.py).

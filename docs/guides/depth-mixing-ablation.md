# Depth mixing on MDLM Sudoku

This experiment compares a standard transformer, DeepCrossAttention (DCA),
MUDD connections, and content-based per-position depth attention. All four use
canonical MDLM (absorbing corruption, constrained clean posterior, hazard-weighted
cross entropy), with no self-conditioning. Historical masked posterior-CE runs
are not reused as the control.

## Shared implementation

`DDiTBlock.forward_streams` consumes four inputs ordered Q, K, V, residual.
The baseline and all variants share `_forward`: normalization, time modulation,
RoPE, sequence attention, dropout, and MLP computation. Parameter layout for the
existing fused QKV projection is retained; independent source inputs apply its
three weight slices. `DepthMixingBlocks` owns the history and calls a router with
shape `[batch, position, source, feature]`, receiving `[batch, position, 4, feature]`.
No router mixes token positions. Ordinary sequence attention does that afterward.

| Router | History | How weights are generated | Attention residual |
| --- | --- | --- | --- |
| DCA | Input plus incremental attention+MLP contributions | Feature-specific static weights plus source-dependent ReLU gates | Q mixture |
| MUDD | Input plus complete block outputs | Latest-state RMSNorm and joint width-4S GELU MLP, plus static weights | Independent R mixture |
| Depth attention | Input plus complete block outputs | Four narrow query/key comparisons and softmax across source depth | Independent R mixture |

History semantics are a necessary difference, not just a change of activation.
DCA appends the direct attention+MLP contribution (without subtracting residuals).
Its final learned GRN reads the full contribution history. MUDD and depth attention
route after each block and use the final R output. Final unused Q/K/V streams
remain in the eager accounting, as in MUDD's joint-router pseudocode.

References: [DCA](https://arxiv.org/html/2502.06785v1),
[MUDDFormer](https://arxiv.org/html/2502.12170v1).

DCA follows the written GRN-v3 equation `b + ReLU(w^T history)`. The paper's
Figure 3 differs in the placement of normalization/ReLU. Its stated zero gate
initialization would permanently disable the dynamic term with PyTorch's ReLU
subgradient. The experiment therefore explicitly uses gate std 0.001. This is a
documented implementation choice, not an exact paper reproduction. Tests also
cover paper-zero initialization and its baseline equivalence.

MUDD initializes the dynamic output projection to zero and the static weights to
select the latest state. Its optional FFN-depth redistribution and PrePostDANorm
are omitted to isolate the connection mechanism. Depth attention uses key_dim=8,
raw hidden values, affine-free RMSNorm, and a zero-initialized per-stream gate
interpolating from the latest state to the retrieved mixture. Gates receive
first-step gradients; query/key parameters learn once the gates open.

## Compute and protocol

Eight layers, width512, eight heads, 180 tokens, GELU MLP. FLOP matching uses
PyTorch FlopCounterMode with explicit math attention on CPU, counting a real
forward and backward pass. Counts below are per sequence; normalization,
elementwise gating, softmax, and data movement are not fully counted. The target
is within 1% of counted training matrix FLOPs, not identical wall-clock time.

| Architecture | MLP width | Forward GFLOPs | Forward+backward GFLOPs | Parameters |
| --- | ---: | ---: | ---: | ---: |
| Baseline | 2048 | 9.593 | 28.779 | 25,257,741 |
| DCA | 2032 | 9.571 | 28.712 | 25,209,485 |
| MUDD | 2032 | 9.612 | 28.837 | 25,225,469 |
| Depth attention | 1936 | 9.602 | 28.807 | 24,601,517 |

Thus this is compute-matched, not exactly parameter-matched. Extra activation
storage and separate Q/K/V projections can increase latency despite similar FLOPs.
Compiled dead-code elimination may remove unused final streams.

The sweep uses one seed, 20k optimizer updates, global batch256 (microbatch64 with
four accumulation steps), learning rate3e-4, 2500-step warmup, EMA0.9999, BF16,
and torch.compile. The existing cached 48k/2k Sudoku Hard dataset is reused.
Generation callbacks run every1000 updates; validation loss every4000.

All variants use the existing nested validation-only tuner: native ancestral
(16 temperature trials), native greedy clean estimate (two terminal choices),
and EB greedy (16 temperature/entropy-bound trials). Native policies use128 NFE;
EB has a fixed cap within128 and can finish earlier. Tuning and test puzzle
indices and generation seeds are disjoint and inherited from the existing
Sudoku protocol. Only the validation-selected winner receives final evaluation.
Training loss/periodic solve rates remain available for fixed-step comparison.

## Configuration

The implementation is selected by `configs/depth_mixing/{baseline,dca,mudd,attention}.yaml`.
The MLP widths there are calibrated for this 8x512 experiment; recalibrate them
when changing backbone size. The routers themselves accept arbitrary dimensions.

```bash
uv run --locked --extra tuning --extra scoring --extra tracking \
  python -m core.sweeps.prepare -m +sweep=mdlm_depth_mixing_sudoku
uv run --locked --extra tuning --extra scoring --extra tracking \
  python -m core.sweeps.prepare collect \
  /home/patrick/.cache/duo/plans/mdlm-depth-mixing-sudoku-20k-20261031
```

Plans, frozen source, FLOP evidence, and launch metadata live under that plan
folder. Results/checkpoints/Optuna databases live under
`/home/patrick/.cache/duo/nested-tuning/mdlm-depth-mixing-sudoku-20k-20261031`.
W&B project: `rz-text-dlp/mdlm-depth-mixing-sudoku-20k-20261031`.
The dedicated Ray driver uses GPUs6,7, two concurrent cases. No existing sweep
is stopped or modified. See launch.json for actual submission state.

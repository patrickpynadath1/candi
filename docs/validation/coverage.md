# Validation coverage

Implementation, successful execution, and reproduction are different claims.
**Implemented** means a maintained code/config path exists. **Execution tested**
means that a specified fixture or integration ran. **Reproduced** means a
comparison against a defined reference succeeded. A short pilot is evidence
that a method can learn, not proof of convergence or paper-level performance.

The latest reported regression suite before this documentation consolidation
passed **402 tests**. That total is a snapshot, not a promise that every method,
task, precision, sampler, modifier and distributed strategy combination has
been tested. Run the current suite from the repository root:

```bash
uv run python -m pytest core/tests -q
```

GPU and real sandbox checks require those facilities; skipped tests do not
establish their behavior on the current machine.

## Methods and tasks

| Area | Evidence | Boundary of the claim |
| --- | --- | --- |
| MDLM / Sudoku | Exact saved-checkpoint generation parity for ancestral and EB over 2,000 puzzles; controlled loss and gradients identical | No fresh 20,000-step convergence reproduction |
| Gaussian one-hot | 2,000-step Sudoku pilot; Euler/Heun/ODE evaluations; exact before/after representation-extraction regression | Sampler intentionally differs from original continuous implementation |
| Gaussian embeddings | Three-step CUDA/BF16 train/checkpoint smoke; all three continuous solvers; sphere-rank reference equality | No trained embedding-model quality benchmark |
| Uniform, posterior-uniform, SEDD, DUO, D3PM | Process/objective/config/sampling regression tests | No full-budget pretrained quality reproduction claimed |
| Autoregressive / permuted AR | Causal visibility, KV-cache equivalence and execution tests; permuted-AR 10k-step Text8 pilot | Ordinary AR quality not established; Text8 pilot is a small model |
| Spherical flow matching | CUDA geometry/adaptation tests and 2k-step Sudoku run | Early run remains inside 2,500-step warmup, not 20k-step reproduction |
| Simplex | Copied numerical primitives and nine bridge combinations match source exactly; CUDA training/sampling cases | No standalone trained Sudoku quality claim; source paper protocol differs |
| RePlaid | Corrected reference output/loss/gradient audits; training and sampling checks; corrected 2k-step pilot | Successful reference used 50k steps and different data; not reproduced here |
| Blurring | Process/schedule/model/sampling and block-boundary tests | No full-budget quality reproduction recorded |
| CANDI hybrid | Close imported-checkpoint result plus fresh standard-mask 2k-step run | Standard masking intentionally changes training corruption |
| Uniform + simplex hybrid | Full 2k-step training and 2,000-puzzle evaluation | Single pilot, no tuning or convergence claim |
| Sudoku task | Full checkpoint parity; answer sanitization, metric/dataclass parity | Puzzle prompt preservation and answer clue consistency are separate |
| Text8 task | All 35 token IDs and 25 controlled packed windows match original; real data smoke and generation metrics | Multi-worker legacy packing boundaries are not reproduced generally |
| OpenWebText task | Controlled metric math parity, training fixture, cached scorer and real pretrained MDLM point | BOS boundary bug intentionally fixed; no full training reproduction |
| TinyGSM task | Fixture token/mask parity, two-step training/evaluation, real Bubblewrap checks | Finite data only; no full training or accuracy reproduction |

See [measured results](reproduction-results.md) for exact counts, budgets and
weight selection. Reusable components and method settings are documented in
[architecture](../concepts/architecture.md) and [methods](../index.md).

## Modifiers and infrastructure

Self-conditioning covers posterior-mean and hidden sources, detached bootstrap,
zero-initialized feedback, checkpoint behavior, and sampler sessions. The MDLM
hidden-feedback pilot is recorded separately from earlier superseded feedback
designs. Adaptive ODE sampling rejects self-conditioning because rejected-stage
feedback rollback is not implemented. Generic posterior-mean feedback requires
probability outputs; that is checked at construction/use boundaries.

Block diffusion has attention, masks, loss-routing, prefix-cache and combination
tests. It does not imply every sampler or model can consume the modifier.
Consult [block diffusion](../modifiers/block-diffusion.md) for supported paths.

| Infrastructure | Executed check | Remaining qualification |
| --- | --- | --- |
| DDP | Two GPUs, four MDLM steps, cosine scheduler, EMA, callback generation, checkpoint | Not a matrix of all algorithms/modifiers |
| Restricted FSDP | Two GPUs, two steps, resume to four, standalone checkpoint evaluation | EMA, callback generation, norm clipping and parameter-projection hooks unsupported |
| Native resume | Optimizer, scheduler, EMA and RNG restoration | Exact mid-epoch shuffled/prefetched sample replay not guaranteed |
| LR schedules | Constant, constant-warmup, halving and cosine tests | Cosine is local implementation; missing legacy class not a parity reference |
| Frontier sweeps | Cartesian-grid/report/Pareto/resume tests and one pretrained OpenWebText point | No completed multi-point real-model frontier claimed |
| Self-containment | Isolated-tree entrypoint/config and targeted tests, then root runner/config relocation | External Python dependencies, datasets, weights and scorers still required |

## Explicit omissions

Maze and Zebra tasks, legacy hub-path workflows, latent flow matching,
directly predicted continuous velocity, RePlaid hybrids, and simplex teacher/
student distillation are not included. Learned contextual or compressing
encoders are not implemented by the tokenwise representation interface.
Legacy optimizer-state import and bitwise interrupted-run data replay are also
outside the current guarantees. Removing legacy source did not turn these
omissions into implemented functionality.

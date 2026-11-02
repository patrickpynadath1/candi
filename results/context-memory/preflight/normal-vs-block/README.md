# Normal masked diffusion versus block diffusion

RTX A6000, batch8, BF16, eight512-wide layers, seed1, AdamW. Both compile transformer layers;5 warmups and30 measured forward/backward/optimizer steps. Compilation, EMA, data loading, validation and gradient accumulation are excluded. Synthetic inputs; these are execution measurements, not quality comparisons. Peak allocated memory includes parameters, gradients and optimizer state; allocator reserved peaks are in the raw JSONs.

Normal uses the same MDLM algorithm with block_diffusion.enabled=false. Block uses block size32, clean_loss=none, x0_causal=false, no context accumulator. Config pairs differ only in enabled; parameter counts match. Each task runs paired configurations sequentially on the same GPU; tasks run concurrently on GPUs6/7. Short runs cannot establish small timing differences.

| Task | Checkpoint | Normal MB | Block MB | Memory ratio | Normal ms | Block ms | Time ratio |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| text8 | False | 745 | 1154 | 1.55 | 39.8 | 40.4 | 1.02 |
| text8 | True | 534 | 548 | 1.03 | 54.4 | 53.5 | 0.98 |
| tinystories | False | 1846 | 3884 | 2.10 | 38.8 | 110.1 | 2.84 |
| tinystories | True | 680 | 1118 | 1.64 | 57.7 | 139.5 | 2.42 |

The block implementation concatenates clean/noisy hidden sequences (2L transformer positions), applies a block visibility mask, and currently projects both streams to logits even though clean loss is disabled. Ordinary diffusion processes L noisy positions with full attention. Peak memory is not simply proportional to positions because persistent state and checkpointing also matter. These ratios describe this implementation and workload, not a universal block-diffusion cost.

Reproduce with scripts/benchmark_training_memory.py configs/<task>-*.yaml --compile-target transformer-layer --batch-size 8 --warmup 5 --steps 30 --output <fresh-output>. Use uv and the documented environment. Training campaigns were not launched or changed.

## Operation attribution

Additional compiled/checkpointed TinyStories profiles identify FlashAttention
for normal diffusion, and aten::_efficient_attention_* (CUTLASS memory-efficient
attention) for block diffusion's explicit visibility mask. Forward including
checkpoint replay:2.67ms vs21.52ms; backward:4.22ms vs44.29ms. Thus attention
accounts for6.89ms vs65.81ms of these profiles. These are instrumented operator
times, not an independent end-to-end benchmark or exact additive decomposition
of the earlier wall times. Profiles used separate A6000s (normalGPU6, blockGPU7);
GPU6 had another process resident, so fine-grained ratios are provisional.
Kernel selection is directly observed. Raw tables: normal-operations.txt and
block-operations.txt.

Both versions have identical parameter state. Block diffusion adds clean-stream
QKV/MLP/norm activations and associated backward/replay, plus mask/backend
buffers. It also projects clean logits despite no clean loss. We have not
attributed every byte of the438MB checkpointed peak difference; do not label it
all as KV cache or as the attention mask. The deep accumulator implementation
still computes a full clean transformer stream. Getting close to ordinary
diffusion compute therefore requires either a cheaper clean encoder or a
different source of context writes, besides optimizing masked attention.

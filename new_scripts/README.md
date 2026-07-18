# text8 tiny-DiT verification

`run_all.sh` trains all nine algorithms for 20,000 optimizer steps on GPUs
5, 6, and 7, then evaluates 128 samples at each configured temperature/NFE
point and writes the diversity/coherence frontiers under `new_results/`.

The shared settings live in `configs/config_text8_frontier.yaml`. Training
checkpoints and logs are written to `new_runs/`. The two reported text8 metrics
come directly from `text_metrics.py`: `total_unique_words` (diversity) and
`total_valid_words` (coherence).

AR and permutation-trained AR require one forward pass per generated position,
so they are evaluated at their natural 128 NFE only. Other methods use NFE
16, 32, 64, and 128. Temperatures are 0.7, 0.85, 1.0, 1.15, and 1.3.

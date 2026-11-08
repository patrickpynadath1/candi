# E2D2 encoder/decoder scaling

The encoder/decoder depth study is the worked example
[block-encoder-decoder](../../experiments/block-encoder-decoder/README.md): E2D2 with
the encoder or the decoder at 1, 2, 4 and 8 layers, a no-encoder (raw context) control
and one shared 8-layer block transformer, on text8, TinyStories BPE-10k and Sudoku-hard.
It is prepared with `+sweep=block_encoder_decoder`; the block models are the options
of `configs/block/`. See [encoder/decoder wrappers](encoder-decoder.md) for how the
models work.

# Softmax Feedback CANDI

`candi_softmax_looped` uses the same training and sampling procedure as
[hidden-state looping](candi_looped.md), but carries token probabilities:

```text
p_previous = stop_gradient(softmax(previous_raw_logits))
feedback = p_previous @ input_embedding_table
hidden_input = input_projection(preconditioned_current_embeds + feedback)
```

The softmax excludes the reserved mask class and uses temperature 1, before
CANDI's output temperature and revealed-token clamping. The input embedding
table is the same learned table used for clean/noisy inputs, not the untied
output classifier table. There is no feedback normalization, extra projection,
or additional learned parameter. The shared input projection handles different
input-embedding and hidden dimensions and applies its bias only once. The
feedback is added after CANDI's normal input preconditioning, so it is not
attenuated by the continuous-noise scale.

Training runs the full batch with no feedback, then recomputes a random half
using detached first-pass probabilities. Only each example's final prediction
contributes to the loss. Gradients through the feedback lookup update the input
embedding table, but do not flow into the previous prediction.

Generation makes one forward pass per timestep, feeding the preceding step's
probabilities into the next. State resets between sampling requests. The initial
step has zero feedback. Standalone validation loss retains two passes, as for
the other looped variants. Carry size is `[batch, length, data_vocab_size]`;
for large vocabularies this can exceed hidden-state carry size.

The hard-Sudoku preset selects EB=0.1 and greedy token choices:

```bash
bash new_scripts/train_sudoku_looped_one.sh 3 candi_softmax_looped \
  /home/patrick/duo/new_sudoku_runs/looping_hard_20k_seed1/candi_softmax_looped \
  callbacks.sudoku_eval.num_steps=180
```

Use `new_scripts/eval_sudoku_looped_eb_greedy.sh` for checkpoint evaluation.
This is a separately trained variant; the full-hidden/register checkpoints
were trained with different feedback and should not be treated as its weights.

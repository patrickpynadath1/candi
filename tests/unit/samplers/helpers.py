from types import SimpleNamespace

import torch


def config(*, predictor="ancestral", steps=2, noise_removal="none"):
    return SimpleNamespace(
        sampling=SimpleNamespace(
            predictor=predictor,
            steps=steps,
            noise_removal=noise_removal,
            use_float64=False,
        ),
        algo=None,
    )


class ModelBase:
    device = torch.device("cpu")
    dtype = torch.float32
    num_tokens = 3
    vocab_size = 4
    token_offset = False
    tokenizer = SimpleNamespace(bos_token_id=1, eos_token_id=3)
    config = config()


def onehot_prior(num_samples, num_tokens, vocab_size=4):
    tokens = torch.zeros(num_samples, num_tokens, dtype=torch.long)
    return torch.nn.functional.one_hot(tokens, vocab_size).float()

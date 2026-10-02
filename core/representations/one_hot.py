from torch import nn
from torch.nn import functional as F

from .spec import RepresentationSpec


class OneHotRepresentation(nn.Module):
    def __init__(self, vocab_size, dimension=None):
        super().__init__()
        if vocab_size < 2:
            raise ValueError("vocab_size must be at least two")
        if dimension is not None and dimension != vocab_size:
            raise ValueError("one-hot dimension must equal vocabulary size")
        self.spec = RepresentationSpec("one_hot", vocab_size, vocab_size, True)

    def encode(self, tokens):
        return F.one_hot(tokens, self.spec.vocab_size).float()

    def posterior_mean(self, probabilities):
        return probabilities

    def decode(self, state):
        return state.argmax(-1)

"""Token-ID state without an independently learned input table."""
from torch import nn
from .spec import RepresentationSpec


class TokenRepresentation(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.spec = RepresentationSpec("tokens", vocab_size, vocab_size, False)

    def encode(self, tokens):
        return tokens

    clean_state = encode

    def decode(self, state):
        return state

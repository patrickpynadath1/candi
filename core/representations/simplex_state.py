"""Log-simplex states of a single-state simplex network, without parameters.

The standalone simplex recipe (core/algorithms/simplex_diffusion.py) holds its
view on the algorithm, while the network's ``SimplexInput`` slot owns the token
table and embeds the state (expectation or argmax). The view's representation
therefore only maps clean tokens to states; it registers no tensors, so the recipe's
parameters and checkpoint keys stay those of the network.
"""

import torch
from torch import nn
from torch.nn import functional as F


class SimplexStateRepresentation(nn.Module):
    def __init__(self, vocab_size, state_dtype=torch.float32):
        super().__init__()
        self.vocab_size, self.state_dtype = int(vocab_size), state_dtype

    def encode(self, tokens):
        """Tokens: the Dirichlet process corrupts token ids directly."""
        return tokens

    def clean_state(self, tokens):
        """Log one-hot clean state, as ``DirichletProcess.encode``."""
        return F.one_hot(tokens, self.vocab_size).to(self.state_dtype).log()

import torch
import torch.nn as nn


class VQ_GaussResid(nn.Module):
    def __init__(self, num_codes, dim, residual_compression_rate=None):
        super().__init__()
        self.num_codes = num_codes
        self.codebook = nn.Embedding(num_codes, dim)
        self.residual_compression = residual_compression_rate

    def forward(self, x):
        probs = torch.einsum("bld, nd -> bln", x, self.codebook.weight)
        x_argmaxed = torch.argmax(probs, dim=-1)
        residual = x - self.codebook(x_argmaxed)
        if self.residual_compression is not None:
            residual = residual.reshape(residual.shape[0], 
                                        residual.shape[1] // self.residual_compression, 
                                        self.residual_compression, residual.shape[-1]).mean(dim=2)
        

        return self.codebook(x_argmaxed), residual
    
class VQ_STE(nn.Module):
    def __init__(self, num_codes, dim, residual_compression_rate=None):
        super().__init__()
        self.num_codes = num_codes
        self.codebook = nn.Embedding(num_codes, dim)
        self.residual_compression = residual_compression_rate

    def forward(self, x):
        probs = torch.einsum("bld, nd -> bln", x, self.codebook.weight)
        # expected_x = torch.einsum("bln, nd -> bld", probs.softmax(dim=-1), self.codebook.weight)
        x_argmaxed = self.codebook(torch.argmax(probs, dim=-1))

        # STE
        x_argmaxed = x_argmaxed + (x - x.detach())
        

        return x_argmaxed
    

class VQ_MixtureResidual(nn.Module):
    def __init__(self, num_codes, dim, residual_compression_rate=None):
        super().__init__()
        self.num_codes = num_codes
        self.codebook = nn.Embedding(num_codes, dim)
        self.residual_compression = residual_compression_rate



    def forward(self, x):
        probs = torch.einsum("bld, nd -> bln", x, self.codebook.weight).softmax(dim=-1)
        mixture = torch.einsum("bln, nd -> bld", probs, self.codebook.weight)
        x_argmaxed = torch.argmax(probs, dim=-1)
        residual = mixture - self.codebook(x_argmaxed)
        if self.residual_compression is not None:
            residual = residual.reshape(residual.shape[0], 
                                        residual.shape[1] // self.residual_compression, 
                                        self.residual_compression, residual.shape[-1]).mean(dim=2)
        

        return self.codebook(x_argmaxed), residual

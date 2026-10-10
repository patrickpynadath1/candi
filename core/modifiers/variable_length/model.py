"""Gap heads over the same backbone features used for content prediction."""

from dataclasses import dataclass
import torch
from torch import nn
from torch.nn import functional as F
from core.models.prediction import ModelPrediction


@dataclass
class SequencePrediction:
    tokens: torch.Tensor
    expected_counts: torch.Tensor
    gap_tokens: torch.Tensor
    feedback: torch.Tensor | None
    nfe: int = 1


class GapHead(nn.Module):
    def __init__(self, hidden_size, vocab_size):
        super().__init__()
        self.left_boundary = nn.Parameter(torch.randn(hidden_size) / hidden_size**0.5)
        self.right_boundary = nn.Parameter(torch.randn(hidden_size) / hidden_size**0.5)
        self.projection = nn.Sequential(
            nn.Linear(2 * hidden_size, hidden_size), nn.SiLU()
        )
        self.count = nn.Linear(hidden_size, 1)
        self.content = nn.Linear(hidden_size, vocab_size)
        self.vocab_size = int(vocab_size)

    def forward(self, hidden, valid):
        # hidden includes a learned left boundary, so even empty strings have a
        # contextual gap prediction. True clean positions never enter the model.
        right = torch.cat(
            [hidden[:, 1:], self.right_boundary.expand(len(hidden), 1, -1)], 1
        )
        right_valid = F.pad(valid, (0, 1), value=False)
        right = torch.where(
            right_valid[..., None], right, self.right_boundary.to(right)
        )
        gaps = self.projection(torch.cat([hidden, right], -1))
        return F.softplus(self.count(gaps).squeeze(-1).float()) + 1e-6, self.content(
            gaps
        )


def forward_features(model, head, states, points, time, valid, conditioning=None):
    block_forward = getattr(model, "forward_sequence_features", None)
    if block_forward is not None:
        blocked = block_forward(
            head, states, points, time, valid, conditioning=conditioning
        )
        if blocked is not None:
            return blocked
    hidden, condition = model.prepare_fused(
        states, points, time, self_condition=conditioning
    )
    boundary = head.left_boundary.to(hidden).expand(len(hidden), 1, -1)
    hidden = torch.cat([boundary, hidden], 1)
    keys = F.pad(valid, (1, 0), value=True)
    output = model.forward_hidden(
        hidden, condition, return_features=True, attention_mask=keys[:, None, None, :]
    )
    expected, gap_logits = head(output.hidden, valid)
    tokens = ModelPrediction(
        output.logits[:, 1:, : head.vocab_size], output.hidden[:, 1:]
    )
    return tokens, expected, gap_logits

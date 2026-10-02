import torch


class WeightedCrossEntropy:
    def __call__(self, log_probs, targets, keep_rate, keep_derivative):
        if torch.any(keep_rate >= 1):
            raise ValueError("MDLM loss requires a noisy time, not the clean endpoint")
        ce = -log_probs.gather(-1, targets[..., None]).squeeze(-1)
        weight = -keep_derivative / (1 - keep_rate)
        return ce * weight[:, None]

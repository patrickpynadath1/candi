import torch
from torch import nn, FloatTensor


class EncNormalizer(nn.Module):
    def __init__(self, cfg_or_path=None):
        super().__init__()

        if cfg_or_path is not None and hasattr(cfg_or_path, "embedding"):
            dim = cfg_or_path.embedding.dim
        else:
            dim = None

        if dim is None:
            self.register_buffer("enc_mean", torch.empty(0), persistent=True)
            self.register_buffer("enc_std", torch.empty(0), persistent=True)
        else:
            self.register_buffer("enc_mean", torch.zeros(1, 1, dim), persistent=True)
            self.register_buffer("enc_std", torch.ones(1, 1, dim), persistent=True)
        self.register_buffer("stats_initialized", torch.tensor(False), persistent=True)

        if cfg_or_path is not None and isinstance(cfg_or_path, str):
            state_dict = torch.load(cfg_or_path, map_location="cpu")
            self.set_stats(state_dict["enc_mean"], state_dict["enc_std"])

    def set_stats(self, enc_mean: FloatTensor, enc_std: FloatTensor):
        enc_mean = enc_mean.detach().float().view(1, 1, -1)
        enc_std = enc_std.detach().float().clamp_min(1e-4).view(1, 1, -1)

        if self.enc_mean.numel() == 0:
            self.enc_mean = enc_mean
            self.enc_std = enc_std
        else:
            self.enc_mean.copy_(enc_mean)
            self.enc_std.copy_(enc_std)
        self.stats_initialized.fill_(True)

    @property
    def is_initialized(self) -> bool:
        return bool(self.stats_initialized.item())

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        stats_key = f"{prefix}stats_initialized"
        mean_key = f"{prefix}enc_mean"
        std_key = f"{prefix}enc_std"

        if stats_key not in state_dict:
            has_stats = (
                mean_key in state_dict
                and std_key in state_dict
                and state_dict[mean_key].numel() > 0
                and state_dict[std_key].numel() > 0
            )
            state_dict[stats_key] = torch.tensor(has_stats)

        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def forward(self, *args, **kwargs):
        encoding = args[0]
        if not self.is_initialized:
            return encoding
        return self.normalize(encoding)

    def normalize(self, encoding: FloatTensor) -> FloatTensor:
        if not self.is_initialized:
            return encoding
        return (encoding - self.enc_mean) / self.enc_std

    def denormalize(self, pred_x_0: FloatTensor) -> FloatTensor:
        if not self.is_initialized:
            return pred_x_0
        return pred_x_0 * self.enc_std + self.enc_mean

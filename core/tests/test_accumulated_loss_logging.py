"""Training logs average the optimizer batch, including partial final batches."""

import torch
import lightning as L
from core.tests.test_mdlm import config
from core.training.module import TrainingModule


class KnownLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.0))

    def loss(self, tokens, mask, **kwargs):
        value = tokens.float().mean() + self.weight * 0
        return {"loss": value, "num_tokens": torch.tensor(float(tokens.numel()))}


def test_logs_accumulated_mean_and_partial_batch():
    module = TrainingModule(config("training.ema=0"))
    module.algorithm = KnownLoss()
    observed = []
    module.log = lambda name, value, **kw: observed.append((name, float(value)))
    dataset = [
        {"input_ids": torch.tensor([value]), "loss_mask": torch.ones(1)}
        for value in range(1, 6)
    ]
    trainer = L.Trainer(
        accelerator="cpu",
        max_steps=3,
        accumulate_grad_batches=2,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        num_sanity_val_steps=0,
    )
    trainer.fit(
        module, train_dataloaders=torch.utils.data.DataLoader(dataset, batch_size=1)
    )
    assert [value for name, value in observed if name == "train/loss"] == [
        1.5,
        3.5,
        5.0,
    ]

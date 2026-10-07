"""Epoch remainders must retain full time coverage and gradient magnitude."""

import lightning as L
import pytest
import torch
from torch.utils.data import DataLoader

from core.tests.test_mdlm import config
from core.time_sampling.uniform import UniformTimeSampler
from core.training.module import TrainingModule


class Probe(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(()))
        self.times = []

    def loss(self, input_ids, loss_mask, **kwargs):
        kwargs.pop("corruption_mask")
        kwargs.pop("step")
        kwargs.pop("training")
        self.times.append(UniformTimeSampler().sample(len(input_ids), "cpu", **kwargs))
        return {"loss": self.weight, "num_tokens": loss_mask.sum()}


class ProbeModule(TrainingModule):
    def __init__(self):
        super().__init__(config("training.ema=0", "loader.global_batch_size=8"))
        self.algorithm = Probe()
        self.gradients = []
        self.losses = []

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0)

    def on_before_optimizer_step(self, optimizer):
        self.gradients.append(self.algorithm.weight.grad.item())
        total, count = self._train_metric_totals["loss"]
        self.losses.append((total / count).item())
        super().on_before_optimizer_step(optimizer)


@pytest.mark.parametrize("microbatches", [2, 8, 10, 11])
def test_epoch_remainder_time_coverage_and_gradient(microbatches):
    model = ProbeModule()
    data = [
        {"input_ids": torch.zeros(1, dtype=torch.long), "loss_mask": torch.ones(1)}
        for _ in range(microbatches * 2)
    ]
    trainer = L.Trainer(
        accelerator="cpu",
        devices=1,
        max_epochs=2,
        accumulate_grad_batches=4,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        limit_val_batches=0,
    )
    trainer.fit(model, train_dataloaders=DataLoader(data, batch_size=2))
    assert model.gradients == pytest.approx([1.0] * (2 * ((microbatches + 3) // 4)))
    assert model.losses == pytest.approx([1.0] * len(model.gradients))
    for epoch in range(2):
        times = model.algorithm.times[epoch * microbatches : (epoch + 1) * microbatches]
        for start in range(0, microbatches, 4):
            window = torch.cat(times[start : start + 4])
            strata = ((window - 0.001) / 0.999 * len(window)).floor().long()
            assert torch.equal(strata.sort().values, torch.arange(len(window)))

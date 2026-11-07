import lightning as L
import torch
from torch.utils.data import DataLoader, DistributedSampler


class WholeStepSampler(DistributedSampler):
    """Shuffle each epoch, shard by rank, and keep only whole optimizer steps.

    Lightning steps the optimizer on a trailing partial accumulation window.
    That window would see only the first stratified time partition, so drop it.
    A fresh permutation per epoch means no example is permanently excluded.
    Subclassing DistributedSampler stops Lightning from replacing it under DDP.
    """

    def __init__(self, dataset, *, num_replicas, rank, step_examples, seed):
        super().__init__(
            dataset, num_replicas=num_replicas, rank=rank, shuffle=True, seed=seed
        )
        per_rank = len(dataset) // num_replicas
        self.num_samples = per_rank - per_rank % step_examples
        self.total_size = self.num_samples * num_replicas
        if self.num_samples == 0:
            raise ValueError("training set is smaller than one optimizer step")

    def __iter__(self):
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        order = torch.randperm(len(self.dataset), generator=generator)
        order = order[: self.total_size]
        return iter(order[self.rank : self.total_size : self.num_replicas].tolist())


class TaskDataModule(L.LightningDataModule):
    def __init__(self, task, loader_config, accumulation_steps, seed=0):
        super().__init__()
        self.task, self.loader_config = task, loader_config
        self.accumulation_steps, self.seed = accumulation_steps, seed

    def prepare_data(self):
        self.task.prepare_data()

    def setup(self, stage=None):
        if stage in (None, "fit"):
            self.train_dataset = self.task.training_dataset()
        self.valid_dataset = self.task.validation_dataset()

    def train_dataloader(self):
        partitions = self.trainer.world_size * self.accumulation_steps
        global_batch = self.loader_config.global_batch_size
        if global_batch % partitions:
            raise ValueError(
                "global batch must be divisible by world size times accumulation"
            )
        local = global_batch // partitions
        sampler = WholeStepSampler(
            self.train_dataset,
            num_replicas=self.trainer.world_size,
            rank=self.trainer.global_rank,
            step_examples=local * self.accumulation_steps,
            seed=self.seed,
        )
        return DataLoader(
            self.train_dataset,
            batch_size=local,
            sampler=sampler,
            drop_last=True,
            num_workers=self.loader_config.num_workers,
            pin_memory=self.loader_config.pin_memory,
            collate_fn=self.task.collate_training,
        )

    def val_dataloader(self):
        return DataLoader(
            self.valid_dataset,
            batch_size=self.loader_config.eval_batch_size,
            shuffle=False,
            num_workers=self.loader_config.num_workers,
            pin_memory=self.loader_config.pin_memory,
            collate_fn=self.task.collate_training,
        )

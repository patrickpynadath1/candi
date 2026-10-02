import lightning as L
from torch.utils.data import DataLoader


class TaskDataModule(L.LightningDataModule):
    def __init__(self, task, loader_config, accumulation_steps):
        super().__init__()
        self.task, self.loader_config = task, loader_config
        self.accumulation_steps = accumulation_steps

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
        if len(self.train_dataset) < local * self.trainer.world_size:
            raise ValueError("training set is smaller than one distributed microbatch")
        return DataLoader(
            self.train_dataset,
            batch_size=local,
            shuffle=True,
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

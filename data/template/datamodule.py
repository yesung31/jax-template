import warnings

import grain.python as grain
import numpy as np

from core.data import DataModule


class ArraySource(grain.RandomAccessDataSource):
    def __init__(self, features, labels):
        self._features = features
        self._labels = labels

    def __len__(self):
        return len(self._features)

    def __getitem__(self, idx):
        return {"input": self._features[idx], "label": self._labels[idx]}


class TemplateDataModule(DataModule):
    def __init__(self, batch_size=32, num_workers=0, seed=0, **kwargs):
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed
        warnings.warn(
            "TemplateDataModule is using dummy random data. "
            "Replace this with your actual data loading logic.",
            UserWarning,
        )

    def prepare_data(self):
        # Download data, etc.
        pass

    def setup(self, stage=None):
        # Create dummy data: 100 samples, 32 dimensions
        self.train_source = ArraySource(
            np.random.randn(100, 32).astype(np.float32),
            np.random.randint(0, 2, (100,)).astype(np.int32),
        )
        self.val_source = ArraySource(
            np.random.randn(20, 32).astype(np.float32),
            np.random.randint(0, 2, (20,)).astype(np.int32),
        )
        self.test_source = ArraySource(
            np.random.randn(20, 32).astype(np.float32),
            np.random.randint(0, 2, (20,)).astype(np.int32),
        )

    def train_dataloader(self):
        return (
            grain.DataLoader(
                data_source=self.train_source,
                sampler=grain.IndexSampler(
                    num_records=len(self.train_source),
                    shuffle=True,
                    seed=self.seed,
                    shard_options=grain.NoSharding(),
                ),
                worker_count=self.num_workers,
                operations=[grain.Batch(batch_size=self.batch_size, drop_remainder=True)],
            )
        )

    def val_dataloader(self):
        return (
            grain.DataLoader(
                data_source=self.val_source,
                sampler=grain.IndexSampler(
                    num_records=len(self.val_source),
                    shuffle=False,
                    seed=self.seed,
                    shard_options=grain.NoSharding(),
                ),
                worker_count=self.num_workers,
                operations=[grain.Batch(batch_size=self.batch_size, drop_remainder=False)],
            )
        )

    def test_dataloader(self):
        return (
            grain.DataLoader(
                data_source=self.test_source,
                sampler=grain.IndexSampler(
                    num_records=len(self.test_source),
                    shuffle=False,
                    seed=self.seed,
                    shard_options=grain.NoSharding(),
                ),
                worker_count=self.num_workers,
                operations=[grain.Batch(batch_size=self.batch_size, drop_remainder=False)],
            )
        )

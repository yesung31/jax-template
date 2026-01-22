import warnings

import numpy as np

from core.dataloader import JAXDataLoader, NumpyDataset


class TemplateDataModule:
    def __init__(self, batch_size=32, num_workers=0, **kwargs):
        self.batch_size = batch_size
        self.num_workers = (
            num_workers  # Not used in simple JAX loader but kept for config compatibility
        )
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
        # JAX/Flax often uses numpy for loading, then converts to device array inside step
        self.train_dataset = NumpyDataset(
            np.random.randn(100, 32).astype(np.float32),
            np.random.randint(0, 2, (100,)).astype(np.int32),
        )
        self.val_dataset = NumpyDataset(
            np.random.randn(20, 32).astype(np.float32),
            np.random.randint(0, 2, (20,)).astype(np.int32),
        )
        self.test_dataset = NumpyDataset(
            np.random.randn(20, 32).astype(np.float32),
            np.random.randint(0, 2, (20,)).astype(np.int32),
        )

    def train_dataloader(self):
        return JAXDataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True
        )

    def val_dataloader(self):
        return JAXDataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)

    def test_dataloader(self):
        return JAXDataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)

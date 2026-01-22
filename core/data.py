import math
from abc import ABC, abstractmethod
from typing import Any, Optional

import numpy as np


class Dataset(ABC):
    """Abstract base class for datasets."""

    @abstractmethod
    def __len__(self) -> int:
        pass

    @abstractmethod
    def __getitem__(self, idx: Any) -> Any:
        pass


class DataModule(ABC):
    """Abstract base class for data modules."""

    def __init__(self, **kwargs):
        pass

    def prepare_data(self):
        """Use this to download data or save to disk.
        This is called only once (usually) before setup.
        """
        pass

    @abstractmethod
    def setup(self, stage: str | None = None):
        """Load data, split, and perform transformations.
        This is called on every device (in distributed setting) or once per run.
        """
        pass

    @abstractmethod
    def train_dataloader(self):
        """Returns the training dataloader."""
        pass

    @abstractmethod
    def val_dataloader(self):
        """Returns the validation dataloader."""
        pass

    @abstractmethod
    def test_dataloader(self):
        """Returns the test dataloader."""
        pass


class DataLoader:
    def __init__(self, dataset, batch_size, shuffle=False, drop_last=False, seed=0):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.rng = np.random.default_rng(seed)

    def __iter__(self):
        indices = np.arange(len(self.dataset))
        if self.shuffle:
            self.rng.shuffle(indices)

        for i in range(0, len(indices), self.batch_size):
            batch_indices = indices[i : i + self.batch_size]

            if self.drop_last and len(batch_indices) < self.batch_size:
                continue

            # Fetch data
            batch = self.dataset[batch_indices]
            yield batch

    def __len__(self):
        if self.drop_last:
            return len(self.dataset) // self.batch_size
        else:
            return math.ceil(len(self.dataset) / self.batch_size)

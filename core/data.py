import math
from abc import ABC, abstractmethod


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

    @property
    def train_steps(self):
        return len(self.train_source) // self.batch_size

    @property
    def val_steps(self):
        return math.ceil(len(self.val_source) / self.batch_size)

    @property
    def test_steps(self):
        return math.ceil(len(self.test_source) / self.batch_size)

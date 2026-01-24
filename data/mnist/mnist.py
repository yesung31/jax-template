import os
import struct
from pathlib import Path

import grain.python as grain
import numpy as np

from core.data import DataModule


class MNISTDataSource(grain.RandomAccessDataSource):
    def __init__(self, images, labels):
        self._images = images
        self._labels = labels

    def __len__(self):
        return len(self._images)

    def __getitem__(self, idx):
        # Normalize images to [0, 1] and add channel dimension
        img = self._images[idx].astype(np.float32) / 255.0
        img = img[..., np.newaxis]  # (28, 28, 1)
        lbl = self._labels[idx].astype(np.int32)
        return {"input": img, "label": lbl}


def load_mnist_images(path):
    with open(path, "rb") as f:
        magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
        images = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, rows, cols)
    return images


def load_mnist_labels(path):
    with open(path, "rb") as f:
        magic, num = struct.unpack(">II", f.read(8))
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    return labels


class MNIST(DataModule):
    def __init__(self, data_dir="~/Datasets/MNIST", batch_size=32, num_workers=0, seed=0, **kwargs):
        self.data_dir = Path(os.path.expanduser(data_dir))
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed

    def setup(self, stage=None):
        raw_dir = self.data_dir / "raw"

        # Check if files exist, if not try .gz
        if (raw_dir / "train-images-idx3-ubyte").exists():
            train_images = load_mnist_images(raw_dir / "train-images-idx3-ubyte")
        else:
            train_images = load_mnist_images(raw_dir / "train-images-idx3-ubyte")

        if (raw_dir / "train-labels-idx1-ubyte").exists():
            train_labels = load_mnist_labels(raw_dir / "train-labels-idx1-ubyte")
        else:
            train_labels = load_mnist_labels(raw_dir / "train-labels-idx1-ubyte")

        if (raw_dir / "t10k-images-idx3-ubyte").exists():
            test_images = load_mnist_images(raw_dir / "t10k-images-idx3-ubyte")
        else:
            test_images = load_mnist_images(raw_dir / "t10k-images-idx3-ubyte")

        if (raw_dir / "t10k-labels-idx1-ubyte").exists():
            test_labels = load_mnist_labels(raw_dir / "t10k-labels-idx1-ubyte")
        else:
            test_labels = load_mnist_labels(raw_dir / "t10k-labels-idx1-ubyte")

        self.train_source = MNISTDataSource(train_images, train_labels)
        self.val_source = MNISTDataSource(test_images, test_labels)
        self.test_source = MNISTDataSource(test_images, test_labels)

    def train_dataloader(self):
        return grain.DataLoader(
            data_source=self.train_source,
            sampler=grain.IndexSampler(
                num_records=len(self.train_source),
                shuffle=True,
                seed=self.seed,
                shard_options=grain.NoSharding(),
                num_epochs=None,
            ),
            worker_count=self.num_workers,
            operations=[grain.Batch(batch_size=self.batch_size, drop_remainder=True)],
        )

    def val_dataloader(self):
        return grain.DataLoader(
            data_source=self.val_source,
            sampler=grain.IndexSampler(
                num_records=len(self.val_source),
                shuffle=False,
                seed=self.seed,
                shard_options=grain.NoSharding(),
                num_epochs=None,
            ),
            worker_count=self.num_workers,
            operations=[grain.Batch(batch_size=self.batch_size, drop_remainder=False)],
        )

    def test_dataloader(self):
        return grain.DataLoader(
            data_source=self.test_source,
            sampler=grain.IndexSampler(
                num_records=len(self.test_source),
                shuffle=False,
                seed=self.seed,
                shard_options=grain.NoSharding(),
                num_epochs=None,
            ),
            worker_count=self.num_workers,
            operations=[grain.Batch(batch_size=self.batch_size, drop_remainder=False)],
        )

import numpy as np
import jax.numpy as jnp
import math

class NumpyDataset:
    def __init__(self, *arrays):
        self.arrays = arrays
        self.length = len(arrays[0])
        assert all(len(a) == self.length for a in arrays)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return tuple(a[idx] for a in self.arrays)

class JAXDataLoader:
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

from typing import Tuple

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset, TensorDataset
from torchvision.datasets import MNIST


def rand_dataset(num_rows=600, num_columns=20) -> Dataset:
    return TensorDataset(torch.rand(num_rows, num_columns))


class MNISTDataset(Dataset):
    """MNIST dataset wrapper for VAE (returns only images as flattened vectors)"""

    def __init__(self, train=True):
        self.mnist = MNIST(root='./data', train=train, download=True, transform=None)

    def __getitem__(self, index):
        img, _ = self.mnist[index]
        img_tensor = torch.tensor(np.array(img), dtype=torch.float32).flatten() / 255.0
        return img_tensor

    def __len__(self):
        return len(self.mnist)


def mnist_dataset(train=True) -> Dataset:
    """
    Returns the MNIST dataset for training or testing.

    Args:
        train (bool): If True, returns the training dataset. Otherwise, returns the testing dataset.

    Returns:
        Dataset: The MNIST dataset.
    """
    return MNISTDataset(train=train)

class CSVDataset(Dataset):

    def __init__(self):
        data = np.loadtxt('./data/test.csv', delimiter=',', dtype=np.float32, skiprows=0)
        self.x = torch.from_numpy(data)
        self.y = torch.from_numpy(data[:, [4]]) if data.shape[1] > 4 else None
        self.n_samples = data.shape[0]

    def __getitem__(self, index):
        return self.x[index]

    def __len__(self):
        return self.n_samples


def test_dataset() -> Dataset:
    return CSVDataset()

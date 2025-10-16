from typing import Tuple
import torch
from torch import Tensor
from torch.utils.data import Dataset, TensorDataset
from torchvision.datasets import MNIST
import numpy as np

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

class CSVDataset():

    def __init__(self):
      
        data1 = np.loadtxt('./data/NF-UNSW-NB15.csv', delimiter=',',
                           dtype=np.str_, skiprows=1)
        # print(data1)
        num1 = 0
        num2 = 0
        for i in data1:
            num2 = 0
            for j in i:
                data1[num1,num2] = float("".join(j.split(".")))
                num2 = num2 + 1
            num1 = num1 + 1
        # print(data1)
        data1 = data1.astype(float)
        # print(data1)

        self.x = torch.from_numpy(data1[:, :]).type(torch.float)
        self.n_samples = data1.shape[0] 

        print(self.x)
        print(self.n_samples)
    
    # support indexing such that dataset[i] can 
    # be used to get i-th sample
    def __getitem__(self, index):
        return self.x[index]
      
    # we can call len(dataset) to return the size
    def __len__(self):
        return self.n_samples
    

def test_dataset() -> CSVDataset:# Dataset:

    # testing
    return CSVDataset()

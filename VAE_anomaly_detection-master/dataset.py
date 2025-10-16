from typing import Tuple
import torch
from torch import Tensor
from torch.utils.data import Dataset, TensorDataset
from torchvision.datasets import MNIST
import numpy as np

def rand_dataset(num_rows=600, num_columns=20) -> Dataset:
    return TensorDataset(torch.rand(num_rows, num_columns))


def mnist_dataset(train=True) -> Dataset:
    """
    Returns the MNIST dataset for training or testing.
    
    Args:
    train (bool): If True, returns the training dataset. Otherwise, returns the testing dataset.
    
    Returns:
    Dataset: The MNIST dataset.
    """
    return MNIST(root='./data', train=train, download=True, transform=None)

class CSVDataset():
    def __init__(self):
      
        data1 = np.loadtxt('./data/test.csv', delimiter=',',
                           dtype=np.float32, skiprows=1)
        
        self.x = torch.from_numpy(data1[:, :25])
        self.y = torch.from_numpy(data1[:, [4]])
        self.n_samples = data1.shape[0] 
    
    # support indexing such that dataset[i] can 
    # be used to get i-th sample
    def __getitem__(self, index):
        return self.x[index], self.y[index]
      
    # we can call len(dataset) to return the size
    def __len__(self):
        return self.n_samples
    

def test_dataset() -> Dataset:

    # testing
    return CSVDataset()

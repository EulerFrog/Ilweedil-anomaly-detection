from abc import abstractmethod
import math
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

class VAEDataset():
    """
        Defines a dataset which when fully extracted is in the form of:

        records = Tensor(
            [data record 1],
            [data record 2],
            .
            .
            .
            [data record n]
        )

        labels = Tensor(
            [label of data record 1],
            [label of data record 2],
            .
            .
            .
            [label of data record n]
        )
    
    """
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def __getitem__(self, index:int) -> Tensor:
        """
            Expects index to be 0 <= index < self.__ln__() 

            Gets the data record stored at index "index"
        """
        pass

    @abstractmethod
    def __getbatch__(self, batch_size:int) -> list:
        """
            Expects batch_size to be 1 <= index <= self.__ln__() 

            This method generates a batch of size "batch_size" of shuffled data records from the VAE dataset. 
            Then, it returns it in the form of [records, labels] where "records" is the random_list 
            of data records and "labels" is the parallel list of labels of each record. 

            *Both "records" and "labels" are tensors based
        """
        pass

    @abstractmethod
    def __len__(self):
        """
            Returns the size of the dataset (number of data records)
        """
        pass

    @abstractmethod
    def __getitemlabel__(self, index: int):
        """
            Expects index to be 0 <= index < self.__ln__() 

            Gets the label of the data record stored at index "index"
        """
        pass

class CSVDataset(VAEDataset):


    def __init__(self):
      
        data1 = np.loadtxt('./data/NF-UNSW-NB15.csv', delimiter=',',
                           dtype=np.str_, skiprows=1)
        num1 = 0
        num2 = 0
        output = []
        outputLabels = []
        for i in data1:
            num2 = 0
            output.append([])

            for j in i:
                
                temp = float("".join(j.split(".")))
                if (num2 >= 4):

                    if (num2 == 4): # protocol
                        temp = temp/17
                        output[num1].append(temp)
                    elif (num2 == 5): # l7 proto
                        temp = temp/92
                        output[num1].append(temp)
                    elif (num2 == 6): # bytes in
                        temp = temp/1000000000
                        output[num1].append(temp)
                    elif (num2 == 7): # bytes out
                        temp = temp/1000000000
                        output[num1].append(temp)
                    elif (num2 == 8): # in pkts
                        temp = temp/1000
                        output[num1].append(temp)
                    elif (num2 == 9): # out pkts
                        temp = temp/1000
                        output[num1].append(temp)
                    elif (num2 == 10): # tcp flags
                        temp = temp/50
                        output[num1].append(temp)
                    elif (num2 == 11): # duration (millis)
                        temp = temp/1000000
                        output[num1].append(temp)
                    elif (num2 == 12): # duration (millis)
                        outputLabels.append(temp)

                num2 = num2 + 1
            num1 = num1 + 1
        output = np.asarray(output, dtype=np.float32)

        self.x = torch.from_numpy(output[:, :]).type(torch.float)
        labels = np.asarray(outputLabels, dtype=np.float32)
        self.labels = torch.from_numpy(labels).type(torch.float)
        self.n_samples = output.shape[0] 

    # support indexing such that dataset[i] can 
    # be used to get i-th sample
    def __getitem__(self, index) -> Tensor:
        return self.x[index]
    
    def __getbatch__(self, batch_size) -> list:
        """
            This method generates a batch of size "batch_size" of shuffled data records from the CSV dataset. 
            Then, it returns it in the form of [record_list, label_list] where "record_list" is the random_list 
            of data records and "label_list" is the parallel list of labels of each record. 

            *Both "record_list" and "label_list" are tensors
        """
        output_tensor = Tensor()
        data_record_tensor = []
        label_tensor = []

        # Shuffle a list of indexes. Indexes from index 0 to 'batch size' will be returned
        indexes = np.arange(0, self.__len__())
        np.random.shuffle(indexes)

        # Gather random data records with labels
        i = 0
        while i < batch_size:
            data_record_tensor.append(self.__getitem__(indexes[i]))
            label_tensor.append(self.__getitemlabel__(indexes[i]))
            i = i + 1            

        # Convert lists to tensors and stack
        data_record_tensor = torch.stack(data_record_tensor)
        label_tensor = torch.stack(label_tensor)
        output_tensor = [data_record_tensor, label_tensor]

        return output_tensor
      
    def __getitemlabel__(self, index) -> int:
        return self.labels[index]
    
    # we can call len(dataset) to return the size
    def __len__(self):
        return self.n_samples
    

def test_dataset() -> Dataset:# Dataset:

    # testing
    return CSVDataset()

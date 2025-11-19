from abc import abstractmethod
import abc
import math
from typing import Tuple

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset, TensorDataset, DataLoader
from torchvision.datasets import MNIST
from sklearn.datasets import make_classification
import pandas as pd
from abc import ABC, abstractmethod

class VAEDataset(ABC):
    """
        Abstract base class for VAE datasets.

        Takes data and labels tensors, separates them into benign (label=0)
        and anomalous (label=1) datasets, and provides dataloaders for each.

        Datasets contain:
        - data: Tensor of shape (n_samples, m_features)
        - labels: Tensor of shape (n_samples,) with values 0 (benign) or 1 (anomalous)
    """

    

    def __init__(self, from_tensor: bool = False, data: Tensor = None, labels: Tensor = None, from_file: bool = False, data_file_path: str = ""):
        """
            Initialize the dataset with data and labels.

            Args:
                from_tensor: If set to true (and no matter what 'from_file' is), VAEDataset will attempt to load from and inputted 'data' and 'labels'
                data: Tensor of shape (n_samples, n_features) containing all samples
                labels: Tensor of shape (n_samples,) with 0 for benign, 1 for 
                from_file: If set to true AND 'from_tensor' is false, VAEDataset will attempt to load from a .csv file with path 'data_file_path'
                    and contents in the format of

                    row 1: feature 1, feature 2, ... , feature m, label 
                    .
                    .
                    .
                    row n:  ...
        """

        self.initialized = False    # Represents whether the dataset's self.data and self.labels have been populated

        # Neither from tensor nor file case
        if not from_tensor and not from_file:
            raise ValueError("VAEDataset (err): Could not initialize dataset (from_tensor and from_file left False)")

        # From tensor case
        if (from_tensor):
            if (data is None or labels is None):
                raise ValueError("VAEDataset (err): Could not initialize dataset (labels and/or data were missing)")
            
            self.initialize_dataset(data=data, labels=labels)

        elif (from_file):

            # Open and init data from inputted .csv file into two tensors, one for data and one for labels
            df = pd.read_csv(data_file_path)
            self.initialize_dataset(
                torch.tensor(df.iloc[:, :-1].values, dtype=torch.float32), 
                torch.tensor(df.iloc[:, -1].values, dtype=torch.int)
            )



    def initialize_dataset(self, data: Tensor, labels: Tensor):
        """
            Method with the purpose of initializing the data and labels of all records in this class.

            Args:
                data: Tensor - Rows of numerical data records this class instance is to wrap
                labels: Tensor - A parallel tensor of labels 0 and 1. Each represents the label 
                    of the data record in 'data' in parallel with the label.
        """
        self.data = data
        self.labels = labels
        self.input_size = self.data.shape[1]

        # Separate benign and anomalous data
        benign_mask = labels == 0
        anomalous_mask = labels == 1
        self.benign_data = data[benign_mask]
        self.anomalous_data = data[anomalous_mask]
        self.benign_data_length = self.benign_data.size(dim=0)
        self.anomalous_data_length = self.anomalous_data.size(dim=0)

        # Set initialized to true
        self.initialized = True

        # Shuffle the dataset. Initalize the starting index of unallocated benign and anomalous data to data loaders. Any index below
        #   an index has not been allocated to a data loader
        self.reset_dataset()

        print(f"Dataset initialized: {len(self.benign_data)} benign, {len(self.anomalous_data)} anomalous samples")

    def reset_dataset(self):
        """
        
        """
        if (self.initialized == False):
            print("VAEDataset (err): Could not reset dataset (self.labels and self.data uninitialized)")
            return
        else:
            self.benign_data = self.benign_data[torch.randperm(self.benign_data.size()[0])] # Shuffle benign data
            self.anomalous_data = self.anomalous_data[torch.randperm(self.anomalous_data.size()[0])] # Shuffle anomalous data
            self.unallocated_benign_data_start_index = 0    # Reset indices of allocated data records for benign and anomalous data
            self.unallocated_anomalous_data_start_index = 0

    def get_benign_dataloader(
        self,
        batch_size: int = 32,
        num_workers: int = 0,
        size: int = 32,
        **kwargs
    ):
        """
            Creates and returns a DataLoader for benign (normal) data.

            Args:
                batch_size: Number of samples per batch
                shuffle: Whether to shuffle the data
                num_workers: Number of worker processes for data loading
                size: Number of records in the data loader
                **kwargs: Additional arguments to pass to DataLoader

            Returns:
                DataLoader for benign data
        """
        if len(self.benign_data) == 0:
            raise ValueError("No benign samples available in dataset")
        
        # Make sure batch_size is equal or greater than size requested
        if size < batch_size:
            raise Exception("VAEDataset (err): Could not create benign data loader (size < batch_size)")
       
        # Make sure enough data can be allocated from the benign dataset
        if size + self.unallocated_benign_data_start_index > self.benign_data_length:
            raise Exception(f"VAEDataset (err): Could not create benign data loader (requested loader of size {size} but only {self.benign_data_length - self.unallocated_benign_data_start_index} records available)")

        slice = self.benign_data[self.unallocated_benign_data_start_index:(self.unallocated_benign_data_start_index+size)]
        self.unallocated_benign_data_start_index = self.unallocated_benign_data_start_index + size
        dataset = TensorDataset(slice)
        return DataLoader(dataset, batch_size=batch_size, shuffle=False,
                         num_workers=num_workers, **kwargs)

    def get_anomalous_dataloader(
        self,
        batch_size: int = 32,
        num_workers: int = 0,
        size: int = 32,
        **kwargs
    ):
        """
            Creates and returns a DataLoader for anomalous data.

            Args:
                batch_size: Number of samples per batch
                shuffle: Whether to shuffle the data
                num_workers: Number of worker processes for data loading
                size: Number of records in the data loader
                **kwargs: Additional arguments to pass to DataLoader

            Returns:
                DataLoader for anomalous data
        """
        if len(self.anomalous_data) == 0:
            raise ValueError("VAEDataset (err): No anomalous samples available in dataset")

        # Make sure batch_size is equal or greater than size requested
        if size < batch_size:
            raise Exception("VAEDataset (err): Could not create anomalous data loader (size < batch_size)")
        
        # Make sure enough data can be allocated from the anomalous dataset
        if size + self.unallocated_anomalous_data_start_index > self.anomalous_data_length:
            raise Exception(f"VAEDataset (err): Could not create anomalous data loader (requested loader of size {size} but only {self.anomalous_data_length - self.unallocated_anomalous_data_start_index} records available)")


        slice = self.anomalous_data[self.unallocated_anomalous_data_start_index:(self.unallocated_anomalous_data_start_index+size)]
        self.unallocated_anomalous_data_start_index = self.unallocated_anomalous_data_start_index + size
        dataset = TensorDataset(slice)
        return DataLoader(dataset, batch_size=batch_size, shuffle=False,
                         num_workers=num_workers, **kwargs)

    def get_dataloader(
        self,
        batch_size: int = 32,
        num_workers: int = 0,
        benign_size: int = 16,
        anomalous_size: int = 16,
        **kwargs      
    ):
        """
            Creates and returns a DataLoader with both anomalous and benign data, containing
            'benign_size' benign data records and 'anomalous_size' anomalous data records.

            Args:
                batch_size: Number of samples per batch
                shuffle: Whether to shuffle the data
                num_workers: Number of worker processes for data loading
                benign_size: Number of benign records in the data loader
                anomalous_size: Number of anomalous records in the data loader
                **kwargs: Additional arguments to pass to DataLoader

            Returns:
                DataLoader for anomalous data
        """
        if len(self.anomalous_data) == 0:
            raise ValueError("No anomalous samples available in dataset")

        if len(self.benign_data) == 0:
            raise ValueError("No benign samples available in dataset")
        
        # Make sure batch_size is equal or greater than size requested
        if benign_size + anomalous_size < batch_size:
            raise Exception(f"VAEDataset (err): Could not create data loader (total size {benign_size + anomalous_size} < batch_size {batch_size})")

         # Make sure enough data can be allocated from the anomalous dataset
        if anomalous_size + self.unallocated_anomalous_data_start_index > self.anomalous_data_length:
            raise Exception(f"VAEDataset (err): Could not create data loader (requested loader of size {anomalous_size} but only {self.anomalous_data_length - self.unallocated_anomalous_data_start_index} records available)")

        # Make sure enough data can be allocated from the benign dataset
        if benign_size + self.unallocated_benign_data_start_index > self.benign_data_length:
            raise Exception(f"VAEDataset (err): Could not create data loader (requested loader of size {benign_size} but only {self.benign_data_length - self.unallocated_benign_data_start_index} records available)")

        # Create slices of anomalous and benign data. Create parallel labels for each data record based on if the record is anomalous or benign.
        anomalous_slice_data = self.anomalous_data[self.unallocated_anomalous_data_start_index:(self.unallocated_anomalous_data_start_index+anomalous_size)]
        anomalous_slice_labels = torch.tensor((), dtype=torch.int)
        anomalous_slice_labels = anomalous_slice_data.new_ones((anomalous_size, 1))
        benign_slice_data = self.benign_data[self.unallocated_benign_data_start_index:(self.unallocated_benign_data_start_index+anomalous_size)]
        benign_slice_labels = torch.tensor((), dtype=torch.int)
        benign_slice_labels = benign_slice_labels.new_zeros((benign_size,1))
        
        # print('anomalous slice')
        # print(anomalous_slice_data)
        # print(anomalous_slice_data.size())
        # print('benign_slice')
        # print(benign_slice_data)
        # print(benign_slice_data.size())

        # Stack slices and convert to TensorDataset.
        slice_data = torch.cat([anomalous_slice_data, benign_slice_data], dim=0)
        slice_labels = torch.cat([anomalous_slice_labels, benign_slice_labels], dim=0)

        # print('total slice')
        # print(slice_data)
        # print(slice_data.size())

        # Adjust starting indexes
        self.unallocated_anomalous_data_start_index = self.unallocated_anomalous_data_start_index + anomalous_size
        self.unallocated_benign_data_start_index = self.unallocated_benign_data_start_index + benign_size

        dataset = TensorDataset(slice_data, slice_labels)
        return DataLoader(dataset, batch_size=int(batch_size), shuffle=False,
                         num_workers=num_workers, **kwargs)

    def get_full_dataloader(
        self,
        batch_size: int = 32,
        shuffle: bool = False,
        num_workers: int = 0,
        **kwargs
    ):
        """
            Creates and returns a DataLoader for all data (benign + anomalous).

            Args:
                batch_size: Number of samples per batch
                shuffle: Whether to shuffle the data
                num_workers: Number of worker processes for data loading
                **kwargs: Additional arguments to pass to DataLoader

            Returns:
                DataLoader for all data with labels
        """
        dataset = TensorDataset(self.data, self.labels)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                         num_workers=num_workers, **kwargs)


class SyntheticAnomalyDataset(VAEDataset):
    """
    Wrapper class for synthetic anomaly detection dataset using make_classification.

    Supports n distinct classes where:
    - Classes 0 to n-2 are normal classes
    - Class n-1 is the anomalous class

    Three splits:
    - 'train': Only normal samples (classes 0 to n-2) for training
    - 'val': Only normal samples (classes 0 to n-2) for validation
    - 'test': All samples including anomalies (classes 0 to n-1) for evaluation
    """

    def __init__(
        self,
        n_samples=10000,
        n_features=20,
        n_classes=2,
        n_informative=10,
        n_redundant=2,
        anomaly_ratio=0.02,
        random_state=42,
        split='train',
        train_ratio=0.6,
        val_ratio=0.2,
        test_ratio=0.2,
        class_sep=1.0
    ):
        """
        Initialize the synthetic anomaly dataset.

        Args:
            n_samples: Total number of samples to generate
            n_features: Number of features per sample
            n_classes: Total number of classes (last class is anomalous)
            n_informative: Number of informative features
            n_redundant: Number of redundant features
            anomaly_ratio: Ratio of anomalies in overall dataset (default 0.02 = 2%)
            random_state: Random seed for reproducibility
            split: 'train', 'val', or 'test'
            train_ratio: Ratio of data for training (default 0.6)
            val_ratio: Ratio of data for validation (default 0.2)
            test_ratio: Ratio of data for testing (default 0.2)
            class_sep: Separation between classes (higher = more distinct, default 1.0)
        """
        assert split in ['train', 'val', 'test'], "split must be 'train', 'val', or 'test'"
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"
        assert n_classes >= 2, "Must have at least 2 classes (1 normal + 1 anomaly)"

        self.n_classes = n_classes
        self.anomaly_class = n_classes - 1  # Last class is anomaly

        # Calculate weights: distribute normal ratio evenly among normal classes
        normal_ratio = 1.0 - anomaly_ratio
        normal_class_ratio = normal_ratio / (n_classes - 1)
        weights = [normal_class_ratio] * (n_classes - 1) + [anomaly_ratio]

        # Generate synthetic data
        X, y = make_classification(
            n_samples=n_samples,
            n_features=n_features,
            n_classes=n_classes,
            n_informative=n_informative,
            n_redundant=n_redundant,
            n_clusters_per_class=1,
            weights=weights,
            class_sep=class_sep,
            random_state=random_state,
            flip_y=0  # No label noise
        )

        # Convert to float32
        X = X.astype(np.float32)

        # Calculate split indices
        train_end = int(n_samples * train_ratio)
        val_end = int(n_samples * (train_ratio + val_ratio))

        # Split data
        if split == 'train':
            X_split = X[:train_end]
            y_split = y[:train_end]

            # Filter to only normal samples (classes 0 to n-2)
            normal_mask = y_split < self.anomaly_class
            self.x = torch.from_numpy(X_split[normal_mask]).type(torch.float)
            self.labels = torch.from_numpy(y_split[normal_mask]).type(torch.long)

            # Count samples per class
            class_counts = {i: (y_split[normal_mask] == i).sum() for i in range(n_classes - 1)}
            print(f"Training set: {len(self.x)} normal samples from {n_classes - 1} classes")
            print(f"  Class distribution: {class_counts}")

        elif split == 'val':
            X_split = X[train_end:val_end]
            y_split = y[train_end:val_end]

            # Filter to only normal samples (classes 0 to n-2)
            normal_mask = y_split < self.anomaly_class
            self.x = torch.from_numpy(X_split[normal_mask]).type(torch.float)
            self.labels = torch.from_numpy(y_split[normal_mask]).type(torch.long)

            # Count samples per class
            class_counts = {i: (y_split[normal_mask] == i).sum() for i in range(n_classes - 1)}
            print(f"Validation set: {len(self.x)} normal samples from {n_classes - 1} classes")
            print(f"  Class distribution: {class_counts}")

        else:  # split == 'test'
            X_split = X[val_end:]
            y_split = y[val_end:]

            # Keep all samples (normal + anomaly) with labels
            self.x = torch.from_numpy(X_split).type(torch.float)
            self.labels = torch.from_numpy(y_split).type(torch.long)

            # Count samples per class
            class_counts = {i: (y_split == i).sum() for i in range(n_classes)}
            n_normal = sum((y_split == i).sum() for i in range(n_classes - 1))
            n_anomaly = (y_split == self.anomaly_class).sum()

            print(f"Test set: {len(self.x)} samples ({n_normal} normal, {n_anomaly} anomalies)")
            print(f"  Class distribution: {class_counts}")
            print(f"  Anomaly class: {self.anomaly_class}")

        # Convert labels to binary: 0 for benign, 1 for anomalous
        if split == 'train' or split == 'val':
            # Train and val sets only have normal samples, so all labels are 0
            binary_labels = torch.zeros(len(self.x), dtype=torch.long)
        else:  # test split
            # Test set has both normal and anomalous samples
            # Convert: normal classes (0 to n-2) -> 0, anomaly class (n-1) -> 1
            binary_labels = (self.labels == self.anomaly_class).long()

        # Initialize parent class
        super().__init__(self.x, binary_labels)

        # Store additional attributes
        self.n_samples = len(self.x)
        self.split = split

    def __getitem__(self, index):
        """Get a single sample"""
        return self.x[index]

    def __getitemlabel__(self, index):
        """Get label for a single sample"""
        return self.labels[index]

    def __len__(self):
        """Return the size of the dataset"""
        return self.n_samples

    def __getbatch__(self, batch_size: int) -> list:
        """
            This method generates a batch of size "batch_size" of shuffled data records.
            Returns [data_records, labels] where both are tensors.

            Args:
                batch_size: Number of samples in the batch

            Returns:
                List containing [data_tensor, label_tensor]
        """
        # Shuffle indexes
        perm = torch.randperm(len(self.x))
        selected_indices = perm[:min(batch_size, len(self.x))]

        # Get data and labels
        data_tensor = self.x[selected_indices]
        label_tensor = self.labels[selected_indices]

        return [data_tensor, label_tensor]




# class NetflowDatset():
#     """
#         Class created for research that initializes a dataset for 
#         training the anomaly detection VAE on.
#     """
#     def __init__(self, data_file: str, config=None):
#         # self.blank = config.blank a bunch of times
#         # self.data = data
#         # self.label = grab labels from data
#         # steps: load_csv -> separate the last column into self.labels and the rest into self.data
#         # throw some breakpoints for debugging
#         df = pd.read_csv(data_file)
#         self.data = df.iloc[:, :-1].values
#         self.labels = df.iloc[:, -1].values
#         self.input_size = self.data.shape[1]
#         print("Data loaded successfully")
#         print("Data shape:", self.data.shape)
#         print("Labels shape:", self.labels.shape)
#         print("Input size: ", str(self.input_size))
#         # for i in range(0,len(self.data)):
#         #     print("Record " + str(i))
#         #     print(self.data[i])
#         #     print(self.labels[i])
#         #     input()

#     def __getinputsize__(self):
#         return self.input_size

#     def __len__(self):
#         # return len(self.data)
#         return len(self.data)

#     def __getitem__(self, idx):
#         # return self.data[idx], self.label[idx]
#         sample = torch.tensor(self.data[idx], dtype=torch.float32)
#         return sample
    
#     def __getitemlabel__(self, idx):
#         label = torch.tensor(self.labels[idx], dtype=torch.long)
#         return label
    
#     def __getbatch__(self, batch_size: int, label: int = 0):


#         output_tensor = Tensor()
#         data_record_tensor = []
#         indexes = []
#         label_tensor = []
#         # print("From NetflowDataset: Get batch")

#         # Shuffle a list of indexes matching the inputted label. Indexes from index 0 to 'batch size' will be returned
#         # print(self.labels)
#         for i in range(0, len(self.labels)):
#             if self.labels[i] == label:
#                 indexes.append(i)

#         indexes = np.array(indexes)
#         np.random.shuffle(indexes)

#         # If there are not enough records to provide in the batch, raise an acception
#         if (batch_size > len(indexes)):
#             raise Exception("NetflowDataset: Not enough records to provide for batch size {" + str(batch_size) + "}.") 
        
#         # Gather random data records with labels
#         i = 0
#         while i < batch_size:
#             data_record_tensor.append(self.__getitem__(indexes[i]))
#             label_tensor.append(self.__getitemlabel__(indexes[i]))
#             i = i + 1            
#         # print("     - Batch unstacked. Data then labels")
#         # print(data_record_tensor)
#         # print(label_tensor)

#         # Convert lists to tensors and stack
#         data_record_tensor = torch.stack(data_record_tensor)
#         label_tensor = torch.stack(label_tensor)
#         output_tensor = [data_record_tensor, label_tensor]

#         # print("     - Batch stacked. Data then labels")
#         # print(data_record_tensor)
#         # print(label_tensor)
#         # print("******************")
#         return output_tensor

# class CSVDataset():
#     """
#         CSV dataset used to test VAE prior to receiving opensearch data.
#     """

#     def __init__(self):
      
#         data1 = np.loadtxt('./data/NF-UNSW-NB15.csv', delimiter=',',
#                            dtype=np.str_, skiprows=1)
#         num1 = 0
#         num2 = 0
#         output = []
#         outputLabels = []

#         for i in data1:
#             num2 = 0
#             output.append([])

#             for j in i:
                
#                 temp = float("".join(j.split(".")))
#                 if (num2 >= 4):

#                     if (num2 == 4): # protocol
#                         temp = temp/17
#                         output[num1].append(temp)
#                     elif (num2 == 5): # l7 proto
#                         temp = temp/92
#                         output[num1].append(temp)
#                     elif (num2 == 6): # bytes in
#                         temp = temp/1000000000
#                         output[num1].append(temp)
#                     elif (num2 == 7): # bytes out
#                         temp = temp/1000000000
#                         output[num1].append(temp)
#                     elif (num2 == 8): # in pkts
#                         temp = temp/1000
#                         output[num1].append(temp)
#                     elif (num2 == 9): # out pkts
#                         temp = temp/1000
#                         output[num1].append(temp)
#                     elif (num2 == 10): # tcp flags
#                         temp = temp/50
#                         output[num1].append(temp)
#                     elif (num2 == 11): # duration (millis)
#                         temp = temp/1000000
#                         output[num1].append(temp)
#                     elif (num2 == 12): # label
#                         outputLabels.append(temp)

#                 num2 = num2 + 1
#             num1 = num1 + 1
#         output = np.asarray(output, dtype=np.float32)

#         self.x = torch.from_numpy(output[:, :]).type(torch.float)
#         labels = np.asarray(outputLabels, dtype=np.float32)
#         self.labels = torch.from_numpy(labels).type(torch.float)
#         self.n_samples = output.shape[0] 

#     # support indexing such that dataset[i] can 
#     # be used to get i-th sample
#     def __getitem__(self, index) -> Tensor:
#         return self.x[index]
    
#     def __getbatch__(self, batch_size: int, label: int) -> list:
#         """
#             This method generates a batch of size "batch_size" of shuffled data records from the CSV dataset. 
#             Then, it returns it in the form of [record_list, label_list] where "record_list" is the random_list 
#             of data records and "label_list" is the parallel list of labels of each record. 

#             *Both "record_list" and "label_list" are tensors
#         """

#         output_tensor = Tensor()
#         data_record_tensor = []
#         indexes = []
#         label_tensor = []
#         print("From CSVDataset: Get batch")

#         # Shuffle a list of indexes matching the inputted label. Indexes from index 0 to 'batch size' will be returned
#         print(self.labels)
#         for i in range(0, len(self.labels)):
#             if self.labels[i] == label:
#                 indexes.append(i)

#         indexes = np.array(indexes)
#         np.random.shuffle(indexes)

#         # If there are not enough records to provide in the batch, raise an acception
#         if (batch_size > len(indexes)):
#             raise Exception("CSVDataset: Not enough records to provide for batch size {" + str(batch_size) + "}.") 
        
#         # Gather random data records with labels
#         i = 0
#         while i < batch_size:
#             data_record_tensor.append(self.__getitem__(indexes[i]))
#             label_tensor.append(self.__getitemlabel__(indexes[i]))
#             i = i + 1            
#         print("     - Batch unstacked. Data then labels")
#         print(data_record_tensor)
#         print(label_tensor)

#         # Convert lists to tensors and stack
#         data_record_tensor = torch.stack(data_record_tensor)
#         label_tensor = torch.stack(label_tensor)
#         output_tensor = [data_record_tensor, label_tensor]

#         print("     - Batch stacked. Data then labels")
#         print(data_record_tensor)
#         print(label_tensor)
#         # print("******************")
#         return output_tensor
      
#     def __getitemlabel__(self, index) -> int:
#         return self.labels[index]
    
#     def __getinputsize__(self):
#         return self.labels.shape[1]
    
#     # we can call len(dataset) to return the size
#     def __len__(self):
#         return self.n_samples
    

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


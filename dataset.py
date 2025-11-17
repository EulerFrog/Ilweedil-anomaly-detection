import math
from typing import Tuple

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset, TensorDataset, DataLoader
from torchvision.datasets import MNIST
from sklearn.datasets import make_classification
from abc import ABC, abstractmethod


class VAEDataset(ABC):
    """
        Abstract base class for VAE datasets.

        Takes data and labels tensors, separates them into benign (label=0)
        and anomalous (label=1) datasets, and provides dataloaders for each.

        Datasets contain:
        - data: Tensor of shape (n_samples, n_features)
        - labels: Tensor of shape (n_samples,) with values 0 (benign) or 1 (anomalous)
    """

    def __init__(self, data: Tensor, labels: Tensor):
        """
            Initialize the dataset with data and labels.

            Args:
                data: Tensor of shape (n_samples, n_features) containing all samples
                labels: Tensor of shape (n_samples,) with 0 for benign, 1 for anomalous
        """
        self.data = data
        self.labels = labels

        # Separate benign and anomalous data
        benign_mask = labels == 0
        anomalous_mask = labels == 1

        self.benign_data = data[benign_mask]
        self.anomalous_data = data[anomalous_mask]

        print(f"Dataset initialized: {len(self.benign_data)} benign, {len(self.anomalous_data)} anomalous samples")

    def get_benign_dataloader(
        self,
        batch_size: int = 32,
        shuffle: bool = True,
        num_workers: int = 0,
        **kwargs
    ):
        """
            Creates and returns a DataLoader for benign (normal) data.

            Args:
                batch_size: Number of samples per batch
                shuffle: Whether to shuffle the data
                num_workers: Number of worker processes for data loading
                **kwargs: Additional arguments to pass to DataLoader

            Returns:
                DataLoader for benign data
        """
        if len(self.benign_data) == 0:
            raise ValueError("No benign samples available in dataset")

        dataset = TensorDataset(self.benign_data)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                         num_workers=num_workers, **kwargs)

    def get_anomalous_dataloader(
        self,
        batch_size: int = 32,
        shuffle: bool = True,
        num_workers: int = 0,
        **kwargs
    ):
        """
            Creates and returns a DataLoader for anomalous data.

            Args:
                batch_size: Number of samples per batch
                shuffle: Whether to shuffle the data
                num_workers: Number of worker processes for data loading
                **kwargs: Additional arguments to pass to DataLoader

            Returns:
                DataLoader for anomalous data
        """
        if len(self.anomalous_data) == 0:
            raise ValueError("No anomalous samples available in dataset")

        dataset = TensorDataset(self.anomalous_data)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
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


class MNISTDataset(VAEDataset):
    """
    MNIST dataset wrapper for VAE (returns flattened images).
    Digit 0 is treated as anomalous, all other digits (1-9) are benign.
    """

    def __init__(self, train=True):
        # Load MNIST dataset
        mnist = MNIST(root='./data', train=train, download=True, transform=None)

        # Process all images and labels
        all_data = []
        all_labels = []

        for idx in range(len(mnist)):
            img, label = mnist[idx]
            # Flatten and normalize image
            img_tensor = torch.tensor(np.array(img), dtype=torch.float32).flatten() / 255.0
            all_data.append(img_tensor)
            # Binary label: 1 if digit 0 (anomalous), 0 otherwise (benign)
            binary_label = 1 if label == 0 else 0
            all_labels.append(binary_label)

        # Stack into tensors
        data_tensor = torch.stack(all_data)
        labels_tensor = torch.tensor(all_labels, dtype=torch.long)

        # Initialize parent class
        super().__init__(data_tensor, labels_tensor)

        # Store for backwards compatibility
        self.mnist = mnist

    def __getitem__(self, index):
        """Get a flattened, normalized image"""
        return self.data[index]

    def __len__(self):
        return len(self.data)


def mnist_dataset(train=True) -> Dataset:
    """
    Returns the MNIST dataset for training or testing.

    Args:
        train (bool): If True, returns the training dataset. Otherwise, returns the testing dataset.

    Returns:
        Dataset: The MNIST dataset.
    """
    return MNISTDataset(train=train)


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
                        outputLabels.append(num2)

                num2 = num2 + 1
            num1 = num1 + 1
        output = np.asarray(output, dtype=np.float32)

        data_tensor = torch.from_numpy(output[:, :]).type(torch.float)
        labels_array = np.asarray(outputLabels, dtype=np.float32)

        # Convert labels to binary: 0 for benign, 1 for anomalous
        # Assuming label 0 in CSV means benign, anything else is anomalous
        labels_binary = (labels_array > 0).astype(np.float32)
        labels_tensor = torch.from_numpy(labels_binary).type(torch.long)

        # Initialize parent class with data and labels
        super().__init__(data_tensor, labels_tensor)

        # Keep these for backwards compatibility
        self.x = data_tensor
        self.labels = labels_tensor
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
    

def test_dataset() -> CSVDataset:# Dataset:

    # testing
    return CSVDataset()

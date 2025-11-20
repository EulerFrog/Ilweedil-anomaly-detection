import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader

class NetflowDataset(Dataset):
    def __init__(self, data_file, config=None):
        # self.blank = config.blank a bunch of times
        # self.data = data
        # self.label = grab labels from data
        # steps: load_csv -> separate the last column into self.labels and the rest into self.data
        # throw some breakpoints for debugging
        df = pd.read_csv(data_file)
        self.data = df.iloc[:, :-1].values
        self.labels = df.iloc[:, -1].values
        print("Data loaded successfully")
        print("Data shape:", self.data.shape)
        print("Labels shape:", self.labels.shape)

    def __len__(self):
        # return len(self.data)
        return len(self.data)

    def __getitem__(self, idx):
        # return self.data[idx], self.label[idx]
        sample = torch.tensor(self.data[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return sample, label


import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader

class NetflowDatset(Dataset):
    def __init__(self, data, config=None):
        # self.blank = config.blank a bunch of times
        # self.data = data
        # self.label = grab labels from data

    def __len__(self):
        # return len(self.data)

    def __getitem__(self, idx):
        # return self.data[idx], self.label[idx]

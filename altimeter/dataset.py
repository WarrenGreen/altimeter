import json
from collections import defaultdict
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, Subset


def split_on_key(dataset, key):
    """
    Split dataset based on values for given key

    Args:
        dataset (Dataset):
        key (str):

    Returns:
        (List(Dataset)):
    """
    split_indices = defaultdict(list)

    for index in range(len(dataset)):
        datapoint = dataset[index]
        key_value = datapoint[key]
        split_indices[key_value].append(index)

    return {
        key: Subset(dataset, indices) for key, indices in split_indices.items()
    }


class JSONLDataset(Dataset):
    """
    In-memory jsonl dataset
    """
    def __init__(self, filepath):
        self.filepath = filepath
        self.datapoints = []
        with open(self.filepath, "r") as f:
            for line in f:
                data = json.loads(line)
                self.datapoints.append(data)

    def __len__(self):
        return len(self.datapoints)

    def __getitem__(self, index):
        return self.datapoints[index]


class SequenceDataset(Dataset):
    """
    Create datapoints of sequences
    """
    def __init__(self, dataset, sequence_length):
        self.dataset = dataset
        self.sequence_length = sequence_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        X = [self.dataset[idx] for idx in range(max(0, index-self.sequence_length), index)]
        X = [0.0] * (self.sequence_length - len(X)) + X
        y = self.dataset[index]
        return X, y


class SingleKeyDataset(Dataset):
    def __init__(self, dataset, key):
        self.dataset = dataset
        self.key = key

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        datapoint = self.dataset[index]
        return datapoint[self.key]


class NormalizeBatchDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        datapoint = self.dataset[index]
        if isinstance(datapoint, Tuple) and len(datapoint) == 2:
            X, y = datapoint
        else:
            X, y = datapoint, None
        min_val = np.min(X)
        max_val = np.max(X)
        top = X - min_val
        bottom = max_val - min_val
        if bottom != 0:
            X = top / bottom
        if y is not None:
            if bottom != 0:
                y = (y - min_val) / bottom
            return X, y
        else:
            return X


class TensorDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        datapoint = self.dataset[index]
        if isinstance(datapoint, Tuple) and len(datapoint) == 2:
            X, y = datapoint
        else:
            X, y = datapoint, None
        if y is not None:
            return torch.Tensor(X), torch.tensor(y)
        else:
            return torch.Tensor(X)

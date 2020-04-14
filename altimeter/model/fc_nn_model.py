import numpy as np
from torch import nn


class FCNNModel(nn.Module):
    """
    Fully-connected neural network to predict motion model.
    """
    def __init__(self, k=3, dropout_prob=0.5):
        super().__init__()
        self.k = k
        self.fc1 = nn.Linear(self.k, 256)
        self.dropout1(p=dropout_prob)
        self.fc2 = nn.Linear(256, 128)
        self.dropout2(p=dropout_prob)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, prev_values):
        if len(prev_values) < self.k:
            raise ValueError(f"Size of input be at least `k`: {len(prev_values)} != {self.k}")
        prev_values = prev_values[:self.k]
        x = self.fc1(prev_values)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        return x

    def predict(self, prev_values):
        return self.forward(prev_values)

    def preprocess(self, prev_values, y=None):
        if len(prev_values) < self.k:
            raise ValueError(f"Size of input must be at least `k`: {len(prev_values)} != {self.k}")
        prev_values = prev_values[:self.k]
        min_val = np.min(prev_values)
        max_val = np.max(prev_values)
        x = (prev_values - min_val) / (max_val - min_val)
        if y is not None:
            y = (y - min_val) / (max_val - min_val)
            return x, y
        else:
            return x

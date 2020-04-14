from copy import copy

import numpy as np
from sklearn.linear_model import LinearRegression


class LinearModel:
    def __init__(self, k=3, buffer_ratio=4, allow_sub_k=True):
        self.k = k
        self.buffer_ratio = buffer_ratio
        self.model = LinearRegression()
        self.variance = 3.
        self.allow_sub_k = allow_sub_k

    def predict(self, prev_values):
        if not self.allow_sub_k and len(prev_values) < self.k:
            raise ValueError(f"Size of input must be at least `k`: {len(prev_values)} != {self.k}")
        k = min(len(prev_values), self.k)
        xx = np.array(list(range(k))).reshape(-1, 1)
        yy = np.array(copy(prev_values[-k:])).reshape(-1, 1)
        self.model.fit(xx, yy)
        new_mu = self.model.predict([[k - 1]])
        delta = new_mu - prev_values[-1]

        return new_mu, delta

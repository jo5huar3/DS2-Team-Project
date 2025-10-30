import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

class LorenzStartsDataset(Dataset):
    """
    (Memory-light) Dataset that yields (warm, x, y) by slicing from the base series
    using precomputed start indices. No huge window tensor is created.
    """
    def __init__(self, series, starts, warm_length, x_length,
                 standardize=True, mean=None, std=None):
        self.series = series  # (T, D) float32
        self.starts = np.asarray(starts, dtype=np.int64)
        self.W = int(warm_length)
        self.X = int(x_length)
        self.total = self.W + self.X + 1
        self.standardize = bool(standardize)
        self.mean = None if mean is None else np.asarray(mean, dtype=np.float32)
        self.std  = None if std  is None else np.asarray(std,  dtype=np.float32)

        T = series.shape[0]
        if T <= self.total:
            raise ValueError(f"Need > {self.total} time steps, got {T}")
        if self.standardize:
            assert self.mean is not None and self.std is not None, "Need mean/std for standardization."

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, i):
        s = int(self.starts[i])
        wxy = self.series[s : s + self.total, :]     # (total, D) view
        warm = np.ascontiguousarray(wxy[: self.W, :], dtype=np.float32)
        x    = np.ascontiguousarray(wxy[self.W : self.W + self.X, :], dtype=np.float32)
        y    = np.ascontiguousarray(wxy[self.W + 1 : self.W + self.X + 1, :], dtype=np.float32)

        if self.standardize:
            warm = (warm - self.mean) / self.std
            x    = (x    - self.mean) / self.std
            y    = (y    - self.mean) / self.std

        return torch.from_numpy(warm), torch.from_numpy(x), torch.from_numpy(y)

class LorenzWindowViewDataset(Dataset):
    """
    Dataset that consumes a sliding_window_view (N, TOTAL, D) and an index array,
    yielding (warm, x, y) tensors without materializing a giant windows array.
    """
    def __init__(self, window_view, indices, warm_length, x_length,
                 standardize=True, mean=None, std=None):
        self.wins = window_view
        self.idx = np.asarray(indices, dtype=np.int64)
        self.W = int(warm_length)
        self.X = int(x_length)
        self.TOTAL = self.W + self.X + 1
        self.standardize = bool(standardize)
        self.mean = None if mean is None else np.asarray(mean, dtype=np.float32)
        self.std  = None if std  is None else np.asarray(std,  dtype=np.float32)

        # basic checks
        if self.wins.ndim != 3:
            raise ValueError(f"window_view must be 3D (N, TOTAL, D), got {self.wins.shape}")
        if self.wins.shape[1] != self.TOTAL:
            raise ValueError(f"Expected TOTAL={self.TOTAL}, got {self.wins.shape[1]}")
        if self.standardize and (self.mean is None or self.std is None):
            raise ValueError("mean/std required when standardize=True")

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        wxy = self.wins[self.idx[i]]  # (TOTAL, D) view
        warm = np.ascontiguousarray(wxy[: self.W], dtype=np.float32)
        x    = np.ascontiguousarray(wxy[self.W : self.W + self.X], dtype=np.float32)
        y    = np.ascontiguousarray(wxy[self.W + 1 : self.W + self.X + 1], dtype=np.float32)

        if self.standardize:
            warm = (warm - self.mean) / self.std
            x    = (x    - self.mean) / self.std
            y    = (y    - self.mean) / self.std

        return torch.from_numpy(warm), torch.from_numpy(x), torch.from_numpy(y)

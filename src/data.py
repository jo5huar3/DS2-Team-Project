import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from typing import Sequence, Optional, Tuple

class LorenzStartsDataset(Dataset):
    """
    (Memory-light) Dataset that yields (warm, x, y) by slicing from the base series
    using precomputed start indices. No huge window tensor is created.
    """
    def __init__(self, series, starts, warm_length, x_length, y_length=1,
                 standardize=True, mean=None, std=None):
        self.series = series  # (T, D) float32
        self.starts = np.asarray(starts, dtype=np.int64)
        self.W = int(warm_length)
        self.X = int(x_length)
        self.Y = int(y_length)
        self.total = self.W + self.X + self.Y
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

class ToSkorchXY(Dataset):
    """
    Wraps a dataset that yields (warm, x, y) into ((warm, x), y)
    so skorch can pass (warm, x) into module.forward(warm, x).
    """
    def __init__(self, base):
        self.base = base
    def __len__(self): 
        return len(self.base)
    def __getitem__(self, i):
        warm, x, y = self.base[i]  # warm:(W,D), x:(X,D), y:(X,D)
        window = torch.cat([warm, x], dim=0)  # (W+X, D)
        target = y[-1]                        # (D,)
        return window, target
    
class RandomWindowDataset(Dataset):
    """
    Memory-light dataset that slices from `series` on-the-fly.

    Each item returns:
        warm: (warm_len, D)
        x:    (x_len, D)
        y:    (y_len, D)
    """
    def __init__(self, series: np.ndarray, starts: Sequence[int],
                    warm_len: int, x_len: int, y_len: int):
        self.series = np.asarray(series)
        self.starts = np.asarray(starts, dtype=np.int64)
        self.W = int(warm_len); self.X = int(x_len); self.Y = int(y_len)
        self.total = self.W + self.X + self.Y

        T = self.series.shape[0]
        if T < self.total:
            raise ValueError(f"series too short for total window={self.total}")

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, idx: int):
        i0 = int(self.starts[idx])
        sl = self.series[i0 : i0 + self.total]
        #warm = sl[:self.W]
        x    = sl[:self.W+self.X]
        y    = sl[self.W+self.X:self.W+self.X+self.Y]
        # torch tensors (float32)
        return (
            #torch.from_numpy(warm.astype(np.float32)),
            torch.from_numpy(x.astype(np.float32)),
            torch.from_numpy(y.astype(np.float32)),
        )
    
def sample_random_starts(
    series_len: int,
    warm_len: int,
    x_len: int,
    y_len: int,
    n_samples: int = 2000,
    seed: Optional[int] = 42,
) -> np.ndarray:
    """
    Randomly sample start indices for sliding windows of total length
    TOTAL = warm_len + x_len + y_len. Samples without replacement if possible,
    otherwise falls back to with-replacement.

    Returns:
        starts: (n_samples,) int array of absolute start indices
    """
    rng = np.random.default_rng(seed)
    total = warm_len + x_len + y_len
    max_start = series_len - total
    if max_start < 0:
        raise ValueError(f"Series too short for TOTAL={total} window.")
    valid_starts = np.arange(0, max_start + 1, dtype=np.int64)

    replace = n_samples > len(valid_starts)
    starts = rng.choice(valid_starts, size=n_samples, replace=replace)
    return starts

def split_train_val(
    starts: np.ndarray, train_frac: float = 0.8, seed: Optional[int] = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Deterministic shuffle + split of the sampled starts.
    """
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(starts))
    n_train = int(round(train_frac * len(starts)))
    train_idx = perm[:n_train]
    val_idx   = perm[n_train:]
    return starts[train_idx], starts[val_idx]
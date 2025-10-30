import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import pandas as pd

def load_series(pkl_file):
    obj = pd.read_pickle(pkl_file)
    arr = obj.values if isinstance(obj, pd.DataFrame) else np.asarray(obj)
    if arr.ndim == 1:
        arr = arr[:, None]
    return arr.astype(np.float32, copy=False)

def compute_stats(series, mean=None, std=None):
    if mean is None:
        mean = series.mean(axis=0).astype(np.float32)
    if std is None:
        std  = series.std(axis=0).astype(np.float32)
    std = np.where(std == 0, 1.0, std)
    return mean, std

def compute_starts(T, warm_length, x_length, step=1):
    total = warm_length + x_length + 1
    if T <= total:
        raise ValueError(f"Need > {total} time steps, got {T}")
    # valid start indices so that [s : s+total] fits in [0..T)
    return np.arange(0, T - total + 1, step, dtype=np.int64)

def build_windows(series, warm_length: int, x_length: int):
    """
    Return a VIEW with shape (N, warm_length + x_length + 1, D).
    Guarantees axis-1 is TOTAL and axis-2 is D.
    """
    series = np.asarray(series, dtype=np.float32)
    if series.ndim == 1:
        series = series[:, None]
    total = warm_length + x_length + 1
    if series.shape[0] <= total:
        raise ValueError(f"Need > {total} time steps, got {series.shape[0]}")

    wins = sliding_window_view(series, window_shape=total, axis=0)  # nominally (N, TOTAL, D)

    # If someone changed it elsewhere and it's (N, D, TOTAL), normalize it:
    if wins.ndim != 3:
        raise ValueError(f"Expected 3D windows, got {wins.shape}")
    if wins.shape[1] == total:
        return wins                               # (N, TOTAL, D)
    if wins.shape[2] == total:
        return np.swapaxes(wins, 1, 2)            # (N, TOTAL, D)

    raise ValueError(f"Windows have unexpected shape {wins.shape} (no axis equals TOTAL={total})")

def split_indices(idx_or_n, train_ratio=0.8, seed=42):
    rng = np.random.default_rng(seed)
    if isinstance(idx_or_n, (int, np.integer)):
        base = np.arange(int(idx_or_n))
    else:
        base = np.asarray(idx_or_n)
    perm = rng.permutation(len(base))
    cut = int(train_ratio * len(base))
    return base[perm[:cut]], base[perm[cut:]]

# Dataset builder for Lorenz63 → sequences for ESN/RNN
import numpy as np
from scipy.integrate import solve_ivp
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader
import torch

# --- 1) Simulate Lorenz63 ---
def lorenz63(t, state, sigma=10.0, rho=28.0, beta=8/3):
    x, y, z = state
    return [sigma*(y-x), x*(rho - z) - y, x*y - beta*z]

def simulate_lorenz63(t0=0.0, t1=100.0, n_points=50000, state0=(1.0, 1.0, 1.0)):
    t_eval = np.linspace(t0, t1, n_points, dtype=np.float64)
    sol = solve_ivp(lorenz63, (t0, t1), state0, t_eval=t_eval, rtol=1e-9, atol=1e-12)
    # (T, 3)
    return t_eval, sol.y.T.astype(np.float32)

# --- 2) Make supervised sequences from a multivariate series ---
def make_sliding_windows(series, window=50, horizon=1, step=1, target_all_dims=True):
    """
    series: (T, D) array
    window: past length
    horizon: predict series[t + horizon]
    step: sliding step
    target_all_dims: if True, predict D-dim next state; else predict last dim only
    """
    T, D = series.shape
    Xs, ys = [], []
    for start in range(0, T - window - horizon + 1, step):
        end = start + window
        Xs.append(series[start:end])                     # (window, D)
        target = series[end + horizon - 1]              # (D,)
        ys.append(target if target_all_dims else target[-1:])
    X = np.stack(Xs, axis=0)                            # (N, window, D)
    y = np.stack(ys, axis=0)                            # (N, D) or (N, 1)
    return X.astype(np.float32), y.astype(np.float32)

# --- 3) Build loaders (time-aware split, scale on train only) ---
def build_loaders_from_series(series, window=50, horizon=1, batch_size=128,
                              split=(0.7, 0.15, 0.15), target_all_dims=True,
                              shuffle_train=False):
    """
    series: (T, D) raw series (no scaling)
    Returns train/val/test DataLoaders and the fitted scaler.
    """
    # Respect temporal order for splits
    T = len(series)
    n_train = int(T * split[0])
    n_val   = int(T * split[1])
    n_test  = T - n_train - n_val

    series_train = series[:n_train]
    series_val   = series[n_train:n_train+n_val]
    series_test  = series[n_train+n_val:]

    # Standardize features (fit on train, apply to all)
    scaler = StandardScaler().fit(series_train)
    train_std = scaler.transform(series_train)
    val_std   = scaler.transform(series_val)
    test_std  = scaler.transform(series_test)

    # Build supervised windows
    Xtr, ytr = make_sliding_windows(train_std, window, horizon, target_all_dims=target_all_dims)
    Xva, yva = make_sliding_windows(val_std,   window, horizon, target_all_dims=target_all_dims)
    Xte, yte = make_sliding_windows(test_std,  window, horizon, target_all_dims=target_all_dims)

    # Torch datasets/loaders
    train_ds = TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(ytr))
    val_ds   = TensorDataset(torch.from_numpy(Xva), torch.from_numpy(yva))
    test_ds  = TensorDataset(torch.from_numpy(Xte), torch.from_numpy(yte))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=shuffle_train, drop_last=False)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, drop_last=False)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, drop_last=False)

    return train_loader, val_loader, test_loader, scaler

# --- 4) End-to-end example ---
if __name__ == "__main__":
    # Simulate (discard initial transient so dynamics settle)
    t, S = simulate_lorenz63(t0=0.0, t1=100.0, n_points=50000, state0=(1.0, 1.0, 1.0))
    transient = 5000
    S = S[transient:]      # shape (T, 3), T ~ 45000

    # Create loaders for next-step prediction with a 100-step context window
    window = 100
    horizon = 1
    train_loader, val_loader, test_loader, scaler = build_loaders_from_series(
        S, window=window, horizon=horizon, batch_size=512, target_all_dims=True
    )

    # Peek at shapes
    xb, yb = next(iter(train_loader))
    print("X batch:", xb.shape, "Y batch:", yb.shape)    # (B, window, 3), (B, 3)

    # Optional: save arrays for non-PyTorch ESN workflows (e.g., NumPy ESN)
    # Note: These are standardized values.
    # Rebuild arrays from loaders for convenience:
    def stack_loader(dl):
        Xs, ys = [], []
        for X, y in dl:
            Xs.append(X.numpy())
            ys.append(y.numpy())
        return np.concatenate(Xs), np.concatenate(ys)

    Xtr_np, ytr_np = stack_loader(train_loader)
    Xva_np, yva_np = stack_loader(val_loader)
    Xte_np, yte_np = stack_loader(test_loader)

    np.savez_compressed(
        "lorenz63_esn_data.npz",
        X_train=Xtr_np, y_train=ytr_np,
        X_val=Xva_np,   y_val=yva_np,
        X_test=Xte_np,  y_test=yte_np,
        window=np.array([window]), horizon=np.array([horizon])
    )
    print("Saved: lorenz63_esn_data.npz")

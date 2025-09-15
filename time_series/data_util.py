import numpy as np
from sklearn.preprocessing import StandardScaler

def lorenz63_rhs(state, sigma=10.0, rho=28.0, beta=8.0/3.0):
    x, y, z = state
    dx = sigma * (y - x)
    dy = x * (rho - z) - y
    dz = x * y - beta * z
    return np.array([dx, dy, dz], dtype=np.float64)

def rk4_step(f, state, dt):
    k1 = f(state)
    k2 = f(state + 0.5 * dt * k1)
    k3 = f(state + 0.5 * dt * k2)
    k4 = f(state + dt * k3)
    return state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

def simulate_lorenz63(T=12000, dt=0.01, x0=(1.0, 1.0, 1.0)):
    out = np.zeros((T, 3), dtype=np.float64)
    s = np.array(x0, dtype=np.float64)
    for t in range(T):
        out[t] = s
        s = rk4_step(lorenz63_rhs, s, dt)
    return out  # shape (T, 3)

# ------------------------------------
# 2) Sliding-window dataset utilities
# ------------------------------------
def make_sliding_windows(data, seq_len=20, horizon=1, washout=200):
    """
    data: (T, 3) Lorenz63 states
    seq_len: input window length (teacher-forced)
    horizon: predict t+1 ... t+horizon (here we use 1 for next-step)
    washout: drop initial transient
    Returns:
      X: (N, seq_len, 3)
      y: (N, 3)  # next-step target
    """
    D = data.astype(np.float32)
    D = D[washout:]  # drop transients
    T = len(D)
    X_list, y_list = [], []
    for t in range(T - seq_len - horizon + 1):
        X_list.append(D[t:t+seq_len, :])
        y_list.append(D[t+seq_len + horizon - 1, :])  # next-step for horizon=1
    X = np.stack(X_list, axis=0)
    y = np.stack(y_list, axis=0)
    return X, y  # (N, L, 3), (N, 3)

def standardize_sequences(X_train, X_test, y_train, y_test):
    """
    Standardize features per-variable using only training statistics.
    Works by reshaping sequences to 2D, fitting scalers, and reshaping back.
    """
    Ntr, L, F = X_train.shape
    Nte = X_test.shape[0]

    x_scaler = StandardScaler()
    y_scaler = StandardScaler()

    Xtr2 = X_train.reshape(-1, F)
    Xte2 = X_test.reshape(-1, F)

    Xtr2 = x_scaler.fit_transform(Xtr2)
    Xte2 = x_scaler.transform(Xte2)

    ytr2 = y_scaler.fit_transform(y_train)
    yte2 = y_scaler.transform(y_test)

    X_train_std = Xtr2.reshape(Ntr, L, F).astype(np.float32)
    X_test_std  = Xte2.reshape(Nte, L, F).astype(np.float32)
    y_train_std = ytr2.astype(np.float32)
    y_test_std  = yte2.astype(np.float32)
    return X_train_std, X_test_std, y_train_std, y_test_std, x_scaler, y_scaler


if __name__ == "__main__":
    pass
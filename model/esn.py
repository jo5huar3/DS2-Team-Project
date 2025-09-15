# esn_lorenz63_skorch_grid.py
import math, time, os
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_squared_error
from skorch import NeuralNetRegressor
from time_series.data_util import lorenz63_rhs, rk4_step, simulate_lorenz63


# ----------------------------------------
# ESN Module: fixed reservoir + readout
# ----------------------------------------
class ESNRegressor(nn.Module):
    def __init__(self,
                 input_size=3,
                 reservoir_size=500,
                 output_size=3,
                 leak=0.7,
                 spectral_radius=0.9,
                 input_scale=0.8,
                 bias_scale=0.0,
                 sparsity=0.05,
                 activation='tanh',
                 seed=42):
        super().__init__()
        self.input_size = input_size
        self.reservoir_size = reservoir_size
        self.output_size = output_size
        self.leak = leak
        self.spectral_radius = spectral_radius
        self.input_scale = input_scale
        self.bias_scale = bias_scale
        self.sparsity = sparsity
        self.seed = seed

        if activation == 'tanh':
            self.act = torch.tanh
        elif activation == 'relu':
            self.act = torch.relu
        else:
            raise ValueError("activation must be 'tanh' or 'relu'")

        g = torch.Generator(device='cpu')
        g.manual_seed(seed)

        # Input weights (N_res, N_in) in [-1, 1]
        W_in = (torch.rand((self.reservoir_size, self.input_size), generator=g) * 2.0 - 1.0)
        W_in *= self.input_scale

        # Bias in [-1, 1]
        b = (torch.rand((self.reservoir_size,), generator=g) * 2.0 - 1.0) * self.bias_scale

        # Sparse recurrent W (N_res, N_res)
        W = torch.zeros(self.reservoir_size, self.reservoir_size)
        nnz = max(1, int(self.sparsity * self.reservoir_size * self.reservoir_size))
        row_idx = torch.randint(self.reservoir_size, (nnz,), generator=g)
        col_idx = torch.randint(self.reservoir_size, (nnz,), generator=g)
        vals = (torch.rand((nnz,), generator=g) * 2.0 - 1.0)
        W[row_idx, col_idx] = vals

        # Scale to desired spectral radius – compute eigvals on CPU in double
        with torch.no_grad():
            W_cpu = W.to(dtype=torch.float64, device='cpu')
            eigvals = torch.linalg.eigvals(W_cpu).abs().real
            max_ev = float(torch.max(eigvals)) if torch.any(eigvals > 0) else 1.0
            if max_ev == 0.0:
                max_ev = 1.0
            W = (W * (self.spectral_radius / max_ev)).to(dtype=torch.float32)

        self.register_buffer('W_in', W_in)
        self.register_buffer('W', W)
        self.register_buffer('b', b)


        # Trainable linear readout
        self.readout = nn.Linear(self.reservoir_size, self.output_size, bias=True)

    def forward(self, X):
        """
        X: (batch, seq_len, input_size=3)
        returns: (batch, output_size=3) for the last step
        """
        B, L, F = X.shape
        assert F == self.input_size

        r = X.new_zeros(B, self.reservoir_size)
        W, W_in, b = self.W, self.W_in, self.b
        alpha = self.leak
        act = self.act

        for t in range(L):
            x_t = X[:, t, :]  # (B, F)
            drive = x_t @ W_in.t() + r @ W.t() + b   # (B, N_res)
            r = (1 - alpha) * r + alpha * act(drive)

        y_hat = self.readout(r)
        return y_hat

# -----------------------------
# k-step free-run evaluation
# -----------------------------
@torch.no_grad()
def rollout_free_run(model_or_search, x0_seq_orig, k, x_scaler, y_scaler):
    """
    model_or_search: skorch NeuralNetRegressor *or* sklearn GridSearchCV wrapping it
    x0_seq_orig: (L, 3) numpy array in ORIGINAL scale for the starting teacher-forced window
    k: number of closed-loop steps to forecast
    Returns:
      preds_orig: (k, 3) numpy array in original scale
    """
    # Get the fitted skorch net regardless of whether we received GridSearchCV
    net = getattr(model_or_search, "best_estimator_", model_or_search)

    # Standardize the starting window with the input scaler
    x_win_std = x_scaler.transform(x0_seq_orig.astype(np.float32))  # (L, 3)

    preds = []
    for _ in range(k):
        # skorch .predict expects numpy; shape (1, L, 3)
        x_batch_std = x_win_std[None, :, :]  # (1, L, 3)
        y_step_std = net.predict(x_batch_std)  # (1, 3) standardized target space (same vars)
        # Back to original scale
        y_step_orig = y_scaler.inverse_transform(y_step_std)  # (1, 3)
        preds.append(y_step_orig[0])

        # Feed the prediction back as next input (standardize with x_scaler)
        next_in_std = x_scaler.transform(y_step_orig)  # (1, 3)
        x_win_std = np.vstack([x_win_std[1:], next_in_std])  # slide window

    return np.stack(preds, axis=0)  # (k, 3)


# ------------------------------------
# Sliding-window dataset utilities
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
    print(f"""\
        Ntr: {Ntr}\n\
        L: {L}\n\
        F: {F}""") 

    x_scaler = StandardScaler()
    y_scaler = StandardScaler()

    Xtr2 = X_train.reshape(-1, F)
    Xte2 = X_test.reshape(-1, F)
    #print(Xtr2.shape)
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
    rng = np.random.default_rng(0)
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"


    # Simulate
    data = simulate_lorenz63(T=15000, dt=0.01, x0=(1.0, 1.0, 1.0))

    # Build dataset
    SEQ_LEN = 20     # teacher-forced context length
    HORIZON = 1      # next-step prediction
    WASHOUT = 500
    X, y = make_sliding_windows(data, seq_len=SEQ_LEN, horizon=HORIZON, washout=WASHOUT)

    # Train/test split without shuffling (time order)
    N_test = 3000
    X_train, X_test = X[:-N_test], X[-N_test:]
    y_train, y_test = y[:-N_test], y[-N_test:]

    # Standardize using training stats
    X_train_std, X_test_std, y_train_std, y_test_std, x_scaler, y_scaler = standardize_sequences(
        X_train, X_test, y_train, y_test
    )

    # skorch wrapper (no inner validation split; CV handles it)
    net = NeuralNetRegressor(
        ESNRegressor,
        module__input_size=3,
        module__output_size=3,
        criterion=nn.MSELoss,
        optimizer=torch.optim.Adam,
        optimizer__lr=2e-3,
        max_epochs=30,
        batch_size=256,
        train_split=None,
        device=device,
        iterator_train__shuffle=False,
    )
    net.set_params(verbose=0)
    # Parameter grid: tune ESN dynamics + size
    param_grid = {
        'module__reservoir_size': [300, 500, 700, 1000],       # was [400, 800]
        'module__leak': [0.6, 0.9],                 # trim
        'module__spectral_radius': [0.85, 1.0],     # trim
        'module__input_scale': [0.6, 1.0],          # trim
        'module__sparsity': [0.03],                 # single value for speed
        'optimizer__lr': [2e-3],                    # single value for speed
        'max_epochs': [12],                         # was [25, 40]
    }
    '''
    param_grid = {
    # reservoir dynamics
    'module__reservoir_size': [300, 500, 700],           # capacity 3
    'module__leak': [0.5, 0.7, 0.9, 1.0],                # integration speed 4
    'module__spectral_radius': [0.7, 0.85, 1.0, 1.1],    # memory/ESP 4
    'module__input_scale': [0.4, 0.6, 0.9, 1.2],         # drive strength 4
    'module__sparsity': [0.02, 0.03, 0.05],              # connectivity 3
    'module__bias_scale': [0.0, 0.1, 0.2],               # bias drive 3
    'module__activation': ['tanh', 'relu'],              # nonlinearity 2

    # optimization/regularization for readout (Adam)
    'optimizer__lr': [1e-3, 2e-3, 3e-3],                 # 3
    'optimizer__weight_decay': [0.0, 1e-4, 1e-3],        # L2 ~ ridge 3
    'batch_size': [256, 512, 1024],                      # 3
    'max_epochs': [12, 25],                              # 2

    # robustness to random reservoir draws
    'module__seed': [42, 123, 314],
    }'''
    cv = TimeSeriesSplit(n_splits=3)
    
    gs = GridSearchCV(
        net,
        param_grid=param_grid,
        scoring='neg_mean_squared_error',  # on standardized targets
        cv=2,
        refit=True,
        n_jobs=-1,
        verbose=0,
    )

    gs = RandomizedSearchCV(net, param_distributions=param_grid, n_iter=100,
                                cv=cv, scoring='neg_mean_squared_error',
                                n_jobs=-1, verbose=0, random_state=0)

    start_time = time.time()
    # Fit on standardized data
    gs.fit(X_train_std, y_train_std)

    print("Best params:", gs.best_params_)
    print("Best CV MSE (std space):", -gs.best_score_)

    # One-step test evaluation
    y_pred_std = gs.predict(X_test_std).astype(np.float32)
    y_pred = y_scaler.inverse_transform(y_pred_std)
    test_mse = mean_squared_error(y_test, y_pred)
    print("One-step Test MSE (original scale):", test_mse)

  # Choose starting window
    x0_seq = X_test[0]  # (SEQ_LEN, 3) in ORIGINAL scale

    K = 10  # steps to forecast
    preds = rollout_free_run(gs, x0_seq, K, x_scaler, y_scaler)

    # Build ground truth for comparison.
    # let train_windows = number of training windows (i.e., len(X_train))
    train_windows = X_train.shape[0]
    SEQ_LEN = X_test.shape[1]  # or your constant
    WASHOUT = WASHOUT          # keep your constant
    # In make_sliding_windows, the window starting at index t uses data[t : t+SEQ_LEN]
    # and the first predicted "next" state is at data[t+SEQ_LEN].
    t0_global = WASHOUT + train_windows  # this is where X_test[0] begins
    gt_start = t0_global + SEQ_LEN       # first next-step after the window
    gt = data[gt_start : gt_start + K]   # (K, 3) in original scale

    from sklearn.metrics import mean_squared_error
    rollout_mse = mean_squared_error(gt, preds)
    print(f"{K}-step Free-run MSE (original scale): {rollout_mse:.6f}")

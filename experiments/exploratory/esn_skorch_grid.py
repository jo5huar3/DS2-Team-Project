#Best params: {'max_epochs': 12, 'module__input_scale': 0.6, 'module__leak': 0.9, 'module__reservoir_size': 300, 'module__sparsity': 0.03, 'module__spectral_radius': 1.0, 'optimizer__lr': 0.002}
# esn_lorenz63_skorch_grid.py
import time
import numpy as np
import torch
import torch.nn as nn

from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from skorch import NeuralNetRegressor
from time_series.data_util import *



# ----------------------------------------
# 3) ESN Module: fixed reservoir + readout
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
# 4) k-step free-run evaluation
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

def per_dim_rmse(y_true, y_pred, eps=1e-12):
    mse = np.mean((y_true - y_pred) ** 2, axis=0)
    return np.sqrt(np.maximum(mse, 0.0) + eps)

# ---------------
# 5) Main routine
# ---------------
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
# Best params: {'max_epochs': 12, 'module__input_scale': 0.6, 'module__leak': 0.9, 'module__reservoir_size': 300, 'module__sparsity': 0.03, 'module__spectral_radius': 1.0, 'optimizer__lr': 0.002}
    best_params = {
        "reservoir_size": 300,
        "leak": 0.9,
        "spectral_radius": 1.0,
        "input_scale": 0.6,
        "sparsity": 0.03,
        "bias_scale": 0.0,         # unchanged
        "activation": "tanh",
        "seed": 42,
    }

    net = NeuralNetRegressor(
        ESNRegressor,
        module__input_size=3,
        module__output_size=3,
        module__reservoir_size=best_params["reservoir_size"],
        module__leak=best_params["leak"],
        module__spectral_radius=best_params["spectral_radius"],
        module__input_scale=best_params["input_scale"],
        module__sparsity=best_params["sparsity"],
        module__bias_scale=best_params["bias_scale"],
        module__activation=best_params["activation"],
        module__seed=best_params["seed"],
        criterion=nn.MSELoss,
        optimizer=torch.optim.Adam,
        optimizer__lr=2e-3,        # from your best config
        max_epochs=12,             # from your best config
        batch_size=1024,           # large batch for speed
        train_split=None,          # no internal val split
        device=device,
        iterator_train__shuffle=False,
    )

    # --------------------
    # 3) Train the model
    # --------------------
    t0 = time.time()
    net.fit(X_train_std, y_train_std)
    train_secs = time.time() - t0
    print(f"Fit time: {train_secs:.2f}s")

    # ----------------------
    # 4) One-step evaluation
    # ----------------------
    y_pred_std = net.predict(X_test_std).astype(np.float32)
    y_pred = y_scaler.inverse_transform(y_pred_std)

    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    rmse_dim = per_dim_rmse(y_test, y_pred)
    rmse_overall = float(np.sqrt(mse))

    print("\n=== One-step metrics (original scale) ===")
    print(f"MSE:  {mse:.6f}")
    print(f"RMSE: {rmse_overall:.6f}")
    print(f"MAE:  {mae:.6f}")
    print(f"R^2:  {r2:.6f}")
    print(f"Per-dim RMSE [x, y, z]: {rmse_dim}")

    # -----------------------------------------------
    # 5) k-step free-run rollout and its test MSE
    # -----------------------------------------------
    K = 50  # steps to forecast closed-loop
    x0_seq = X_test[0]  # ORIGINAL scale starting window

    preds = rollout_free_run(net, x0_seq, K, x_scaler, y_scaler)

    # Build ground truth aligned with the chosen window
    # If X_test[0] corresponds to data indices [t0 : t0+SEQ_LEN-1],
    # the first “next” state (target) is at data[t0+SEQ_LEN].
    train_windows = X_train.shape[0]
    t0_global = WASHOUT + train_windows       # where X_test[0] starts in 'data'
    gt_start = t0_global + SEQ_LEN            # first next-step after the window
    gt = data[gt_start : gt_start + K]        # shape (K, 3)

    rollout_mse = mean_squared_error(gt, preds)
    rollout_rmse = float(np.sqrt(rollout_mse))
    print("\n=== Closed-loop rollout metrics (original scale) ===")
    print(f"{K}-step MSE:  {rollout_mse:.6f}")
    print(f"{K}-step RMSE: {rollout_rmse:.6f}")


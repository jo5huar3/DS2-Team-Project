# train_esn_lstm.py

import numpy as np
import torch
import torch.nn as nn
from skorch import NeuralNetRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# your helpers (adjust imports to your layout):
from time_series.data_util import simulate_lorenz63, make_sliding_windows, standardize_sequences
from model.esn_lstm_readout import ESNRegressorLSTM

np.random.seed(0)
torch.manual_seed(0)

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# ----- data -----
data = simulate_lorenz63(T=15000, dt=0.01, x0=(1.0, 1.0, 1.0))
SEQ_LEN, HORIZON, WASHOUT = 20, 1, 500
X, y = make_sliding_windows(data, seq_len=SEQ_LEN, horizon=HORIZON, washout=WASHOUT)

N_test = 3000
X_train, X_test = X[:-N_test], X[-N_test:]
y_train, y_test = y[:-N_test], y[-N_test:]

X_train_std, X_test_std, y_train_std, y_test_std, x_scaler, y_scaler = standardize_sequences(
    X_train, X_test, y_train, y_test
)

# ----- model (best ESN params + LSTM readout) -----
net = NeuralNetRegressor(
    ESNRegressorLSTM,
    module__input_size=3,
    module__output_size=3,
    # ESN dynamics (your best):
    module__reservoir_size=300,
    module__leak=0.9,
    module__spectral_radius=1.0,
    module__input_scale=0.6,
    module__sparsity=0.03,
    module__bias_scale=0.0,
    module__activation='tanh',
    module__seed=42,
    # LSTM readout hyperparams (tune if you like):
    module__lstm_hidden=64,
    module__lstm_layers=1,
    module__lstm_dropout=0.0,
    module__bidirectional=False,

    criterion=nn.MSELoss,
    optimizer=torch.optim.Adam,
    optimizer__lr=2e-3,      # you can keep your 2e-3
    max_epochs=12,           # start modest; you may need more than linear readout
    batch_size=512,          # BPTT over L=20 is cheap; larger batch is fine
    train_split=None,
    device=device,
    iterator_train__shuffle=False,
)

# (optional) clip gradients to stabilize BPTT on LSTM
from skorch.callbacks import GradientNormClipping
net.initialize()
net.set_params(callbacks__grad_clip=GradientNormClipping(gradient_clip_value=1.0, gradient_clip_norm_type=2))
# fit
net.fit(X_train_std, y_train_std)

# ----- one-step metrics -----
y_pred_std = net.predict(X_test_std).astype(np.float32)
y_pred = y_scaler.inverse_transform(y_pred_std)

mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = float(np.sqrt(mse))
r2 = r2_score(y_test, y_pred)
per_dim_rmse = np.sqrt(np.mean((y_test - y_pred) ** 2, axis=0))

print("\n=== One-step metrics (original scale) ===")
print(f"MSE:  {mse:.6f}")
print(f"RMSE: {rmse:.6f}")
print(f"MAE:  {mae:.6f}")
print(f"R^2:  {r2:.6f}")
print(f"Per-dim RMSE [x, y, z]: {per_dim_rmse}")

# ----- optional: closed-loop rollout utility -----
@torch.no_grad()
def rollout_free_run(net_or_search, x0_seq_orig, k, x_scaler, y_scaler):
    net = getattr(net_or_search, "best_estimator_", net_or_search)
    x_win_std = x_scaler.transform(x0_seq_orig.astype(np.float32))
    preds = []
    for _ in range(k):
        y_std = net.predict(x_win_std[None, :, :])            # (1, 3)
        y = y_scaler.inverse_transform(y_std)                 # (1, 3)
        preds.append(y[0])
        x_win_std = np.vstack([x_win_std[1:], x_scaler.transform(y)])  # append
    return np.stack(preds, axis=0)

K = 1000
x0_seq = X_test[0]
preds = rollout_free_run(net, x0_seq, K, x_scaler, y_scaler)

# align GT with first test window
train_windows = X_train.shape[0]
t0_global = WASHOUT + train_windows
gt_start = t0_global + SEQ_LEN
gt = data[gt_start:gt_start + K]

rollout_mse = mean_squared_error(gt, preds)
rollout_rmse = float(np.sqrt(rollout_mse))
print("\n=== Closed-loop rollout metrics (original scale) ===")
print(f"{K}-step MSE:  {rollout_mse:.6f}")
print(f"{K}-step RMSE: {rollout_rmse:.6f}")

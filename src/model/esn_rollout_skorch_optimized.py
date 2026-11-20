
import math
from dataclasses import dataclass
from typing import Tuple, Optional, Dict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from skorch import NeuralNetRegressor
from skorch.dataset import unpack_data
from skorch.utils import to_device
from skorch.helper import predefined_split


# ============================================================
# Utilities
# ============================================================

def set_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_windows(series: np.ndarray, warm: int, xlen: int, horizon: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create rolling windows for warm-up (W), teacher-forced (X), and rollout targets (K).
    series: (T, D) standardized float array
    returns: warm (N,W,D), teach (N,X,D), roll (N,K,D)
    """
    T, D = series.shape
    total = warm + xlen + horizon
    N = T - total
    if N <= 0:
        raise ValueError(f"Series too short: need > {total} steps, got {T}")
    warm_seqs = np.stack([series[i:i+warm] for i in range(N)], axis=0)
    teach_seqs = np.stack([series[i+warm:i+warm+xlen] for i in range(N)], axis=0)
    roll_tgts = np.stack([series[i+warm+xlen:i+total] for i in range(N)], axis=0)
    return warm_seqs.astype(np.float32), teach_seqs.astype(np.float32), roll_tgts.astype(np.float32)


class RolloutDataset(Dataset):
    """
    Skorch-friendly dataset that returns a dict X and a dummy y.
    X = {'warm': (W,D), 'teacher': (X,D), 'roll': (K,D)}
    """
    def __init__(self, warm: np.ndarray, teacher: np.ndarray, roll_tgt: np.ndarray):
        assert warm.shape[0] == teacher.shape[0] == roll_tgt.shape[0]
        self.warm = torch.from_numpy(warm)       # (N,W,D)
        self.teacher = torch.from_numpy(teacher) # (N,X,D)
        self.roll = torch.from_numpy(roll_tgt)   # (N,K,D)

    def __len__(self):
        return self.warm.shape[0]

    def __getitem__(self, i):
        X = {'warm': self.warm[i], 'teacher': self.teacher[i], 'roll': self.roll[i]}
        y = 0.0  # not used; loss computed internally
        return X, y


# ============================================================
# ESN Module (sparse reservoir) with single readout head
# ============================================================

def _power_iteration_spectral_radius(W_dense: torch.Tensor, iters: int = 50) -> float:
    device = W_dense.device
    N = W_dense.shape[0]
    v = torch.randn(N, device=device)
    v = v / (v.norm() + 1e-12)
    for _ in range(iters):
        v = W_dense @ v
        nrm = v.norm() + 1e-12
        v = v / nrm
    rv = v @ (W_dense @ v)
    return float(torch.abs(rv).item())


class ESNForecaster(nn.Module):
    """
    Echo State Network with a single linear readout (trainable) and fixed reservoir.
    Uses sparse CSR matrix for the reservoir for speed.
    Exposes:
      - forward(B,T,D) -> (y_seq, {'r': r_T})
      - step(B,1,D)    -> (y_next, {'r': r_t})
    """
    def __init__(self,
                 input_size: int,
                 reservoir_size: int = 400,
                 leak: float = 0.7,
                 spectral_radius: float = 0.9,
                 input_scale: float = 0.8,
                 bias_scale: float = 0.0,
                 sparsity: float = 0.05,
                 activation: str = 'tanh',
                 seed: int = 42,
                 use_sparse: bool = True):
        super().__init__()
        set_seed(seed)
        self.input_size = int(input_size)
        self.reservoir_size = int(reservoir_size)
        self.leak = float(leak)
        self.act = torch.tanh if activation == 'tanh' else torch.relu
        self.use_sparse = use_sparse

        # Fixed input weights
        W_in = torch.empty(self.reservoir_size, self.input_size).normal_(0, 1.0)
        W_in *= input_scale / math.sqrt(self.input_size)
        self.register_buffer('W_in', W_in)

        # Build sparse reservoir W
        W_dense = torch.zeros(self.reservoir_size, self.reservoir_size)
        mask = torch.rand_like(W_dense) < sparsity
        W_dense[mask] = torch.randn_like(W_dense[mask])
        est = _power_iteration_spectral_radius(W_dense, iters=50)
        if est > 1e-12:
            W_dense *= spectral_radius / est

        if use_sparse:
            W_csr = W_dense.to_sparse_csr()
            self.register_buffer('W_sparse', W_csr)
            self.register_buffer('W', W_dense)  # keep a dense copy for CPU fallback
        else:
            self.register_buffer('W', W_dense)

        b = torch.empty(self.reservoir_size).normal_(0, 1.0) * bias_scale
        self.register_buffer('bias', b)

        # Trainable single readout head
        self.readout = nn.Linear(self.reservoir_size, self.input_size, bias=True)

    def _dense_res_mult(self, r_prev: torch.Tensor) -> torch.Tensor:
        # r_prev: (B, N)  -> (B, N)  via r_prev @ W^T
        return r_prev @ self.W.T

    def _sparse_res_mult(self, r_prev: torch.Tensor) -> torch.Tensor:
        # Use (W @ r_prev^T)^T which is sparse@dense
        # W_sparse: (N,N), r_prev: (B,N)
        return (self.W_sparse @ r_prev.T).T

    def _reservoir_step(self, r_prev: torch.Tensor, x_t: torch.Tensor) -> torch.Tensor:
        drive = x_t @ self.W_in.T + self.bias  # (B, N)
        if self.use_sparse:
            rec = self._sparse_res_mult(r_prev)  # (B,N)
        else:
            rec = self._dense_res_mult(r_prev)   # (B,N)
        pre = drive + rec
        r_t = (1.0 - self.leak) * r_prev + self.leak * self.act(pre)
        return r_t

    def step(self, x_t: torch.Tensor, h: Optional[Dict[str, torch.Tensor]] = None):
        B, one, D = x_t.shape
        assert one == 1 and D == self.input_size
        if h is None or ('r' not in h):
            r_prev = torch.zeros(B, self.reservoir_size, device=x_t.device, dtype=x_t.dtype)
        else:
            r_prev = h['r']
        xt = x_t[:, 0, :]                             # (B,D)
        r_t = self._reservoir_step(r_prev, xt)        # (B,N)
        y_next = self.readout(r_t).unsqueeze(1)       # (B,1,D)
        return y_next, {'r': r_t}

    def forward(self, x_seq: torch.Tensor, h: Optional[Dict[str, torch.Tensor]] = None):
        B, T, D = x_seq.shape
        device = x_seq.device
        if h is None or ('r' not in h):
            r = torch.zeros(B, self.reservoir_size, device=device, dtype=x_seq.dtype)
        else:
            r = h['r']
        outs = []
        for t in range(T):
            r = self._reservoir_step(r, x_seq[:, t, :])
            outs.append(self.readout(r).unsqueeze(1))  # (B,1,D)
        y_seq = torch.cat(outs, dim=1)                 # (B,T,D)
        return y_seq, {'r': r}


# ============================================================
# Rollout-aware losses (vectorized teacher-forced loss)
# ============================================================

@dataclass
class RolloutLossConfig:
    horizon: int = 50
    lambda_1step: float = 0.3


def rollout_losses(module: nn.Module,
                   warm: torch.Tensor,     # (B,W,D)
                   teacher: torch.Tensor,  # (B,X,D)
                   roll_tgt: torch.Tensor, # (B,K,D)
                   cfg: RolloutLossConfig) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Combined loss = lambda * 1-step MSE + (1-lambda) * K-step rollout MSE.
    Uses vectorized teacher-forced loss for speed.
    """
    mse = nn.MSELoss(reduction='mean')
    device = next(module.parameters()).device
    warm, teacher, roll_tgt = warm.to(device), teacher.to(device), roll_tgt.to(device)
    K = cfg.horizon

    # 1) warm-up (no loss)
    with torch.no_grad():
        _, h = module(warm, None)

    # 2) vectorized 1-step teacher-forced MSE
    # Predict y_t from x_t, compare to teacher[:, 1:] (next step)
    y_seq, h = module(teacher, h)                  # (B,X,D)
    if teacher.size(1) > 1:
        loss_1 = mse(y_seq[:, :-1, :], teacher[:, 1:, :])
    else:
        loss_1 = torch.zeros((), device=device)

    # 3) free-run K-step rollout MSE
    prev = teacher[:, -1:, :]                      # (B,1,D)
    preds = []
    for _ in range(K):
        pred, h = module.step(prev, h)
        preds.append(pred)
        prev = pred
    preds = torch.cat(preds, dim=1)                # (B,K,D)
    loss_roll = mse(preds, roll_tgt)

    total = cfg.lambda_1step * loss_1 + (1.0 - cfg.lambda_1step) * loss_roll
    return total, loss_1.detach(), loss_roll.detach()


# ============================================================
# Skorch subclass with rollout loss
# ============================================================

class RolloutNeuralNet(NeuralNetRegressor):
    def __init__(self, *args, horizon: int = 50, lambda_1step: float = 0.3, **kwargs):
        super().__init__(*args, **kwargs)
        self.horizon = horizon
        self.lambda_1step = lambda_1step
        # Make Jupyter responsive: keep DataLoader single-process by default
        self.set_params(iterator_train__num_workers=0,
                        iterator_valid__num_workers=0,
                        iterator_train__persistent_workers=False,
                        iterator_valid__persistent_workers=False)

    def train_step_single(self, batch, **fit_params):
        Xi, yi = unpack_data(batch)
        warm   = to_device(Xi['warm'], self.device)
        teach  = to_device(Xi['teacher'], self.device)
        roll   = to_device(Xi['roll'], self.device)

        self.module_.train()
        self.optimizer_.zero_grad(set_to_none=True)

        loss, loss_1, loss_roll = rollout_losses(
            self.module_, warm, teach, roll,
            cfg=RolloutLossConfig(horizon=self.horizon, lambda_1step=self.lambda_1step)
        )
        loss.backward()

        gcv = getattr(self, 'gradient_clip_value', None)
        if gcv is not None:
            torch.nn.utils.clip_grad_norm_(self.module_.parameters(), gcv)

        self.optimizer_.step()
        return {'loss': loss}

    @torch.no_grad()
    def validation_step(self, batch, **fit_params):
        Xi, yi = unpack_data(batch)
        warm   = to_device(Xi['warm'], self.device)
        teach  = to_device(Xi['teacher'], self.device)
        roll   = to_device(Xi['roll'], self.device)

        self.module_.eval()
        loss, loss_1, loss_roll = rollout_losses(
            self.module_, warm, teach, roll,
            cfg=RolloutLossConfig(horizon=self.horizon, lambda_1step=self.lambda_1step)
        )
        return {'loss': loss}


# ============================================================
# Example usage (can be replaced with your pipeline)
# ============================================================

def _generate_synthetic_series(T: int = 20000, D: int = 3, seed: int = 0):
    rng = np.random.default_rng(seed)
    series = rng.normal(size=(T, D)).astype(np.float32)
    for t in range(1, T):
        series[t] = 0.7 * series[t-1] + 0.3 * series[t]
    return series

def example_main():
    set_seed(0)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load data or fall back to synthetic
    try:
        from data import load_series  # optional
        series = load_series()
    except Exception:
        series = _generate_synthetic_series(T=30000, D=3, seed=0)

    T, D = series.shape

    # Train-only standardization
    t_split = int(T * 0.80)
    mu = series[:t_split].mean(axis=0)
    sd = series[:t_split].std(axis=0) + 1e-8
    series_std = (series - mu) / sd

    # Windowing
    WARM, XLEN, K = 50, 80, 50  # smaller W/X for speed on CPU
    warm, teach, roll = make_windows(series_std, WARM, XLEN, K)
    N = warm.shape[0]
    n_split = int(N * 0.80)

    warm_tr, teach_tr, roll_tr = warm[:n_split],  teach[:n_split],  roll[:n_split]
    warm_va, teach_va, roll_va = warm[n_split:], teach[n_split:], roll[n_split:]

    train_ds = RolloutDataset(warm_tr, teach_tr, roll_tr)
    valid_ds = RolloutDataset(warm_va, teach_va, roll_va)

    # Model + skorch
    net = RolloutNeuralNet(
        module=ESNForecaster,
        module__input_size=D,
        module__reservoir_size=400,     # smaller & sparse => much faster
        module__leak=0.7,
        module__spectral_radius=0.9,
        module__input_scale=0.8,
        module__bias_scale=0.0,
        module__sparsity=0.05,
        module__activation='tanh',
        module__seed=123,
        module__use_sparse=True,
        max_epochs=8,
        lr=1e-3,
        batch_size=64,
        optimizer=torch.optim.Adam,
        device=device,
        horizon=K,
        lambda_1step=0.3,
        train_split=predefined_split(valid_ds),
    )

    # Optional: reduce BLAS threads to improve interrupt responsiveness
    try:
        import os
        os.environ.setdefault('OMP_NUM_THREADS', '1')
        os.environ.setdefault('MKL_NUM_THREADS', '1')
    except Exception:
        pass

    net.fit(train_ds, y=None)

    # Quick rollout MSE on validation
    from torch.utils.data import DataLoader
    val_loader = DataLoader(valid_ds, batch_size=256, shuffle=False)
    mse = nn.MSELoss(reduction='mean')
    total_mse, total_count = 0.0, 0
    with torch.no_grad():
        for Xi, _ in val_loader:
            warm_b, teach_b, roll_b = Xi['warm'].to(device), Xi['teacher'].to(device), Xi['roll'].to(device)
            _, h = net.module_(warm_b, None)
            prev = teach_b[:, -1:, :]
            preds = []
            for _ in range(K):
                pred, h = net.module_.step(prev, h)
                preds.append(pred)
                prev = pred
            preds = torch.cat(preds, dim=1)
            loss = mse(preds, roll_b).item()
            total_mse += loss * preds.shape[0]
            total_count += preds.shape[0]
    print(f"Validation 50-step rollout MSE: {total_mse / max(1, total_count):.6f}")


if __name__ == '__main__':
    example_main()

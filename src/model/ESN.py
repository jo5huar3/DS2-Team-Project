# esn_lorenz63_skorch_grid.py
import math, time, os
from typing import Optional
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
        self.Wout = nn.Linear(self.reservoir_size, self.output_size, bias=True)

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

        y_hat = self.Wout(r)
        return y_hat

    def forward_step(self, r, u):
        """
        Single step update of reservoir state.
        r: (B, N) reservoir state
        u: (B, F) input
        returns: (B, N) updated reservoir state
        """
        drive = u @ self.W_in.t() + r @ self.W.t() + self.b  # (B, N)
        r_new = (1 - self.leak) * r + self.leak * self.act(drive)
        return r_new
    
    @torch.no_grad()
    def warmup(self, U):
        """
        Teacher-forcing warmup to initialize the reservoir state.
        U: (B, L0, input_size) or (L0, input_size) or numpy
        Returns:
            r: (B, reservoir_size) final warmup state
            last_in: (B, input_size) last warmup input
        """
        U = torch.as_tensor(U, dtype=torch.float32, device=self.W.device)
        if U.dim() == 2:
            U = U.unsqueeze(0)  # (1, L0, F)

        B, L0, F = U.shape
        assert F == self.input_size, f"Expected input_size={self.input_size}, got {F}"

        r = torch.zeros(B, self.reservoir_size, dtype=U.dtype, device=U.device)
        for t in range(L0):
            r = self.forward_step(r, U[:, t, :])
        return r, U[:, -1, :]

    def forward_seq(self, warm: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Teacher-forced sequence prediction.
        warm: (B, W, F)   -- primes the reservoir
        x:    (B, X, F)   -- inputs where each step predicts next-step y
        returns y_hat: (B, X, F)
        """
        # Warm up the reservoir with teacher forcing
        r, _last_in = self.warmup(warm)          # r: (B, N)
        B, X, F = x.shape
        outs = []
        for t in range(X):
            r = self.forward_step(r, x[:, t, :]) # update reservoir with x_t
            y_t = self.Wout(r)                   # predict next step
            outs.append(y_t.unsqueeze(1))
        return torch.cat(outs, dim=1)            # (B, X, F)

    @torch.no_grad()
    def rollout(self, warm: torch.Tensor, steps: int) -> torch.Tensor:
        """
        Free-run autoregressive rollout after warmup.
        warm: (B, W, F)
        returns: (B, steps, F)
        """
        r, last_in = self.warmup(warm)           # (B, N), (B, F)
        x_prev = last_in
        outs = []
        for _ in range(steps):
            r = self.forward_step(r, x_prev)
            y = self.Wout(r)
            outs.append(y.unsqueeze(1))
            x_prev = y                           # feed prediction back in
        return torch.cat(outs, dim=1)

# esn_lorenz63.py
# End-to-end: simulate Lorenz63, train ESN, evaluate 1-step and closed-loop.
# Requires: numpy, torch, matplotlib (optional for plots)

import math
import numpy as np
import torch
import torch.nn as nn
from skorch import NeuralNetClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler

# ---------------------------
# Lorenz63 simulator (RK4)
# ---------------------------
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

def simulate_lorenz63(T=20000, dt=0.01, x0=(1.0, 1.0, 1.0),
                      sigma=10.0, rho=28.0, beta=8.0/3.0,
                      discard=1000):
    """Return array U shape [T,3] after discarding transients."""
    state = np.array(x0, dtype=np.float64)
    U = []
    for _ in range(discard + T):
        state = rk4_step(lambda s: lorenz63_rhs(s, sigma, rho, beta), state, dt)
        if _ >= discard:
            U.append(state.copy())
    U = np.array(U, dtype=np.float32)
    return U

class MLPModule(nn.Module):
    def __init__(self, in_dim=32, hidden=128, out_dim=3):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, out_dim)
        )
    def forward(self, X): return self.seq(X)

# ---------------------------
# Echo State Network
# ---------------------------
class ESN(nn.Module):
    def __init__(
        self,
        input_dim: int,
        reservoir_dim: int = 1000,
        output_dim: int = 3,
        spectral_radius: float = 0.9,
        sparsity: float = 0.95,
        leaking_rate: float = 0.3,
        input_scale: float = 0.5,
        fb_scale: float = 0.2,      # >0 enables output feedback
        ridge_reg: float = 1e-5,
        seed: int = 42,
        device: str = "cpu",
    ):
        super().__init__()
        self.input_dim = input_dim
        self.reservoir_dim = reservoir_dim
        self.output_dim = output_dim
        self.spectral_radius = spectral_radius
        self.sparsity = sparsity
        self.leaking_rate = leaking_rate
        self.input_scale = input_scale
        self.fb_scale = fb_scale
        self.ridge_reg = ridge_reg
        self.device = device

        rng = np.random.default_rng(seed)

        # Input weights: reservoir_dim x (1 + input_dim)  (prepend bias 1)
        Win = rng.uniform(-1, 1, size=(reservoir_dim, 1 + input_dim)).astype(np.float32)
        self.Win = torch.tensor(self.input_scale * Win, device=device)

        # Reservoir weights: sparse random, rescaled to spectral radius
        W = rng.uniform(-1, 1, size=(reservoir_dim, reservoir_dim)).astype(np.float32)
        mask = rng.uniform(0, 1, size=W.shape) < self.sparsity
        W[mask] = 0.0
        eigvals = np.linalg.eigvals(W)
        rad = np.max(np.abs(eigvals)) + 1e-12
        W *= (self.spectral_radius / rad)
        self.W = torch.tensor(W, device=device)

        # Optional output feedback
        if self.fb_scale > 0.0:
            Wfb = rng.uniform(-1, 1, size=(reservoir_dim, output_dim)).astype(np.float32)
            self.Wfb = torch.tensor(self.fb_scale * Wfb, device=device)
        else:
            self.Wfb = None

        self.Wout = None  # learned via ridge regression

    @torch.no_grad()
    def _reservoir_step(self, x_prev, u_t, y_prev=None):
        """
        One ESN step. x_prev: [B,N], u_t: [B,Din], y_prev: [B,Dout] or None.
        t-1 feedback happens via W @ x_prev and (optional) Wfb @ y_prev.
        """
        B = u_t.shape[0]
        ones = torch.ones(B, 1, device=self.device)
        drive = torch.cat([ones, u_t], dim=1)                       # [B, 1+Din]
        pre = drive @ self.Win.T + x_prev @ self.W.T                # + W x_{t-1}
        if self.Wfb is not None and y_prev is not None:
            pre = pre + y_prev @ self.Wfb.T                         # + W_fb y_{t-1}
        x_tilde = torch.tanh(pre)
        x_t = (1 - self.leaking_rate) * x_prev + self.leaking_rate * x_tilde
        return x_t

    @torch.no_grad()
    def collect_states(self, U, Y=None, washout: int = 200):
        """
        Build design matrix H = [1; U_t; X_t] and targets Tgt = U_{t+1}, after washout.
        U: [T,Din], Y: [T,Dout] (required if feedback enabled for teacher forcing).
        """
        U = torch.as_tensor(U, device=self.device, dtype=torch.float32)
        T = U.shape[0]
        if self.Wfb is not None:
            assert Y is not None, "Provide Y when feedback is enabled (teacher forcing)."
            Y = torch.as_tensor(Y, device=self.device, dtype=torch.float32)

        x = torch.zeros(1, self.reservoir_dim, device=self.device)
        y_prev = torch.zeros(1, self.output_dim, device=self.device) if self.Wfb is not None else None

        states, drives = [], []
        for t in range(T - 1):
            u_t = U[t:t+1, :]
            if self.Wfb is not None:
                y_prev = Y[t:t+1, :]  # teacher-forced true y_{t}
            x = self._reservoir_step(x, u_t, y_prev=y_prev)
            if t >= washout:
                states.append(x.clone())
                drives.append(u_t.clone())

        if not states:
            raise ValueError("Washout too large for sequence length.")

        H = torch.cat(
            [torch.ones(len(states), 1, device=self.device),
             torch.cat(drives, dim=0),
             torch.cat(states, dim=0)],
            dim=1
        )  # [T', 1+Din+N]
        Tgt = U[washout+1:, :]  # [T', Dout]
        return H, Tgt

    @torch.no_grad()
    def fit(self, U, washout: int = 200, Y_for_feedback=None):
        """Closed-form ridge regression for Wout to predict next-step U_{t+1}."""
        if self.fb_scale > 0.0:
            assert Y_for_feedback is not None, "Pass Y_for_feedback=U when using feedback."
        H, Tgt = self.collect_states(U, Y=Y_for_feedback, washout=washout)
        lam = self.ridge_reg
        HT = H.T
        G = HT @ H
        d = G.shape[0]
        G = G + lam * torch.eye(d, device=self.device)
        Wout_T = torch.linalg.solve(G, HT @ Tgt)  # [d, Dout]
        self.Wout = Wout_T.T.contiguous()         # [Dout, d]
        return self

    @torch.no_grad()
    def predict_open_loop(self, U, washout: int = 200):
        """One-step-ahead predictions using true U_t as drive (no feedback term used)."""
        assert self.Wout is not None, "Call fit() first."
        U = torch.as_tensor(U, device=self.device, dtype=torch.float32)
        T = U.shape[0]
        x = torch.zeros(1, self.reservoir_dim, device=self.device)
        preds = []
        for t in range(T - 1):
            u_t = U[t:t+1, :]
            x = self._reservoir_step(x, u_t, y_prev=None)
            if t >= washout:
                H_t = torch.cat([torch.ones(1,1,device=self.device), u_t, x], dim=1)
                yhat = (self.Wout @ H_t.T).T
                preds.append(yhat)
        if not preds:
            return np.zeros((0, self.output_dim), dtype=np.float32)
        return torch.cat(preds, dim=0).cpu().numpy()

    @torch.no_grad()
    def predict_closed_loop(self, u0, steps: int, warm_start_U=None):
        """
        Autonomous rollout: start from u0 (shape [D]), optionally warm-start with a short
        slice of true inputs to settle x. During rollout, prediction at t is fed as input at t+1.
        If feedback is enabled, y_{t-1} used inside reservoir is the last prediction.
        """
        assert self.Wout is not None, "Call fit() first."
        x = torch.zeros(1, self.reservoir_dim, device=self.device)
        outputs = []

        # Warm start
        if warm_start_U is not None and len(warm_start_U) > 0:
            U = torch.as_tensor(warm_start_U, device=self.device, dtype=torch.float32)
            y_prev = None
            for t in range(U.shape[0]):
                u_t = U[t:t+1, :]
                x = self._reservoir_step(x, u_t, y_prev=(outputs[-1] if (self.Wfb is not None and outputs) else None))
                H_t = torch.cat([torch.ones(1,1,device=self.device), u_t, x], dim=1)
                yhat = (self.Wout @ H_t.T).T
                outputs.append(yhat)

        # Closed-loop
        u = torch.as_tensor(u0, device=self.device, dtype=torch.float32).reshape(1, -1)
        for _ in range(steps):
            y_prev = outputs[-1] if (self.Wfb is not None and outputs) else None
            x = self._reservoir_step(x, u, y_prev=y_prev)
            H_t = torch.cat([torch.ones(1,1,device=self.device), u, x], dim=1)
            yhat = (self.Wout @ H_t.T).T
            outputs.append(yhat)
            u = yhat  # feed prediction as next input

        return torch.cat(outputs, dim=0).cpu().numpy()

# ---------------------------
# Metrics
# ---------------------------
def mse(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    return float(np.mean((a - b) ** 2))

# ---------------------------
# Main experiment
# ---------------------------
def main():
    seed = 0
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Device (ESN uses mostly CPU math; CUDA doesn't help much, but allowed)
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    print(f"Using {device} device")

    # 1) Simulate Lorenz63
    T_total = 25000
    dt = 0.01
    U = simulate_lorenz63(T=T_total, dt=dt, x0=(1.0, 1.0, 1.0), discard=1000)
    # U shape [T_total, 3]

    # 2) Split train/test
    T_train = 18000
    U_train = U[:T_train]
    U_test  = U[T_train:]

    # 3) Standardize using train stats; keep transformer to invert later
   
    scaler = StandardScaler().fit(U_train)
    U_train_n = scaler.transform(U_train).astype(np.float64)
    U_test_n  = scaler.transform(U_test).astype(np.float64)

    net = NeuralNetClassifier(
        MLPModule,
        max_epochs=20,
        lr=1e-3,
        iterator_train__shuffle=True,
        device=device,
    )

    param_grid = {
        "lr": [1e-2, 3e-3, 1e-3],
        "module__hidden": [64, 128, 256],
        "max_epochs": [10, 20],
    }

    gs = GridSearchCV(net, param_grid=param_grid, cv=5, n_jobs=-1, refit=True)
    gs.fit(U_train)
    print(gs.best_params_, gs.best_score_)
    best_model = gs.best_estimator_


'''
    # 4) Build & train ESN
    esn = ESN(
        input_dim=3,
        reservoir_dim=1200,
        output_dim=3,
        spectral_radius=0.95,
        sparsity=0.95,
        leaking_rate=0.3,
        input_scale=0.5,
        fb_scale=0.2,          # try feedback on
        ridge_reg=1e-3,
        seed=seed,
        device=device
    )

    washout = 300
    esn.fit(U_train_n, washout=washout, Y_for_feedback=U_train_n)

    # 5) 1-step-ahead prediction on train and test (open-loop)
    preds_train_n = esn.predict_open_loop(U_train_n, washout=washout)  # aligns with U_{washout+1:}
    gt_train_n    = U_train_n[washout+1:, :]
    train_mse_n   = mse(preds_train_n, gt_train_n)

    preds_test_n = esn.predict_open_loop(U_test_n, washout=washout)
    gt_test_n    = U_test_n[washout+1:, :]
    test_mse_n   = mse(preds_test_n, gt_test_n)

    # Back to original units (optional reporting)
    preds_test = scaler.inverse_transform(preds_test_n)
    gt_test    = scaler.inverse_transform(gt_test_n)
    test_mse   = mse(preds_test, gt_test)

    print(f"One-step MSE (normalized) - train: {train_mse_n:.6f}, test: {test_mse_n:.6f}")
    print(f"One-step MSE (original units) - test: {test_mse:.6f}")

    # 6) Closed-loop autonomous rollout on test: seed from the last true test point
    #    and compare to ground-truth continuation (use the *next* slice of U_test_n)
    steps = 3000
    # Warm-start with a short true slice to settle reservoir
    warm_len = 300
    warm_slice = U_test_n[:warm_len]
    u0 = U_test_n[warm_len - 1]            # last warm-start state
    rollout_n = esn.predict_closed_loop(u0, steps=steps, warm_start_U=warm_slice)
    # Compare only the closed-loop part against held-out truth:
    gt_roll_n = U_test_n[warm_len:warm_len+steps]
    closed_mse_n = mse(rollout_n[:len(gt_roll_n)], gt_roll_n)

    # Back to original units for readability
    rollout = scaler.inverse_transform(rollout_n)
    gt_roll = scaler.inverse_transform(gt_roll_n)
    closed_mse = mse(rollout[:len(gt_roll)], gt_roll)

    print(f"Closed-loop {steps}-step MSE (normalized): {closed_mse_n:.6f}")
    print(f"Closed-loop {steps}-step MSE (original units): {closed_mse:.6f}")

    # 7) (Optional) Plot a window of test 1-step preds and closed-loop rollout
    try:
        import matplotlib.pyplot as plt

        # 1-step on test
        K = min(2000, len(preds_test))
        t_axis = np.arange(K) * dt
        fig1 = plt.figure(figsize=(10, 6))
        for i, lbl in enumerate(["x", "y", "z"]):
            ax = plt.subplot(3,1,i+1)
            ax.plot(t_axis, gt_test[:K, i], label=f"GT {lbl}")
            ax.plot(t_axis, preds_test[:K, i], linestyle="--", label=f"Pred {lbl}")
            ax.set_ylabel(lbl)
            if i == 0:
                ax.set_title("One-step-ahead prediction (test)")
            if i == 2:
                ax.set_xlabel("time")
            ax.legend(loc="best")
        plt.tight_layout()

        # Closed-loop
        K2 = min(2000, len(rollout))
        t_axis2 = np.arange(K2) * dt
        fig2 = plt.figure(figsize=(10, 6))
        for i, lbl in enumerate(["x", "y", "z"]):
            ax = plt.subplot(3,1,i+1)
            ax.plot(t_axis2, gt_roll[:K2, i], label=f"GT {lbl}")
            ax.plot(t_axis2, rollout[:K2, i], linestyle="--", label=f"ESN closed-loop {lbl}")
            ax.set_ylabel(lbl)
            if i == 0:
                ax.set_title("Closed-loop rollout vs ground-truth continuation")
            if i == 2:
                ax.set_xlabel("time")
            ax.legend(loc="best")
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print("Plotting skipped (matplotlib not available or other issue):", e)
'''
if __name__ == "__main__":
    main()

# rnn_lorenz63_multistep.py
import math
import random
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


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

def simulate_lorenz63(n_steps, dt, x0, sigma=10.0, rho=28.0, beta=8.0/3.0, discard=100):
    """Simulate one trajectory. Discard initial steps to pass transient."""
    rhs = lambda s: lorenz63_rhs(s, sigma=sigma, rho=rho, beta=beta)
    s = np.array(x0, dtype=np.float64)
    # burn-in
    for _ in range(discard):
        s = rk4_step(rhs, s, dt)
    traj = np.zeros((n_steps, 3), dtype=np.float64)
    for t in range(n_steps):
        s = rk4_step(rhs, s, dt)
        traj[t] = s
    return traj  # shape: (n_steps, 3)


# ---------------------------
# Simple standard scaler
# ---------------------------
class StandardScaler:
    def __init__(self):
        self.mean_ = None
        self.std_ = None

    def fit(self, x):
        self.mean_ = x.mean(axis=(0, 1))  # expect (N, T, C)
        self.std_ = x.std(axis=(0, 1))
        self.std_[self.std_ == 0] = 1.0
        return self

    def transform(self, x):
        return (x - self.mean_) / self.std_

    def inverse_transform(self, x):
        return x * self.std_ + self.mean_


# ---------------------------
# Dataset: sliding windows
# ---------------------------
class LorenzWindows(Dataset):
    def __init__(self, trajectories, in_len, out_len, scaler=None):
        """
        trajectories: list of arrays, each (T, 3)
        Returns windows (X, Y):
          X: (in_len, 3)
          Y: (out_len, 3)
        """
        self.samples_X = []
        self.samples_Y = []

        for traj in trajectories:
            T = traj.shape[0]
            total = in_len + out_len
            for start in range(0, T - total + 1):
                x = traj[start:start+in_len]
                y = traj[start+in_len:start+in_len+out_len]
                self.samples_X.append(x)
                self.samples_Y.append(y)

        self.samples_X = np.stack(self.samples_X, axis=0)  # (N, in_len, 3)
        self.samples_Y = np.stack(self.samples_Y, axis=0)  # (N, out_len, 3)

        # normalize if scaler provided; else fit one on this dataset
        if scaler is None:
            self.scaler = StandardScaler().fit(self.samples_X)
        else:
            self.scaler = scaler

        self.samples_X = self.scaler.transform(self.samples_X)
        self.samples_Y = self.scaler.transform(self.samples_Y)

    def __len__(self):
        return self.samples_X.shape[0]

    def __getitem__(self, idx):
        x = torch.tensor(self.samples_X[idx], dtype=torch.float32)  # (in_len, 3)
        y = torch.tensor(self.samples_Y[idx], dtype=torch.float32)  # (out_len, 3)
        return x, y


# ---------------------------
# Encoder-Decoder RNN
# ---------------------------
class EncoderDecoderRNN(nn.Module):
    def __init__(self, input_size=3, hidden_size=128, num_layers=1, nonlinearity="tanh"):
        super().__init__()
        self.encoder = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            nonlinearity=nonlinearity,
            batch_first=True,
        )
        self.decoder_cell = nn.RNNCell(input_size=input_size, hidden_size=hidden_size, nonlinearity=nonlinearity)
        self.out_proj = nn.Linear(hidden_size, input_size)

    def forward(self, src, tgt=None, out_len=50, teacher_forcing_ratio=0.5):
        """
        src: (B, T_in, C)
        tgt: (B, T_out, C) normalized targets (optional during training)
        returns: preds (B, T_out, C)
        """
        B = src.size(0)
        _, h = self.encoder(src)           # h: (num_layers, B, H)
        h = h[-1]                           # (B, H)
        dec_in = src[:, -1, :]              # start decoding from last observed state, (B, C)

        outputs = []
        for t in range(out_len):
            h = self.decoder_cell(dec_in, h)     # (B, H)
            y = self.out_proj(h)                 # (B, C)
            outputs.append(y.unsqueeze(1))       # accumulate along time

            use_tf = self.training and (tgt is not None) and (random.random() < teacher_forcing_ratio)
            dec_in = tgt[:, t, :] if use_tf else y

        return torch.cat(outputs, dim=1)   # (B, out_len, C)

def pick_device():
    # Prefer CUDA if present
    if torch.cuda.is_available():
        # (Optional) allow TF32 on newer NVIDIA for speed
        torch.set_float32_matmul_precision("high")
        return torch.device("cuda")

    # Next, try Apple Silicon (MPS)
    has_mps = getattr(torch.backends, "mps", None) is not None
    if has_mps and torch.backends.mps.is_available():
        return torch.device("mps")

    # Fallback
    return torch.device("cpu")

# ---------------------------
# Training / Evaluation
# ---------------------------
@dataclass
class Config:
    seed: int = 42
    device = pick_device()
    print(f"Using device: {device}")

    # Data generation
    n_traj_train: int = 50
    n_traj_val: int = 10
    dt: float = 0.01
    steps_per_traj: int = 3000

    # Windowing
    in_len: int = 50
    out_len: int = 50

    # Model / training
    hidden_size: int = 128
    num_layers: int = 1
    nonlinearity: str = "tanh"
    batch_size: int = 128
    epochs: int = 15
    lr: float = 1e-3
    grad_clip: float = 1.0

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def make_trajectories(n_traj, steps_per_traj, dt):
    trajs = []
    for _ in range(n_traj):
        # Random initial condition in a moderate box
        x0 = np.random.uniform(low=-15.0, high=15.0, size=(3,))
        traj = simulate_lorenz63(n_steps=steps_per_traj, dt=dt, x0=x0)
        trajs.append(traj.astype(np.float32))
    return trajs

def train_one_epoch(model, loader, optimizer, mse, epoch, cfg):
    model.train()
    total_loss = 0.0
    # Scheduled teacher forcing: decay from 1.0 → 0.1
    tf_ratio = 1.0 - 0.9 * (epoch / max(1, cfg.epochs - 1))
    for X, Y in loader:
        X = X.to(cfg.device)  # (B, in_len, 3)
        Y = Y.to(cfg.device)  # (B, out_len, 3)
        pred = model(X, tgt=Y, out_len=cfg.out_len, teacher_forcing_ratio=tf_ratio)
        loss = mse(pred, Y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()
        total_loss += loss.item() * X.size(0)
    return total_loss / len(loader.dataset)

@torch.no_grad()
def eval_rollout(model, loader, mse, cfg, scaler):
    """Closed-loop rollout (no teacher forcing) on validation set.
       Returns average MSE over the whole predicted horizon in ORIGINAL units.
    """
    model.eval()
    total_mse = 0.0
    n = 0
    for X, Y in loader:
        X = X.to(cfg.device)
        Y = Y.to(cfg.device)
        pred_norm = model(X, tgt=None, out_len=cfg.out_len, teacher_forcing_ratio=0.0)  # (B, out_len, 3)

        # compute error in original units for interpretability
        pred = torch.tensor(scaler.inverse_transform(pred_norm.cpu().numpy()), dtype=torch.float32, device=cfg.device)
        target = torch.tensor(scaler.inverse_transform(Y.cpu().numpy()), dtype=torch.float32, device=cfg.device)

        loss = mse(pred, target)
        total_mse += loss.item() * X.size(0)
        n += X.size(0)
    return total_mse / n

def main():
    cfg = Config()
    set_seed(cfg.seed)

    # 1) Generate trajectories
    train_trajs = make_trajectories(cfg.n_traj_train, cfg.steps_per_traj, cfg.dt)
    val_trajs   = make_trajectories(cfg.n_traj_val,   cfg.steps_per_traj, cfg.dt)

    # 2) Build datasets + scaler (fit on train X)
    tmp_ds_for_scaler = LorenzWindows(train_trajs, cfg.in_len, cfg.out_len, scaler=None)
    scaler = tmp_ds_for_scaler.scaler  # fitted on train X
    train_ds = tmp_ds_for_scaler
    val_ds   = LorenzWindows(val_trajs, cfg.in_len, cfg.out_len, scaler=scaler)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=cfg.batch_size, shuffle=False, drop_last=False)

    # 3) Model
    model = EncoderDecoderRNN(
        input_size=3,
        hidden_size=cfg.hidden_size,
        num_layers=cfg.num_layers,
        nonlinearity=cfg.nonlinearity
    ).to(cfg.device)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    mse = nn.MSELoss()

    # 4) Train
    for epoch in range(cfg.epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, mse, epoch, cfg)
        val_mse = eval_rollout(model, val_loader, mse, cfg, scaler)
        print(f"Epoch {epoch+1:02d}/{cfg.epochs} | train(MSE, normalized)={train_loss:.6f} "
              f"| val(MSE, closed-loop, original-units)={val_mse:.6f}")

    # 5) Quick qualitative check on one batch
    X, Y = next(iter(val_loader))  # normalized
    X = X.to(cfg.device); Y = Y.to(cfg.device)
    with torch.no_grad():
        pred_norm = model(X, tgt=None, out_len=cfg.out_len, teacher_forcing_ratio=0.0)
    # Inverse-transform (B, T, C) back to original units
    pred = torch.tensor(scaler.inverse_transform(pred_norm.cpu().numpy()))
    target = torch.tensor(scaler.inverse_transform(Y.cpu().numpy()))
    print("\nExample rollout (first sample):")
    print("pred[0, :5] =\n", pred[0, :5])
    print("true[0, :5] =\n", target[0, :5])


if __name__ == "__main__":
    main()

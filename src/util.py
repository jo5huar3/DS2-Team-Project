import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import pandas as pd
import torch
import matplotlib.pyplot as plt
from dataclasses import asdict, dataclass
from typing import Optional, Dict, Any

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




try:
    from sklearn.metrics import mean_squared_error, r2_score
except Exception:
    def mean_squared_error(y_true, y_pred, multioutput="uniform_average"):
        y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
        per_dim = ((y_true - y_pred) ** 2).mean(axis=0)
        return per_dim if multioutput == "raw_values" else per_dim.mean()
    def r2_score(y_true, y_pred, multioutput="uniform_average"):
        y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
        ybar = y_true.mean(axis=0)
        ss_res = ((y_true - y_pred) ** 2).sum(axis=0)
        ss_tot = ((y_true - ybar) ** 2).sum(axis=0) + 1e-12
        per_dim = 1.0 - ss_res/ss_tot
        return per_dim if multioutput == "raw_values" else per_dim.mean()

def report_rollout_metrics(
    esn,
    valid_ds,
    WARM: int,
    XLEN: int,
    K: int = 50,
    idx: int = 0,
    readout_fn=None,             # optional: callable(r)->(B,D); defaults to esn.Wout
    axes_labels=('x','y','z'),
    plot: bool = True,
    device: str | torch.device | None = None,
):
    """Robust K-step rollout reporter that guarantees (K,D) alignment."""
    if device is None:
        device = next(esn.parameters()).device if any(p.requires_grad for p in esn.parameters()) else torch.device('cpu')

    def _readout(r):
        return readout_fn(r) if readout_fn is not None else esn.Wout(r)

    esn.eval()
    with torch.no_grad():
        # ---- 1) fetch window & D ----
        warm_std, x_std, y_std = valid_ds[idx]                 # tensors, std space
        D = int(warm_std.shape[-1])

        # ---- 2) rollout K in std space ----
        start_window_std = torch.cat([warm_std, x_std], dim=0).cpu().numpy()  # (WARM+XLEN, D)
        seq   = torch.as_tensor(start_window_std, dtype=torch.float32, device=device).unsqueeze(0)  # (1,W+X,D)
        warm  = seq[:, :WARM, :]
        teach = seq[:, WARM:, :]

        r, _ = esn.warmup(warm)
        for t in range(teach.size(1)):
            r = esn.forward_step(r, teach[:, t, :])

        preds = []
        x_next = _readout(r)  # start one step ahead
        for _ in range(K):
            r = esn.forward_step(r, x_next)
            y_t = _readout(r)
            preds.append(y_t)
            x_next = y_t
        preds_std = torch.cat(preds, dim=1).squeeze(0).detach().cpu().numpy()  # expected (K,D)

    # ---- 3) exact GT in std space if available; else fallback ----
    gt_std = None
    if all(hasattr(valid_ds, a) for a in ("series", "mean", "std", "starts")):
        s0 = int(valid_ds.starts[idx])
        series = valid_ds.series
        mu, sd = valid_ds.mean, valid_ds.std
        total = WARM + XLEN
        gt_orig = series[s0 + total : s0 + total + K]                     # (K,D)
        if len(gt_orig) > 0:
            gt_std = (gt_orig - mu) / sd

    if gt_std is None:
        y_np = y_std.cpu().numpy()                                        # (XLEN,D)
        if y_np.shape[0] >= K:
            gt_std = y_np[:K]
        else:
            pad = np.full((K - y_np.shape[0], D), np.nan, dtype=np.float32)
            gt_std = np.vstack([y_np, pad])

    # ---- 4) HARDEN SHAPES: ensure both are (K,D), not flattened ----
    preds_std = np.asarray(preds_std)
    gt_std    = np.asarray(gt_std)

    if preds_std.ndim == 1:
        preds_std = preds_std.reshape(-1, D)
    if gt_std.ndim == 1:
        gt_std = gt_std.reshape(-1, D)

    # Align lengths (min along time)
    K_eff = min(preds_std.shape[0], gt_std.shape[0])
    preds_std = preds_std[:K_eff]
    gt_std    = gt_std[:K_eff]

    # ---- 5) mask NaNs (from fallback GT) and compute metrics ----
    mask = ~np.isnan(gt_std).any(axis=1)           # (K_eff,)
    gt = gt_std[mask]                              # (K_use,D)
    pd = preds_std[mask]                           # (K_use,D)

    mse_overall = float(mean_squared_error(gt, pd))
    per_dim_mse = np.asarray(mean_squared_error(gt, pd, multioutput="raw_values"))
    try:
        per_dim_r2 = np.asarray(r2_score(gt, pd, multioutput="raw_values"))
    except TypeError:  # fallback wrapper returned scalar
        ybar = gt.mean(axis=0)
        ss_res = ((gt - pd) ** 2).sum(axis=0)
        ss_tot = ((gt - ybar) ** 2).sum(axis=0) + 1e-12
        per_dim_r2 = 1.0 - ss_res / ss_tot

    # ---- 6) prints ----
    print(f"{K}-step rollout MSE (std): {mse_overall:.6f}")
    print("per-dim MSE:", np.array2string(per_dim_mse, precision=6))
    print("R² per dim :", np.array2string(per_dim_r2,  precision=6))

    # ---- 7) histogram ----
    if plot and gt.shape[0] > 0:
        names = list(axes_labels) if axes_labels and len(axes_labels) == D else [f"dim {i}" for i in range(D)]
        plt.figure(figsize=(12, 4))
        for i, name in enumerate(names):
            plt.subplot(1, D, i+1)
            plt.hist(gt[:, i], bins=60, alpha=0.4, label='True', density=True)
            plt.hist(pd[:, i], bins=60, alpha=0.4, label='Pred', density=True)
            plt.title(f'Distribution: {name}')
            plt.xlabel(name); plt.ylabel('Density'); plt.legend()
        plt.tight_layout(); plt.show()

    return dict(
        mse_overall=mse_overall,
        per_dim_mse=per_dim_mse,
        per_dim_r2=per_dim_r2,
        gt_std=gt,                # (K_use, D) after masking
        preds_std=pd,             # (K_use, D) after masking
        start_idx=idx,
        K_requested=K,
        K_effective=int(gt.shape[0]),
    )

# ---------------- 1) Ridge readout (STANDARDIZED) ----------------
def make_readout_std(Theta: torch.Tensor, mu_R: torch.Tensor, std_R: torch.Tensor,
                     D: int, step: int = 0):
    """
    Build a standardized-space readout fn using ridge Θ.
      r -> (r - mu_R)/std_R -> [Rz|1] @ Θ_block
    Θ layout is ((M+1), L*D); 'step' selects which D-wide block (use 0 for 1-step).
    """
    if not torch.is_tensor(Theta):
        Theta = torch.as_tensor(Theta, dtype=torch.float32)
    Theta = Theta.to(torch.float32).contiguous()
    mu_R  = mu_R.to(torch.float32).detach().contiguous()
    std_R = std_R.to(torch.float32).detach().contiguous()

    start, end = step * D, (step + 1) * D
    Theta1 = Theta[:, start:end].contiguous()  # ((M+1), D), float32

    def _f(r: torch.Tensor) -> torch.Tensor:
        # r: (B,M)
        dev, dt = r.device, r.dtype
        rz = (r - mu_R.to(dev, dt)) / std_R.to(dev, dt)
        ones = torch.ones(rz.size(0), 1, device=dev, dtype=dt)
        return torch.cat([rz, ones], dim=1) @ Theta1.to(dev, dt)  # (B,D), standardized
    return _f

# ---------------- 2) Autonomous rollout (STANDARDIZED) ----------------
@torch.inference_mode()
def rollout_autonomous_std(esn,
                           warm_std: torch.Tensor,
                           K: int,
                           readout_fn,
                           device: str = "cpu",
                           sync_x_std: Optional[torch.Tensor] = None):
    """
    Autonomous free-run in STANDARDIZED space.

    Args:
      esn:            trained ESN (not modified)
      warm_std:       (WARM,D) tensor, standardized; used for warmup
      K:              rollout horizon (# of free-run steps)
      readout_fn:     callable(r)->(B,D) returning STANDARDIZED predictions (e.g., from make_readout_std)
      device:         'cpu' / 'cuda' / 'mps'
      sync_x_std:     optional (XLEN,D) STANDARDIZED teacher sequence to sync reservoir
                      before free-run (recommended: your start window 'x' part)

    Returns:
      preds_std:      (K,D) NumPy array in STANDARDIZED space (float32)
    """
    esn.eval()
    warm_std = warm_std.to(device, torch.float32).unsqueeze(0)   # (1,W,D)
    r, _ = esn.warmup(warm_std)                                  # (1,M)

    if sync_x_std is not None:
        sync_x_std = sync_x_std.to(device, torch.float32).unsqueeze(0)  # (1,X,D)
        for t in range(sync_x_std.size(1)):
            r = esn.forward_step(r, sync_x_std[:, t, :])         # teacher-forced sync

    preds = []
    # bootstrap the loop with the current state
    x_next = readout_fn(r)                                       # (1,D) standardized
    for _ in range(int(K)):
        r = esn.forward_step(r, x_next)
        y_t = readout_fn(r)                                      # (1,D) standardized
        preds.append(y_t)
        x_next = y_t
    return torch.cat(preds, dim=0).squeeze(1).detach().cpu().numpy().astype("float32")
    # shape (K,D)

# ---------------- 3) Save / Load utilities ----------------
@dataclass
class ESNConfig:
    input_size: int
    output_size: int
    reservoir_size: int
    leak: float
    spectral_radius: float
    input_scale: float
    bias_scale: float
    sparsity: float
    activation: str
    seed: int

def _extract_esn_config(esn) -> Dict[str, Any]:
    # Adapt this if your ESNRegressor uses different attribute names.
    return dict(
        input_size      = getattr(esn, "input_size"),
        output_size     = getattr(esn, "output_size"),
        reservoir_size  = getattr(esn, "reservoir_size"),
        leak            = float(getattr(esn, "leak")),
        spectral_radius = float(getattr(esn, "spectral_radius")),
        input_scale     = float(getattr(esn, "input_scale")),
        bias_scale      = float(getattr(esn, "bias_scale")),
        sparsity        = float(getattr(esn, "sparsity")),
        activation      = str(getattr(esn, "activation")),
        seed            = int(getattr(esn, "seed", 0)),
    )

def save_esn_and_ridge(path: str,
                       esn,
                       Theta: Optional[torch.Tensor] = None,
                       mu_R: Optional[torch.Tensor] = None,
                       std_R: Optional[torch.Tensor] = None,
                       meta: Optional[Dict[str, Any]] = None):
    """
    Save ESN state and (optionally) ridge readout Θ + R stats (all float32) to a single .pt file.

    path:  output filename, e.g., "esn_ridge_checkpoint.pt"
    meta:  any extra fields to keep (e.g., {'L':10, 'alpha':1e-2})
    """
    ckpt = {
        "esn_config": _extract_esn_config(esn),
        "esn_state" : {k: v.detach().to("cpu", torch.float32) for k, v in esn.state_dict().items()},
        "meta"      : meta or {},
    }
    if Theta is not None:
        ckpt["ridge"] = {
            "Theta":  Theta.detach().to("cpu", torch.float32),
            "mu_R":   mu_R.detach().to("cpu", torch.float32) if mu_R is not None else None,
            "std_R":  std_R.detach().to("cpu", torch.float32) if std_R is not None else None,
        }
    torch.save(ckpt, path)

def load_esn_and_ridge(path: str, device: str = "cpu", ESNClass=None):
    """
    Load ESN + optional ridge from .pt. If ESNClass is provided, returns a restored model on device.
    Returns: (ckpt, esn_or_None, ridge_or_None, readout_fn_or_None)
      - ridge_or_None: dict with Theta/mu_R/std_R if present
      - readout_fn_or_None: if ridge present, returns a step-0 standardized readout function
    """
    ckpt = torch.load(path, map_location="cpu")
    esn_obj = None
    if ESNClass is not None and "esn_config" in ckpt and "esn_state" in ckpt:
        esn_obj = ESNClass(**ckpt["esn_config"]).to(device)
        # make sure parameter dtypes are float32
        state = {k: v.to(torch.float32) for k, v in ckpt["esn_state"].items()}
        esn_obj.load_state_dict(state)

    ridge = None
    readout_fn = None
    if "ridge" in ckpt and ckpt["ridge"]["Theta"] is not None:
        ridge = ckpt["ridge"]
        # If we also reconstructed the ESN, we can return a ready readout_fn (step=0)
        if esn_obj is not None:
            D = getattr(esn_obj, "output_size")
            readout_fn = make_readout_std(ridge["Theta"], ridge["mu_R"], ridge["std_R"], D=D, step=0)
    return ckpt, esn_obj, ridge, readout_fn

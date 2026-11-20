
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional

def _ensure_1d(a) -> np.ndarray:
    a = np.asarray(a).ravel().astype(float)
    return a

def _uniform_dt(t: np.ndarray) -> float:
    return float(np.median(np.diff(t)))

def _fft_psd(x: np.ndarray, fs: float):
    x = _ensure_1d(x)
    n = len(x)
    if n < 4:
        return np.array([0.0]), np.array([0.0])
    w = np.hanning(n)
    xw = (x - x.mean()) * w
    X = np.fft.rfft(xw)
    psd = (np.abs(X) ** 2) / (np.sum(w**2) * fs)
    freqs = np.fft.rfftfreq(n, d=1.0/fs)
    psd = np.maximum(psd, 1e-20)
    return freqs, psd

def _js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    p = p / (p.sum() + 1e-20)
    q = q / (q.sum() + 1e-20)
    m = 0.5 * (p + q)
    def kl(a, b):
        mask = (a > 0) & (b > 0)
        return np.sum(a[mask] * np.log2(a[mask] / b[mask]))
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)

def _interp_to(t_src: np.ndarray, y_src: np.ndarray, t_tgt: np.ndarray) -> np.ndarray:
    return np.interp(t_tgt, t_src, y_src)

def _acorr(x: np.ndarray, norm=True) -> np.ndarray:
    x = x - x.mean()
    c = np.correlate(x, x, mode="full")
    c = c[c.size//2:]
    if norm:
        c = c / (c[0] + 1e-12)
    return c

def _acf_half_life(x: np.ndarray) -> float:
    ac = _acorr(x, norm=True)
    idx = np.argmax(ac <= 0.5)
    if idx == 0 and ac[0] > 0.5:
        return np.nan
    return float(idx)

def _crosscorr_lag(x: np.ndarray, y: np.ndarray) -> int:
    x = x - np.mean(x)
    y = y - np.mean(y)
    c = np.correlate(x, y, mode="full")
    lag = np.argmax(c) - (len(x) - 1)
    return int(lag)

def _simple_peaks(x: np.ndarray) -> np.ndarray:
    x = _ensure_1d(x)
    if len(x) < 3:
        return np.array([], dtype=int)
    dx = np.diff(x)
    sign = np.sign(dx)
    sign[sign == 0] = 1
    zero_cross = np.where((sign[:-1] > 0) & (sign[1:] < 0))[0] + 1
    return zero_cross.astype(int)

def _peak_preservation(x: np.ndarray, y: np.ndarray, tol: int = 5):
    px = _simple_peaks(x)
    py = _simple_peaks(y)
    if len(px) == 0 or len(py) == 0:
        return (np.nan, np.nan)
    used = np.zeros(len(py), dtype=bool)
    matched = 0
    amp_ratios = []
    for i in px:
        diffs = np.abs(py - i)
        j_candidates = np.where((diffs <= tol) & (~used))[0]
        if j_candidates.size == 0:
            continue
        j = j_candidates[np.argmin(diffs[j_candidates])]
        used[j] = True
        matched += 1
        if x[i] != 0:
            amp_ratios.append(y[py[j]] / (x[i] + 1e-12))
    coverage = matched / max(1, len(px))
    med_amp_ratio = float(np.median(amp_ratios)) if amp_ratios else np.nan
    return (coverage, med_amp_ratio)

def _dtw_distance(a: np.ndarray, b: np.ndarray):
    try:
        from fastdtw import fastdtw
        from math import sqrt
        dist, _ = fastdtw(a, b, dist=lambda u,v: abs(u-v))
        return float(dist) / len(a)
    except Exception:
        return np.nan

def series_stride_metrics(t: np.ndarray, x: np.ndarray, stride_s: int) -> dict:
    t = _ensure_1d(t); x = _ensure_1d(x)
    assert len(t) == len(x) and len(t) > stride_s + 2, "insufficient length"
    dt = _uniform_dt(t)
    fs = 1.0 / dt

    # Stride and upsample back
    t_s = t[::stride_s]
    x_s = x[::stride_s]
    x_hat = _interp_to(t_s, x_s, t)

    # PSDs
    f_orig, P_orig = _fft_psd(x, fs)
    fs_new = fs / stride_s
    f_ny_new = 0.5 * fs_new
    mask = f_orig <= f_ny_new
    ce = (np.trapz(P_orig[mask], f_orig[mask]) / np.trapz(P_orig, f_orig)) if P_orig.sum() > 0 else np.nan

    f_hat, P_hat = _fft_psd(x_hat, fs)
    fmin = max(f_orig.min(), f_hat.min()); fmax = min(f_orig.max(), f_hat.max())
    f_common = np.linspace(fmin, fmax, min(len(f_orig), len(f_hat)))
    Por = np.interp(f_common, f_orig, P_orig)
    Ph  = np.interp(f_common, f_hat,  P_hat)
    jsd_psd = _js_divergence(Por, Ph)

    # Time metrics
    nrmse = float(np.sqrt(np.mean((x - x_hat)**2)) / (np.std(x) + 1e-12))
    r = float(np.corrcoef(x, x_hat)[0,1])

    lag_samp = _crosscorr_lag(x, x_hat)
    lag_frac_period = np.nan
    if len(f_orig) > 2:
        k = np.argmax(P_orig[1:]) + 1
        f_dom = f_orig[k] if f_orig[k] > 0 else np.nan
        if np.isfinite(f_dom) and f_dom > 0:
            T_dom = 1.0 / f_dom
            lag_frac_period = (lag_samp * dt) / T_dom

    cov_peaks, amp_ratio = _peak_preservation(x, x_hat, tol=max(2, stride_s//5))

    hl_x  = _acf_half_life(x)
    hl_xh = _acf_half_life(x_hat)
    acf_half_life_rel_err = (abs(hl_x - hl_xh) / max(1e-9, hl_x)) if np.isfinite(hl_x) else np.nan

    dtw = _dtw_distance(x, x_hat)

    return {
        "dt": dt,
        "stride": int(stride_s),
        "fs_orig": fs,
        "fs_stride": fs_new,
        "nyquist_stride": f_ny_new,
        "energy_below_new_nyquist": float(ce),
        "psd_jsd_bits": float(jsd_psd),
        "nrmse": float(nrmse),
        "pearson_r": float(r),
        "phase_lag_samples": int(lag_samp),
        "phase_lag_fraction_of_period": float(lag_frac_period) if np.isfinite(lag_frac_period) else np.nan,
        "peak_coverage": float(cov_peaks) if cov_peaks is not None else np.nan,
        "peak_median_amplitude_ratio": float(amp_ratio) if amp_ratio is not None else np.nan,
        "acf_half_life_rel_error": float(acf_half_life_rel_err) if np.isfinite(acf_half_life_rel_err) else np.nan,
        "dtw_per_len": float(dtw) if dtw is not None else np.nan,
    }

def compare_original_vs_stride(df_original: pd.DataFrame, stride: int) -> pd.DataFrame:
    out = []
    for var in ["x","y","z"]:
        m = series_stride_metrics(df_original["t"].values, df_original[var].values, stride_s=stride)
        m["variable"] = var
        out.append(m)
    cols = ["variable"] + [c for c in out[0].keys() if c != "variable"]
    return pd.DataFrame(out)[cols]

def lorenz(dt=0.001, T=20.0, sigma=10., rho=28., beta=8/3, x0=(1.0,1.0,1.0)) -> pd.DataFrame:
    n = int(T/dt) + 1
    t = np.linspace(0, T, n)
    x = np.zeros((n,3), dtype=float)
    x[0] = np.array(x0, dtype=float)
    def f(v):
        x,y,z = v
        return np.array([sigma*(y-x), x*(rho - z) - y, x*y - beta*z], dtype=float)
    for i in range(n-1):
        k1 = f(x[i])
        k2 = f(x[i] + 0.5*dt*k1)
        k3 = f(x[i] + 0.5*dt*k2)
        k4 = f(x[i] + dt*k3)
        x[i+1] = x[i] + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)
    return pd.DataFrame({"t": t, "x": x[:,0], "y": x[:,1], "z": x[:,2]})

if __name__ == "__main__":
    df = lorenz(dt=0.001, T=20.0)
    step = 250
    metrics = compare_original_vs_stride(df, step)
    print(metrics.to_string(index=False))

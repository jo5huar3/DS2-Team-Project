import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import welch, resample_poly
from scipy.spatial.distance import jensenshannon

# ---------- helpers (from your snippet) ----------
def psd_energy_below(fs, x, f_cut):
    f, Pxx = welch(x, fs=fs, nperseg=min(len(x)//8, 2048))
    mask = f <= f_cut
    ce = np.trapz(Pxx[mask], f[mask]) / np.trapz(Pxx, f)
    return ce, (f, Pxx)

def jsd_psd(f1, P1, f2, P2):
    f = np.linspace(max(f1.min(), f2.min()), min(f1.max(), f2.max()), 2048)
    P1i = np.interp(f, f1, P1)
    P2i = np.interp(f, f2, P2)
    P1i = P1i / (P1i.sum() + 1e-12)
    P2i = P2i / (P2i.sum() + 1e-12)
    return jensenshannon(P1i, P2i) ** 2  # JSD (distance^2)

def resample_to_original(x_stride, s):
    return resample_poly(x_stride, up=s, down=1)

def time_metrics(x, x_hat):
    n = min(len(x), len(x_hat))
    x = x[:n]; x_hat = x_hat[:n]
    nrmse = np.sqrt(np.mean((x - x_hat) ** 2)) / (x.std() + 1e-9)
    r = np.corrcoef(x, x_hat)[0, 1]
    return nrmse, r

def downsample_safe(x, s):
    return resample_poly(x, up=1, down=s)

# ---------- single-axis analysis ----------
def analyze_resampling(x, fs, stride=4, f_cut=None, title_prefix=None, make_plots=False):
    if f_cut is None:
        f_cut = fs / 10.0

    x_ds  = downsample_safe(x, stride)
    fs_ds = fs / stride

    ce_orig, (f1, P1) = psd_energy_below(fs, x, f_cut)
    ce_ds,   (f2, P2) = psd_energy_below(fs_ds, x_ds, min(f_cut, fs_ds/2))

    jsd = jsd_psd(f1, P1, f2, P2)

    x_up = resample_to_original(x_ds, stride)
    nrmse, r = time_metrics(x, x_up)

    if make_plots and title_prefix:
        Nplot = min(4000, len(x), len(x_up))
        plt.figure()
        plt.plot(x[:Nplot], label="original")
        plt.plot(x_up[:Nplot], label=f"down→up (×{stride})", alpha=0.8)
        plt.title(f"{title_prefix}: Time overlay")
        plt.xlabel("sample"); plt.ylabel("amplitude"); plt.legend(); plt.show()

        plt.figure()
        plt.semilogy(f1, P1, label="orig PSD")
        plt.semilogy(f2, P2, label=f"down PSD (fs={fs_ds:.2f} Hz)")
        plt.axvline(f_cut, linestyle="--", label=f"f_cut={f_cut:.2f} Hz")
        plt.title(f"{title_prefix}: PSD (Welch)")
        plt.xlabel("frequency (Hz)"); plt.ylabel("PSD"); plt.legend(); plt.show()

    return {
        "stride": stride,
        "fs_orig": fs,
        "fs_down": fs_ds,
        "energy_below_orig": ce_orig,
        "energy_below_down": ce_ds,
        "jsd_psd": jsd,
        "nrmse_time": nrmse,
        "corr_time": r,
        "f_psd_orig": f1, "Pxx_orig": P1,
        "f_psd_down": f2, "Pxx_down": P2,
        "x_down": x_ds, "x_up": x_up,
    }

# ---------- batch analysis across x,y,z ----------
def analyze_resampling_axes(x, y, z, fs, strides=(2, 4, 8), f_cut=None, make_plots=False):
    """
    Runs analyze_resampling on each axis for each stride and returns:
      - df: tidy DataFrame with per-axis metrics per stride
      - artifacts: nested dict with PSDs and reconstructed signals
    """
    axes = {"x": x, "y": y, "z": z}
    rows = []
    artifacts = {}

    for axis_name, sig in axes.items():
        artifacts[axis_name] = {}
        for s in strides:
            res = analyze_resampling(
                sig, fs, stride=s, f_cut=f_cut,
                title_prefix=f"Lorenz-63 ({axis_name}) x{s}",
                make_plots=make_plots
            )
            rows.append({
                "axis": axis_name,
                "stride": s,
                "fs_orig": res["fs_orig"],
                "fs_down": res["fs_down"],
                "energy_below_orig": res["energy_below_orig"],
                "energy_below_down": res["energy_below_down"],
                "jsd_psd": res["jsd_psd"],
                "nrmse_time": res["nrmse_time"],
                "corr_time": res["corr_time"],
            })
            artifacts[axis_name][s] = {
                "f_psd_orig": res["f_psd_orig"], "Pxx_orig": res["Pxx_orig"],
                "f_psd_down": res["f_psd_down"], "Pxx_down": res["Pxx_down"],
                "x_down": res["x_down"], "x_up": res["x_up"]
            }

    df = pd.DataFrame(rows).sort_values(["axis", "stride"]).reset_index(drop=True)
    return df, artifacts

# ---------- example usage ----------
# Given Lorenz-63 arrays t, x, y, z and dt:
# fs = 1.0 / dt
# df_metrics, art = analyze_resampling_axes(x, y, z, fs, strides=(2,4,8), f_cut=fs/12, make_plots=False)
# print(df_metrics)
# df_metrics.to_csv("lorenz_resample_metrics.csv", index=False)

from scipy.signal import welch, resample_poly
from scipy.spatial.distance import jensenshannon
import numpy as np
import pandas as pd

# ---------- helpers (robust & band-limited) ----------
def psd_energy_below(fs, x, f_cut, nperseg=None):
    """
    Fraction of PSD energy below f_cut using Welch. Robust defaults for nperseg.
    Returns (fraction_in_band, (freqs, PSD)).
    """
    x = np.asarray(x, float)
    n = len(x)
    if nperseg is None:
        # keep segments not too short and not too long
        nperseg = min(2048, max(256, n // 8))
    nperseg = min(nperseg, n)

    f, Pxx = welch(
        x, fs=fs, nperseg=nperseg,
        detrend="constant", scaling="density", return_onesided=True
    )
    Pxx = np.clip(Pxx, 0.0, None)
    denom = np.trapz(Pxx, f)
    if denom <= 0:
        return 0.0, (f, Pxx)

    mask = f <= float(f_cut)
    num = np.trapz(Pxx[mask], f[mask]) if np.any(mask) else 0.0
    ce = num / denom
    return float(ce), (f, Pxx)


def jsd_psd(f1, P1, f2, P2, f_max=None, base=2):
    """
    Band-limited JSD between normalized PSDs.
    - Compares only up to f_max (defaults to min Nyquist).
    - Uses SciPy's Jensen-Shannon *distance* and squares it to get divergence.
    Returns np.nan if PSD mass in the band is zero for either input.
    """
    f1 = np.asarray(f1, float); P1 = np.asarray(P1, float)
    f2 = np.asarray(f2, float); P2 = np.asarray(P2, float)
    f_hi = min(f1[-1], f2[-1])
    if f_max is not None:
        f_hi = min(f_hi, float(f_max))
    if f_hi <= 0:
        return np.nan

    # Common grid in [0, f_hi]
    m = 2048
    f = np.linspace(0.0, f_hi, m, endpoint=True)

    # Interpolate and clip negative numerical noise
    P1i = np.interp(f, f1, P1, left=0.0, right=0.0)
    P2i = np.interp(f, f2, P2, left=0.0, right=0.0)
    P1i = np.clip(P1i, 0.0, None)
    P2i = np.clip(P2i, 0.0, None)

    s1, s2 = P1i.sum(), P2i.sum()
    if s1 <= 0 or s2 <= 0:
        return np.nan

    p = P1i / s1
    q = P2i / s2

    # SciPy returns the distance in [0,1] when base=2; square for divergence
    return float(jensenshannon(p, q, base=base) ** 2)


def resample_to_original(x_stride, s):
    """Upsample by integer stride s using polyphase filtering."""
    return resample_poly(x_stride, up=int(s), down=1)


def time_metrics(x, x_hat):
    """
    NRMSE normalized by std(x) and Pearson r. Handles constant-series edge cases.
    """
    n = min(len(x), len(x_hat))
    x = np.asarray(x[:n], float)
    x_hat = np.asarray(x_hat[:n], float)

    sd = x.std()
    nrmse = 0.0 if sd <= 1e-12 else float(np.sqrt(np.mean((x - x_hat) ** 2)) / sd)

    sx, sy = x.std(), x_hat.std()
    if sx <= 1e-12 and sy <= 1e-12:
        r = 1.0  # both constant and equal up to shifting
    elif sx <= 1e-12 or sy <= 1e-12:
        r = 0.0  # one constant, the other not
    else:
        r = float(np.corrcoef(x, x_hat)[0, 1])
    return nrmse, r


def downsample_safe(x, s):
    """Downsample by integer stride s with anti-alias filtering (resample_poly)."""
    return resample_poly(x, up=1, down=int(s))


# ---------- single-axis analysis (uses band-limited JSD) ----------
def analyze_resampling(x, fs, stride=4, f_cut=None, title_prefix=None, make_plots=False):
    """
    Downsample->upsample a single series and report:
      energy fractions, band-limited JSD(PSD), time NRMSE & corr, plus artifacts.
    """
    x = np.asarray(x, float)
    if f_cut is None:
        f_cut = fs / 10.0

    x_ds  = downsample_safe(x, stride)
    fs_ds = fs / stride

    ce_orig, (f1, P1) = psd_energy_below(fs, x, f_cut)
    ce_ds,   (f2, P2) = psd_energy_below(fs_ds, x_ds, min(f_cut, fs_ds / 2.0))

    # Compare spectra only on common/allowed band
    f_max = min(f_cut, fs_ds / 2.0)
    jsd = jsd_psd(f1, P1, f2, P2, f_max=f_max, base=2)

    x_up = resample_to_original(x_ds, stride)
    nrmse, r = time_metrics(x, x_up)

    if make_plots and title_prefix:
        import matplotlib.pyplot as plt
        Nplot = min(4000, len(x), len(x_up))
        plt.figure()
        plt.plot(x[:Nplot], label="original")
        plt.plot(x_up[:Nplot], label=f"down→up (×{stride})", alpha=0.8)
        plt.title(f"{title_prefix}: Time overlay")
        plt.xlabel("sample"); plt.ylabel("amplitude"); plt.legend(); plt.show()

        plt.figure()
        plt.semilogy(f1, P1, label="orig PSD")
        plt.semilogy(f2, P2, label=f"down PSD (fs={fs_ds:.2f} Hz)")
        f_cut_eff = min(f_cut, fs_ds / 2.0)
        plt.axvline(f_cut_eff, linestyle="--", label=f"band limit = {f_cut_eff:.2f} Hz")
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

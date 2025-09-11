# lorenz63_plots_both_ways.py
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
# ---------------------------
# Lorenz63 dynamics + RK4
# ---------------------------
def lorenz63_rhs(state, sigma=10.0, rho=28.0, beta=8.0/3.0):
    x, y, z = state
    dx = sigma * (y - x)
    dy = x * (rho - z) - y
    dz = x * y - beta * z
    return np.array([dx, dy, dz], dtype=float)

def rk4_step(f, state, dt, **kwargs):
    k1 = f(state, **kwargs)
    k2 = f(state + 0.5 * dt * k1, **kwargs)
    k3 = f(state + 0.5 * dt * k2, **kwargs)
    k4 = f(state + dt * k3, **kwargs)
    return state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

def simulate_lorenz63(x0=(1.0, 1.0, 1.0), dt=0.01, T=50.0,
                      sigma=10.0, rho=28.0, beta=8.0/3.0):
    n_steps = int(T / dt)
    t = np.linspace(0.0, n_steps * dt, n_steps + 1)
    X = np.empty((n_steps + 1, 3), dtype=float)
    X[0] = np.array(x0, dtype=float)
    s = X[0].copy()
    for i in range(n_steps):
        s = rk4_step(lorenz63_rhs, s, dt, sigma=sigma, rho=rho, beta=beta)
        X[i + 1] = s
    return t, X

# ---------------------------
# PLOTTING HELPERS — Separate figures
# ---------------------------
def plot_timeseries_separate(t, X, dt):
    labels = ["x(t)", "y(t)", "z(t)"]
    for i, lab in enumerate(labels):
        plt.figure()
        plt.plot(t, X[:, i])
        plt.title(f"{lab} — Lorenz63 (dt={dt})")
        plt.xlabel("t")
        plt.ylabel(lab[0])
        plt.grid(True)
        plt.show()
# Function
def plot_histograms_separate(X, dt, burn_in_time=5.0):
    burn_in = int(burn_in_time / dt)
    X_ss = X[burn_in:]
    labels = ["x", "y", "z"]
    for i, lab in enumerate(labels):
        plt.figure()
        plt.hist(X_ss[:, i], bins=200)
        plt.title(f"Histogram of {lab} (after {burn_in_time} time-units burn-in)")
        plt.xlabel(lab)
        plt.ylabel("count")
        plt.grid(True)
        plt.show()

# ---------------------------
# PLOTTING HELPERS — Subplots
# ---------------------------
def plot_timeseries_subplots(t, X, dt):
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.2), constrained_layout=True)
    labels = ["x(t)", "y(t)", "z(t)"]
    for i, ax in enumerate(axes):
        ax.plot(t, X[:, i])
        ax.set_title(labels[i])
        ax.set_xlabel("t")
        ax.set_ylabel(labels[i][0])
        ax.grid(True)
    fig.suptitle(f"Lorenz63 Time Series — dt={dt}")
    plt.show()

def plot_histograms_subplots(X, dt, burn_in_time=5.0):
    burn_in = int(burn_in_time / dt)
    X_ss = X[burn_in:]
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.2), constrained_layout=True)
    labels = ["x", "y", "z"]
    for i, ax in enumerate(axes):
        ax.hist(X_ss[:, i], bins=200)
        ax.set_title(f"Hist {labels[i]}")
        ax.set_xlabel(labels[i])
        ax.set_ylabel("count")
        ax.grid(True)
    fig.suptitle(f"Lorenz63 Steady-State Histograms — dt={dt}")
    plt.show()

# ---------------------------
# MAIN
# ---------------------------
if __name__ == "__main__":
    # Simulation settings
    dt = 0.01
    T = 50.0
    x0 = (1.0, 1.0, 1.0)
    burn_in_time = 5.0  # drop first 5 time-units for histograms

    # Simulate
    t, X = simulate_lorenz63(x0=x0, dt=dt, T=T)

    # Optional: save to CSV
    df = pd.DataFrame({"t": t, "x": X[:, 0], "y": X[:, 1], "z": X[:, 2]})
    df.to_csv("lorenz63_dt001_T50.csv", index=False)

    # ---- Separate figures (one plot per window) ----
    plot_timeseries_separate(t, X, dt)
    plot_histograms_separate(X, dt, burn_in_time=burn_in_time)

    # ---- Subplots (compact 1×3 layouts) ----
    plot_timeseries_subplots(t, X, dt)
    plot_histograms_subplots(X, dt, burn_in_time=burn_in_time)

    print(os.path.abspath("lorenz63_dt001_T50.csv"))
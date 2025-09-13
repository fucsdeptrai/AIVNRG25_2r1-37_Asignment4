import os
import numpy as np
import matplotlib.pyplot as plt
from statistics import mean
from collections import defaultdict
# import train_model đúng chỗ (tùy file của bạn)
from src.train import train_model

# -------------------
# CONFIG (tùy chỉnh)
# -------------------
MEM_SIZE = 64
DIM = 64
STEPS = 1000   # số total samples để train_model dùng (n trong thảo luận)
SEEDS = [0,1,2]   # chạy multi-seed
NS = [50, 100, 200, 500, 1000, 2000, 5000]  # dải n để test
# baseline params (giữ cố định khi vary một param)
BASE_WINDOW = 10
BASE_LR = 0.01
BASE_REG = 1e-3

OUTDIR = "results/plots"
os.makedirs(OUTDIR, exist_ok=True)

# -------------------
# helper chạy experiment
# -------------------
def run_once(n, updater, window, lr, reg, seed):
    """
    Gọi train_model theo API: train_model(mem_size, dim, window, steps, lr, reg, seed, updater_type)
    Trả về mse_mean, cos_mean
    """
    # if your train_model signature is different, adapt here
    mse_mean, cos_mean, update_time, mem_norm = train_model(mem_size=MEM_SIZE,
                                                           dim=DIM,
                                                           window=window,
                                                           steps=n,   # treat steps = n
                                                           lr=lr,
                                                           reg=reg,
                                                           seed=seed,
                                                           updater_type=updater)
    return mse_mean, cos_mean

def collect_for_param(vary_name, vary_values, ns=NS, seeds=SEEDS, updaters=("Omega","Delta"),
                      fixed_params=None):
    """
    vary_name: 'window' or 'lr' or 'reg'
    vary_values: list of values to test for that param
    fixed_params: dict with keys 'window','lr','reg' giving baseline values
    Returns nested dict: results[updater][vary_val][n] = list of mse across seeds
    """
    if fixed_params is None:
        fixed_params = {"window": BASE_WINDOW, "lr": BASE_LR, "reg": BASE_REG}
    results = {u: {v: {n: [] for n in ns} for v in vary_values} for u in updaters}
    for v in vary_values:
        for n in ns:
            for seed in seeds:
                params = fixed_params.copy()
                params[vary_name] = v
                for updater in updaters:
                    mse, cos = run_once(n=n, updater=updater, window=params["window"], lr=params["lr"], reg=params["reg"], seed=seed)
                    results[updater][v][n].append({"mse": mse, "cos": cos})
                    print(f"done updater={updater} {vary_name}={v} n={n} seed={seed} mse={mse:.4e} cos={cos:.4f}")
    return results

# -------------------
# plotting helpers
# -------------------
def plot_metric_vs_n(results, vary_values, metric="mse", vary_name="window", title=None, outname="plot.png"):
    """
    results structure from collect_for_param
    Uses numpy to compute mean and std (ddof=1) robustly.
    """
    plt.figure(figsize=(8,5))
    for updater in results.keys():
        for v in vary_values:
            xs = sorted(results[updater][v].keys())
            ys_mean = []
            ys_std = []
            for n in xs:
                # collect metric values (as floats)
                vals = [float(entry[metric]) for entry in results[updater][v][n]]
                arr = np.array(vals, dtype=float)
                if arr.size == 0:
                    m = np.nan
                    s = 0.0
                else:
                    m = float(np.mean(arr))
                    s = float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0
                ys_mean.append(m)
                ys_std.append(s)
            label = f"{updater} {vary_name}={v}"
            plt.plot(xs, ys_mean, marker='o', label=label)
            ys_mean = np.array(ys_mean)
            ys_std = np.array(ys_std)
            plt.fill_between(xs, ys_mean - ys_std, ys_mean + ys_std, alpha=0.15)
    # optional log-scale for x if values span orders of magnitude
    try:
        plt.xscale('log')
    except Exception:
        pass
    plt.xlabel("n (dataset size)")
    plt.ylabel(metric if metric!='cos' else "cosine")
    plt.title(title or f"{metric} vs n (vary {vary_name})")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    os.makedirs(os.path.dirname(outname) or '.', exist_ok=True)
    plt.savefig(outname)
    print("Saved", outname)
    plt.close()

# -------------------
# Example runs: vary window, keep lr/reg fixed
# -------------------
def main():
    # 1) vary window
    windows = [1,5,10,20]
    res_window = collect_for_param("window", windows, ns=NS, seeds=SEEDS,
                                   updaters=("Omega","Delta"),
                                   fixed_params={"window": BASE_WINDOW,"lr":BASE_LR,"reg":BASE_REG})
    plot_metric_vs_n(res_window, windows, metric="mse", vary_name="window", title="MSE vs n (vary window)", outname="mse_vary_window.png")
    plot_metric_vs_n(res_window, windows, metric="cos", vary_name="window", title="Cosine vs n (vary window)", outname="cos_vary_window.png")

    # 2) vary lr (keep window/reg fixed)
    lrs = [1e-3, 1e-2, 1e-1]
    res_lr = collect_for_param("lr", lrs, ns=NS, seeds=SEEDS,
                               updaters=("Omega","Delta"),
                               fixed_params={"window": BASE_WINDOW,"lr":BASE_LR,"reg":BASE_REG})
    plot_metric_vs_n(res_lr, lrs, metric="mse", vary_name="lr", title="MSE vs n (vary lr)", outname="mse_vary_lr.png")
    plot_metric_vs_n(res_lr, lrs, metric="cos", vary_name="lr", title="Cosine vs n (vary lr)", outname="cos_vary_lr.png")

    # 3) vary reg
    regs = [0.0, 1e-4, 1e-3, 1e-2]
    res_reg = collect_for_param("reg", regs, ns=NS, seeds=SEEDS,
                                updaters=("Omega","Delta"),
                                fixed_params={"window": BASE_WINDOW,"lr":BASE_LR,"reg":BASE_REG})
    plot_metric_vs_n(res_reg, regs, metric="mse", vary_name="reg", title="MSE vs n (vary reg)", outname="mse_vary_reg.png")
    plot_metric_vs_n(res_reg, regs, metric="cos", vary_name="reg", title="Cosine vs n (vary reg)", outname="cos_vary_reg.png")

if __name__ == "__main__":
    main()

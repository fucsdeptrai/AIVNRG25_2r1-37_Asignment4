"""
Utility helpers for plotting and CSV post-processing.
"""
import csv
import matplotlib.pyplot as plt


def read_csv(path):
    rows = []
    with open(path, 'r') as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return rows


def plot_metric_vs_param(csv_path, metric='mse_mean', param='steps', out_png='results/plot.png'):
    rows = read_csv(csv_path)
    # convert to floats
    xs = [float(r[param]) for r in rows]
    ys = [float(r.get(metric, 0.0)) for r in rows]
    plt.figure()
    plt.scatter(xs, ys)
    plt.xlabel(param)
    plt.ylabel(metric)
    plt.title(f'{metric} vs {param}')
    plt.savefig(out_png)
    print('Saved plot to', out_png)


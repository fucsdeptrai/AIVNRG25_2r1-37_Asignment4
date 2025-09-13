"""
Simple bash script to run a grid of experiments and collect results into CSV files (results/).
Usage: bash experiments/run_grid.sh
"""

mkdir -p results
python3 - << 'PY'
import subprocess, itertools, json, os

mem_sizes = [32, 64]
dims = [64]
windows = [20, 50]
steps_list = [4, 8]
lrs = [0.01, 0.005]
regs = [1e-3]
updaters = ['omega', 'delta']
seeds = [0,1,2]

os.makedirs('results', exist_ok=True)
rows = []
for mem_size, dim, window, steps, lr, reg, updater in itertools.product(mem_sizes, dims, windows, steps_list, lrs, regs, updaters):
    for seed in seeds:
        cmd = [
            'python3', 'src/train.py',
            '--mem_size', str(mem_size),
            '--dim', str(dim),
            '--window', str(window),
            '--steps', str(steps),
            '--lr', str(lr),
            '--reg', str(reg),
            '--updater', updater,
            '--seed', str(seed)
        ]
        print('Running:', ' '.join(cmd))
        proc = subprocess.run(cmd, capture_output=True, text=True)
        out = proc.stdout + proc.stderr
        # try to parse printed Metrics: line
        metrics = {}
        for line in out.splitlines():
            if line.strip().startswith('Metrics:'):
                try:
                    metrics_text = line.split('Metrics:')[1].strip()
                    # eval as python dict
                    metrics = eval(metrics_text)
                except Exception as e:
                    print('Could not parse metrics:', e)
        row = dict(mem_size=mem_size, dim=dim, window=window, steps=steps, lr=lr, reg=reg, updater=updater, seed=seed)
        row.update(metrics)
        rows.append(row)
        # save incremental
        import csv
        keys = rows[0].keys()
        with open('results/grid_results.csv', 'w', newline='') as f:
            dict_writer = csv.DictWriter(f, keys)
            dict_writer.writeheader()
            dict_writer.writerows(rows)
PY

echo "Grid run complete. Results saved to results/grid_results.csv"
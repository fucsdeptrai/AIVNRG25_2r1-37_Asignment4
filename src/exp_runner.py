import os
import sys

# ensure project root is in sys.path so `import src.*` works even when running python src/exp_runner.py
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from train import train_model  # hoặc train_model tùy tên hàm bạn export
import csv
from itertools import product

def run_experiments():
    # Lưới siêu tham số (có thể điều chỉnh)
    mem_sizes = [20, 50]
    dims = [10]
    windows = [5, 10]
    steps_list = [1000]
    lrs = [0.01, 0.1]
    regs = [0.0, 0.1]
    seeds = [0, 1]
    updaters = ['Delta', 'Omega']
    os.makedirs('results', exist_ok=True)
    result_file = os.path.join('results', 'exp_results.csv')
    with open(result_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        # Ghi header
        writer.writerow(['mem_size', 'dim', 'window', 'steps', 'lr', 'reg', 
                         'seed', 'updater', 'mse_mean', 'cos_mean', 
                         'update_time', 'mem_norm_change'])
        # Chạy thử nghiệm với tất cả tổ hợp cấu hình
        for mem_size in mem_sizes:
            for dim in dims:
                for window in windows:
                    for steps in steps_list:
                        for lr in lrs:
                            for reg in regs:
                                for seed in seeds:
                                    for updater in updaters:
                                        mse_mean, cos_mean, update_time, mem_norm_change = train_model(
                                            mem_size, dim, window, steps, lr, reg, seed, updater
                                        )
                                        writer.writerow([mem_size, dim, window, steps, lr, reg, seed, updater,
                                                         mse_mean, cos_mean, update_time, mem_norm_change])

if __name__ == '__main__':
    run_experiments()

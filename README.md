# README --- So sánh OmegaUpdater vs DeltaUpdater (toy implementation)

**Ngắn gọn:** repo này chứa một triển khai nhỏ (toy) để so sánh hai
chiến lược cập nhật bộ nhớ ngoài (Omega rule vs Delta rule) trên dữ liệu
tổng hợp (key-value). Mục tiêu: chạy được end-to-end, thu CSV kết quả,
và vẽ đồ thị so sánh.

------------------------------------------------------------------------

# Cấu trúc dự án

    project_root/
    ├─ src/
    │  ├─ data.py                 # sinh dữ liệu synthetic (keys, values)
    │  ├─ updaters.py             # DeltaUpdater, OmegaUpdater
    │  ├─ train.py                # train_model(...) chạy 1 experiment
    │  ├─ exp_runner.py           # chạy grid experiments -> results/exp_results.csv
    │  ├─ plot_sensitivity.py     # so sánh sensitivity (window, lr, reg) vs n
    │  └─ plot_results.py         # plot MSE / 1-cosine vs n (simple)
    ├─ analysis/
    │  └─ analysis.py             # đọc CSV và vẽ boxplots (tùy chọn)
    ├─ experiments/
    │  └─ run_grid.sh             # (tùy) script mẫu để chạy grid
    ├─ results/                   # nơi lưu CSV và plots
    ├─ small_demo.sh              # demo tất cả (tạo venv, chạy thử, chạy exp, analysis)
    ├─ requirements.txt
    └─ README.md

------------------------------------------------------------------------

# Yêu cầu

-   Python 3.8+ (đã test trên Python 3.9/3.10).\
-   PyTorch (CPU build ổn định).\
-   matplotlib, pandas, numpy.\
    Bạn có thể cài nhanh:

``` bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Nếu `requirements.txt` không phù hợp với hệ thống (CUDA) --- cài PyTorch
theo hướng dẫn trên pytorch.org phù hợp máy bạn.

------------------------------------------------------------------------

# Quickstart --- chạy 1 experiment kiểm tra

Chạy từ **project root** (thư mục chứa `src/`):

``` bash
# single run, Omega
python3 -m src.train --updater Omega --mem_size 20 --dim 10 --window 5 --steps 200 --lr 0.01 --reg 0.0 --seed 0
```

Kết quả in ra sẽ gồm `mse_mean`, `cos_mean`, `update_time`,
`mem_norm_change`.\
\> Ghi chú: luôn chạy từ project root hoặc dùng `python -m ...` để
Python tìm `src` đúng.

------------------------------------------------------------------------

# Chạy grid experiments (lưu CSV)

Dùng `exp_runner.py` để chạy một lưới cấu hình và ghi vào
`results/exp_results.csv`:

``` bash
# chạy module (tốt nhất)
python3 -m src.exp_runner --out_csv results/exp_results.csv

# hoặc nếu bạn muốn tùy tham số
python3 -m src.exp_runner --out_csv results/exp_results.csv --mem_sizes 32 64 --dims 64 --windows 20 50 --steps_list 100 200 --lrs 0.01 0.005 --regs 0.0 0.001 --updaters Omega Delta --seeds 0 1 2
```

CSV sẽ có (ít nhất) các cột:

    mem_size, dim, window, steps, lr, reg, seed, updater, mse_mean, cos_mean, update_time_s, mem_norm_change

------------------------------------------------------------------------

# Phân tích & visualization

1.  Sau khi có `results/exp_results.csv`, bạn có thể chạy script
    analysis:

``` bash
python3 analysis/analysis.py
```

2.  Để vẽ đồ thị theo kích thước bộ dữ liệu `n` (so sánh MSE /
    1−cosine):

``` bash
python3 -m src.plot_results
```

3.  Để chạy sensitivity sweep (vary window / lr / reg) và vẽ mean ± std
    cho nhiều `n`:

``` bash
python3 -m src.plot_sensitivity
```

Các plot sẽ được lưu vào `results/plots/` (hoặc nơi script chỉ định).

------------------------------------------------------------------------

# small_demo (1 lệnh chạy thử toàn bộ)

Mình cung cấp `small_demo.sh` --- tạo venv (nếu cần), cài gói, chạy một
single test, chạy exp_runner (grid nhỏ), rồi chạy analysis & plots:

``` bash
chmod +x small_demo.sh
./small_demo.sh
```

Nếu script báo lỗi liên quan
`ModuleNotFoundError: No module named 'src'`, đảm bảo bạn chạy ở
**project root**.

------------------------------------------------------------------------

# Input / Output --- làm rõ

-   **Input chính:** synthetic key--value pairs (do `src/data.py` sinh).
    -   `key`: vector dim (ví dụ 64)\
    -   `value`: linear transform của key + nhiễu\
-   **Hyperparameters (CLI):**
    `mem_size, dim, window, steps (n), lr, reg, updater, seed`\
-   **Output:** CSV với metrics (MSE, cosine, time, mem_norm_change) và
    các plot (png).

------------------------------------------------------------------------

# Một số lưu ý & troubleshooting nhanh

-   `ModuleNotFoundError: No module named 'src'` → chạy từ project root
    hoặc dùng `python3 -m src.<module>`. Nếu bạn chạy bằng
    `python3 src/exp_runner.py`, hãy đảm bảo đường dẫn sys.path đúng
    (mình khuyên dùng `python -m`).
-   Nếu chạy chậm hoặc OOM: giảm `mem_size`, `dim`, `steps` hoặc dùng
    máy có CUDA.
-   Nếu Omega tỏ ra **không ổn định**: tăng `reg` (ridge), giảm
    `window`, hoặc thêm clipping --- mình đã comment trong code những
    chỗ dễ chỉnh.
-   Nếu `plot_sensitivity` báo lỗi về std/mean: đảm bảo `numpy` import
    được và các giá trị trong `results` là số (float), không phải
    strings.

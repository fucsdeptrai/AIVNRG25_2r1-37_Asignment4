import torch
import time
from src.data import generate_synthetic_data
from src.updaters import DeltaUpdater, OmegaUpdater

def train_model(mem_size, dim, window, steps, lr, reg, seed, updater_type):
    """
    Train a memory model on synthetic key-value data using specified updater.
    Returns metrics: mse_mean, cos_mean, update_time, mem_norm_change
    """
    # Sinh dữ liệu tổng hợp
    keys, values = generate_synthetic_data(mem_size, dim, steps, seed=seed)
    torch.manual_seed(seed)
    # Khởi tạo memory (ma trận trọng số) ban đầu bằng 0
    memory = torch.zeros(mem_size, dim)
    # Khởi tạo đối tượng updater tương ứng
    if updater_type == 'Delta':
        updater = DeltaUpdater(lr=lr, reg=reg)
    elif updater_type == 'Omega':
        updater = OmegaUpdater(window=window, reg=reg)
    else:
        raise ValueError("Unknown updater type")
    total_mse = 0.0
    total_cos = 0.0
    total_time = 0.0
    # Vòng lặp huấn luyện qua các mẫu
    for i in range(steps):
        x = keys[i]    # (dim,)
        y = values[i]  # (mem_size,)
        # Dự đoán trước khi cập nhật
        y_pred = memory @ x
        # Tính MSE và cosine similarity
        mse = ((y_pred - y) ** 2).mean().item()
        # Cosine similarity, tránh chia cho 0
        if torch.norm(y_pred).item() > 0 and torch.norm(y).item() > 0:
            cos = torch.dot(y_pred, y) / (torch.norm(y_pred) * torch.norm(y))
        else:
            cos = torch.tensor(0.0)
        total_mse += mse
        total_cos += cos.item()
        # Cập nhật bộ nhớ và đo thời gian
        start = time.time()
        memory = updater.update(memory, x, y)
        end = time.time()
        total_time += (end - start)
    # Tính giá trị trung bình
    mse_mean = total_mse / steps
    cos_mean = total_cos / steps
    # Độ thay đổi norm của memory (do ban đầu memory=0)
    mem_norm_change = torch.norm(memory).item()
    return mse_mean, cos_mean, total_time, mem_norm_change

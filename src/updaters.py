# src/updaters.py
import torch
from collections import deque

class DeltaUpdater:
    def __init__(self, lr=0.01, reg=0.0):
        self.lr = lr
        self.reg = reg

    def update(self, memory, x, y):
        """
        Update memory (weight matrix) with one sample (x, y) using SGD.
        memory: Tensor of shape (mem_size, dim)
        x: Tensor of shape (dim,)
        y: Tensor of shape (mem_size,)
        """
        # Dự đoán và gradient
        y_pred = memory @ x  # dự đoán kích thước mem_size
        error = y_pred - y
        # Tính gradient của loss MSE + điều chuẩn
        grad = torch.ger(error, x) + self.reg * memory
        # Cập nhật ma trận memory
        memory = memory - self.lr * grad
        return memory

class OmegaUpdater:
    def __init__(self, window=10, reg=0.0):
        self.window = window
        self.reg = reg
        # Sử dụng deque để lưu trữ cửa sổ mẫu
        self.buffer_x = deque(maxlen=window)
        self.buffer_y = deque(maxlen=window)

    def update(self, memory, x, y):
        """
        Update memory (weight matrix) by solving ridge regression on buffered samples.
        memory: current memory (unused here, được tính toán lại hoàn toàn)
        x: current key (dim,)
        y: current value (mem_size,)
        """
        # Thêm mẫu hiện tại vào bộ đệm
        self.buffer_x.append(x)
        self.buffer_y.append(y)
        # Tạo ma trận X (dim x N) và Y (mem_size x N) từ buffer
        X = torch.stack(list(self.buffer_x), dim=1)  # dim x N
        Y = torch.stack(list(self.buffer_y), dim=1)  # mem_size x N
        # Nếu có hệ số điều chuẩn, dùng công thức ridge regression
        if self.reg > 0:
            XXT = X @ X.t()  # (dim x dim)
            I = torch.eye(XXT.size(0), device=XXT.device)
            inv = torch.inverse(XXT + self.reg * I)
            W_new = Y @ X.t() @ inv  # (mem_size x dim)
        else:
            # Nếu không dùng điều chuẩn, dùng pseudo-inverse của X
            X_pinv = torch.pinverse(X)
            W_new = Y @ X_pinv
        return W_new

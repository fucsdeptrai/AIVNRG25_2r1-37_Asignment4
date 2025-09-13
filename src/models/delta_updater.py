import torch

class DeltaUpdater:
    def __init__(self, mem_size: int, dim: int, lr: float = 0.01):
        """
        Khởi tạo DeltaUpdater:
        - self.memory: ma trận bộ nhớ (dim x dim), ban đầu zeros.
        - lr: learning rate cho quy tắc Delta.
        """
        self.mem_size = mem_size
        self.dim = dim
        self.lr = lr
        self.memory = torch.zeros(dim, dim)

    def update(self, kv_pairs):
        """
        Cập nhật memory theo luật Delta, lần lượt với mỗi (k, v).
        Trả về self.memory sau cập nhật.
        """
        if not kv_pairs:
            return self.memory

        for k, v in kv_pairs:
            # Tính dự đoán và lỗi cho cặp hiện tại
            pred = self.memory @ k   # (dim,)
            error = v - pred        # (dim,)
            # Cập nhật: outer(error, k)
            self.memory += self.lr * error.unsqueeze(1) @ k.unsqueeze(0)
        return self.memory
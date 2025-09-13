import torch

class OmegaUpdater:
    def __init__(self, mem_size: int, dim: int):
        """
        Khởi tạo OmegaUpdater:
        - self.memory: ma trận bộ nhớ ban đầu (dim x dim).
        - (mem_size có thể không dùng trực tiếp nếu memory vốn là ma trận vuông dim x dim.)
        """
        self.mem_size = mem_size
        self.dim = dim
        # Khởi tạo memory là ma trận 0 (có thể thay đổi nếu cần)
        self.memory = torch.zeros(dim, dim)

    def update(self, kv_pairs):
        """
        Cập nhật memory bằng cách giải hệ least-squares trên toàn bộ cặp (k,v).
        Trả về self.memory sau cập nhật.
        """
        if not kv_pairs:
            return self.memory

        # Ghép keys và values thành ma trận
        K = torch.stack([k for k,v in kv_pairs])  # shape (window, dim)
        V = torch.stack([v for k,v in kv_pairs])  # shape (window, dim)

        # Giải least squares: tìm W sao cho K @ W ≈ V
        # torch.linalg.lstsq trả về namedtuple với .solution là W (dim x dim)
        sol = torch.linalg.lstsq(K, V)
        W = sol.solution  # kích thước (dim, dim)

        # Đặt memory = W^T để phù hợp phép toán memory @ key = value
        self.memory = W.T
        return self.memory
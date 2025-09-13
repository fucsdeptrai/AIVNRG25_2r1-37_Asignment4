import torch

class SyntheticDataset:
    def __init__(self, mem_size: int, dim: int, window: int, seed: int = 0):
        """
        Khởi tạo tập dữ liệu tổng hợp:
        - mem_size: kích thước bộ nhớ (có thể sử dụng để tạo khóa đặc trưng).
        - dim: chiều của vector key/value.
        - window: số lượng mẫu (cặp key,value) sẽ trả về.
        - seed: hạt giống cho tạo số ngẫu nhiên.
        """
        self.mem_size = mem_size
        self.dim = dim
        self.window = window
        self.seed = seed
        torch.manual_seed(seed)

    def get_data(self):
        """
        Trả về danh sách các tuple (key, value):
        Mỗi key, value là Tensor 1 chiều có chiều dài = dim.
        """
        # Sử dụng torch.randn để tạo các vector ngẫu nhiên
        keys = torch.randn(self.window, self.dim)
        values = torch.randn(self.window, self.dim)
        # Chuyển thành list các tuple (key, value)
        kv_pairs = [(keys[i], values[i]) for i in range(self.window)]
        return kv_pairs
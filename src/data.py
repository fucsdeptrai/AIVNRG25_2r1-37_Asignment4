import torch

def generate_synthetic_data(mem_size, dim, steps, seed=None):
    """
    Generate synthetic key-value data.
    Keys are sampled from a Gaussian distribution.
    Values = A @ key + noise, where A is a random matrix.
    Returns:
      keys: Tensor of shape (steps, dim)
      values: Tensor of shape (steps, mem_size)
    """
    if seed is not None:
        torch.manual_seed(seed)
    # Tạo ma trận biến đổi tuyến tính A (mem_size x dim)
    A = torch.randn(mem_size, dim)
    # Tạo các keys ngẫu nhiên (steps x dim)
    keys = torch.randn(steps, dim)
    # Tính values = A * key + noise (steps x mem_size)
    values = keys @ A.t()
    # Thêm nhiễu Gaussian nhỏ
    noise_std = 0.1
    values += noise_std * torch.randn_like(values)
    return keys, values

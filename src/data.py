import torch
import numpy as np

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

# src/data.py
import numpy as np
import torch

def generate_data(dim: int,
                  steps: int,
                  mem_size: int,
                  dependency: str = "ar",   # "ar", "mix", "patterns"
                  window: int = 5,
                  alpha: float = 0.9,
                  n_patterns: int = 10,
                  noise_scale: float = 0.01,
                  seed: int = 0):
    """
    Generate synthetic key-value sequences where keys are D-dimensional vectors
    with temporal dependencies and values are mem_size-dimensional targets.

    Returns:
       keys: torch.Tensor shape (steps, dim), dtype=torch.float32
       values: torch.Tensor shape (steps, mem_size), dtype=torch.float32

    Parameters:
       dependency:
         - "ar": k_t = alpha * k_{t-1} + sqrt(1-alpha^2)*noise  (auto-regressive)
         - "mix": k_t = sum_{i=0..window-1} w_i * z_{t-i} + noise  (linear mixture using past window)
         - "patterns": emit one of n_patterns base vectors with short persistence
       window:
         - window length used both to generate keys (for "mix") and to build values
           (values will depend on the last `window` keys by default).
    """
    rng = np.random.RandomState(seed)
    dim_value = mem_size
    max_window = max(1, window)

    # ---- Generate keys (shape: steps x dim) ----
    if dependency == "ar":
        keys = np.zeros((steps, dim), dtype=np.float32)
        keys[0] = rng.normal(scale=1.0, size=(dim,))
        # keep variance stable: scale noise by sqrt(1-alpha^2)
        noise_scale_ar = np.sqrt(max(0.0, 1.0 - alpha ** 2))
        for t in range(1, steps):
            noise = rng.normal(scale=noise_scale_ar, size=(dim,))
            keys[t] = alpha * keys[t-1] + noise

    elif dependency == "mix":
        # create base noise stream z_t (we'll index z[t] as current, z[t-1] as previous)
        z = rng.normal(size=(steps + max_window, dim)).astype(np.float32)
        # weights decaying for the past window (w[0] corresponds to most recent z[t])
        w = np.array([np.exp(-i / (max_window / 2.0)) for i in range(max_window)], dtype=np.float32)
        w = w / (w.sum() + 1e-12)
        keys = np.zeros((steps, dim), dtype=np.float32)
        for t in range(steps):
            mix = np.zeros(dim, dtype=np.float32)
            # use past values z[t - i], but offset index so we don't go negative
            for i in range(max_window):
                mix += w[i] * z[t + max_window - 1 - i]   # z[...] aligned so first steps have values
            keys[t] = mix + noise_scale * rng.normal(size=(dim,))

    elif dependency == "patterns":
        base = rng.normal(size=(n_patterns, dim)).astype(np.float32)
        q, _ = np.linalg.qr(base.T)
        patterns = q.T[:n_patterns]
        keys = np.zeros((steps, dim), dtype=np.float32)
        current = rng.randint(n_patterns)
        persistence = max(1, max_window // 2)
        for t in range(steps):
            if rng.rand() < 0.1:
                current = rng.randint(n_patterns)
            keys[t] = patterns[current] + noise_scale * rng.normal(size=(dim,))
            if (t % persistence) == 0 and rng.rand() < 0.2:
                current = rng.randint(n_patterns)
    else:
        raise ValueError("Unknown dependency type: choose 'ar','mix' or 'patterns'")

    # --- create values that depend on the window of keys ---
    # Pad keys so we can always take a full 'max_window' window ending at t
    padded = np.vstack([np.zeros((max_window - 1, dim), dtype=np.float32), keys])  # shape: (steps+max_window-1, dim)

    # We'll map concat(window * dim) -> mem_size with a random linear transform + small noise.
    W_big = rng.normal(scale=0.05, size=(max_window * dim, dim_value)).astype(np.float32)
    values = np.zeros((steps, dim_value), dtype=np.float32)
    for t in range(steps):
        window_vec = padded[t : t + max_window].reshape(-1)  # length = max_window * dim
        values[t] = window_vec.dot(W_big) + 0.01 * rng.normal(size=(dim_value,))

    # convert to torch tensors for direct use in training
    keys_t = torch.from_numpy(keys).to(dtype=torch.float32)
    values_t = torch.from_numpy(values).to(dtype=torch.float32)

    return keys_t, values_t


# quick local test
if __name__ == "__main__":
    ks, vs = generate_data(dim=64, steps=1000, mem_size=128, dependency="mix", window=8, seed=42)
    print("keys", ks.shape, "values", vs.shape)

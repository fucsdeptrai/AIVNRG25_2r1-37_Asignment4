"""
Evaluation helpers: reconstruct value from memory given a key and compute MSE & cosine sim.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def reconstruct_value_from_memory(memory: torch.Tensor, key: torch.Tensor) -> torch.Tensor:
    """memory: (M, D), key: (D,) -> recon (D,)
    recon = softmax((key @ mem.T) / sqrt(D)) @ mem
    """
    D = memory.shape[1]
    logits = (key.unsqueeze(0) @ memory.T) / math.sqrt(D)  # (1, M)
    attn = F.softmax(logits, dim=-1)  # (1, M)
    recon = attn @ memory  # (1, D)
    return recon.squeeze(0)


def eval_kv_reconstruction(memory: torch.Tensor, kv_pairs) -> dict:
    """kv_pairs: list of (key_tensor, value_tensor, pos)
    Returns dict with mean_mse and mean_cosine
    """
    mse_loss = nn.MSELoss()
    cos = nn.CosineSimilarity(dim=0)
    mses = []
    coses = []
    for k, v in kv_pairs:
        recon = reconstruct_value_from_memory(memory, k)
        mses.append(float(mse_loss(recon, v)))
        coses.append(float(cos(recon, v)))
    return {"mse_mean": float(sum(mses) / len(mses)), "cos_mean": float(sum(coses) / len(coses))}


import torch

def compute_effective_rank(tensor_states: torch.Tensor, eps: float = 1e-12) -> float:
    """
    Computes Roy-Vetterli (2007) effective rank:
        erank(X) = exp(-sum(p_i ln p_i)) where p_i = sigma_i / sum(sigma)
    """
    if tensor_states.dim() > 2:
        tensor_states = tensor_states.reshape(-1, tensor_states.shape[-1])
    U, S, V = torch.linalg.svd(tensor_states, full_matrices=False)
    p = S / (S.sum() + eps)
    p = p[p > eps]
    entropy = - (p * torch.log(p)).sum()
    return torch.exp(entropy).item()

def compute_decodability_r2(z_T: torch.Tensor, z_0: torch.Tensor) -> float:
    """
    Measures Fisher linear reconstruction fidelity R^2 from final state z_T back to z_0.
    """
    W_dec = torch.linalg.pinv(z_T) @ z_0
    z0_pred = z_T @ W_dec
    ss_tot = torch.norm(z_0 - z_0.mean(0)) ** 2
    ss_res = torch.norm(z_0 - z0_pred) ** 2
    return (1.0 - (ss_res / (ss_tot + 1e-12))).item()

def compute_tensor_memory_capacity(
    h_k: torch.Tensor,
    h_l: torch.Tensor,
    dim: int
) -> float:
    """
    True Tensor Memory Metric M(k, l):
        M(k, l) = R^2(h_k | h_l) * [min(erank(h_k), erank(h_l)) / D] * (1 - exp(-W(k, l)))
    where W(k, l) = ||h_l - h_k||_F / ||h_k||_F.
    """
    r2 = compute_decodability_r2(h_l, h_k)
    erank_k = compute_effective_rank(h_k)
    erank_l = compute_effective_rank(h_l)
    rank_factor = min(erank_k, erank_l) / dim
    
    work = (torch.norm(h_l - h_k) / (torch.norm(h_k) + 1e-12)).item()
    work_factor = 1.0 - math.exp(-work)
    
    return max(0.0, r2) * rank_factor * work_factor

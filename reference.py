from typing import Callable, Dict

import torch


def rms_norm_reference(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    variance = x.pow(2).mean(dim=-1, keepdim=True)
    normalized = x * torch.rsqrt(variance + eps)
    return normalized * weight


REFERENCE_FUNCTIONS: Dict[str, Callable[..., torch.Tensor]] = {
    "rms_norm": rms_norm_reference,
}


from typing import Dict, Optional

import torch


KERNEL_TYPE = "rms_norm"


def kernel_fn(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    variance = x.pow(2).mean(dim=-1, keepdim=True)
    normalized = x * torch.rsqrt(variance + eps)
    return normalized * weight


def get_inputs(
    device: Optional[str] = None,
    batch_size: int = 8,
    seq_len: int = 32,
    hidden_size: int = 64,
    eps: float = 1e-6,
) -> Dict[str, torch.Tensor]:
    resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.randn(batch_size, seq_len, hidden_size, device=resolved_device, dtype=torch.float32)
    weight = torch.randn(hidden_size, device=resolved_device, dtype=torch.float32)
    return {
        "x": x,
        "weight": weight,
        "eps": eps,
    }


def get_flops(inputs: Dict[str, torch.Tensor]) -> int:
    x = inputs["x"]
    return int(x.numel() * 5)


def get_bytes(inputs: Dict[str, torch.Tensor]) -> int:
    x = inputs["x"]
    weight = inputs["weight"]
    return int((x.numel() + x.numel() + weight.numel()) * x.element_size())


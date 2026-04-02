from typing import Dict, Optional

import torch


KERNEL_TYPE = "softmax"


def kernel_fn(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    return torch.softmax(x, dim=dim)


def get_inputs(
    device: Optional[str] = None,
    batch_size: int = 16,
    seq_len: int = 64,
    hidden_size: int = 128,
    dim: int = -1,
) -> Dict[str, torch.Tensor]:
    resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.randn(batch_size, seq_len, hidden_size, device=resolved_device, dtype=torch.float32)
    return {"x": x, "dim": dim}


def get_flops(inputs: Dict[str, torch.Tensor]) -> int:
    return 5 * inputs["x"].numel()


def get_bytes(inputs: Dict[str, torch.Tensor]) -> int:
    tensor = inputs["x"]
    return 2 * tensor.numel() * tensor.element_size()

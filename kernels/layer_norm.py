from typing import Dict, Optional

import torch


KERNEL_TYPE = "layer_norm"


def kernel_fn(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    eps: float = 1e-5,
) -> torch.Tensor:
    return torch.layer_norm(x, x.shape[-1:], weight=weight, bias=bias, eps=eps)


def get_inputs(
    device: Optional[str] = None,
    batch_size: int = 16,
    seq_len: int = 64,
    hidden_size: int = 128,
    eps: float = 1e-5,
) -> Dict[str, torch.Tensor]:
    resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.randn(batch_size, seq_len, hidden_size, device=resolved_device, dtype=torch.float32)
    weight = torch.randn(hidden_size, device=resolved_device, dtype=torch.float32)
    bias = torch.randn(hidden_size, device=resolved_device, dtype=torch.float32)
    return {"x": x, "weight": weight, "bias": bias, "eps": eps}


def get_flops(inputs: Dict[str, torch.Tensor]) -> int:
    return 7 * inputs["x"].numel()


def get_bytes(inputs: Dict[str, torch.Tensor]) -> int:
    tensors = [inputs["x"], inputs["weight"]]
    bias = inputs.get("bias")
    if bias is not None:
        tensors.append(bias)
    return sum(tensor.numel() * tensor.element_size() for tensor in tensors)

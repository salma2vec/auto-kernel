from typing import Dict, Optional

import torch


KERNEL_TYPE = "linear"


def kernel_fn(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    output = torch.matmul(x, weight.transpose(0, 1))
    if bias is not None:
        output = output + bias
    return output


def get_inputs(
    device: Optional[str] = None,
    batch_size: int = 16,
    seq_len: int = 64,
    hidden_size: int = 128,
    out_features: Optional[int] = None,
) -> Dict[str, torch.Tensor]:
    resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    output_size = hidden_size if out_features is None else out_features
    x = torch.randn(batch_size, seq_len, hidden_size, device=resolved_device, dtype=torch.float32)
    weight = torch.randn(output_size, hidden_size, device=resolved_device, dtype=torch.float32)
    bias = torch.randn(output_size, device=resolved_device, dtype=torch.float32)
    return {"x": x, "weight": weight, "bias": bias}


def get_flops(inputs: Dict[str, torch.Tensor]) -> int:
    batch_size, seq_len, hidden_size = inputs["x"].shape
    out_features = inputs["weight"].shape[0]
    return 2 * batch_size * seq_len * hidden_size * out_features


def get_bytes(inputs: Dict[str, torch.Tensor]) -> int:
    tensors = [inputs["x"], inputs["weight"]]
    bias = inputs.get("bias")
    if bias is not None:
        tensors.append(bias)
    return sum(tensor.numel() * tensor.element_size() for tensor in tensors)

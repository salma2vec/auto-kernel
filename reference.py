from typing import Callable, Dict, Optional

import torch


def rms_norm_reference(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    variance = x.pow(2).mean(dim=-1, keepdim=True)
    normalized = x * torch.rsqrt(variance + eps)
    return normalized * weight


def linear_reference(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    output = torch.matmul(x, weight.transpose(0, 1))
    if bias is not None:
        output = output + bias
    return output


def softmax_reference(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    return torch.softmax(x, dim=dim)


def layer_norm_reference(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    eps: float = 1e-5,
) -> torch.Tensor:
    return torch.layer_norm(x, x.shape[-1:], weight=weight, bias=bias, eps=eps)


REFERENCE_FUNCTIONS: Dict[str, Callable[..., torch.Tensor]] = {
    "rms_norm": rms_norm_reference,
    "linear": linear_reference,
    "softmax": softmax_reference,
    "layer_norm": layer_norm_reference,
}

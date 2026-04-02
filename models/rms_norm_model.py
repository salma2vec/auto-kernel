from typing import Optional, Tuple

import torch
import torch.nn as nn


WORKLOAD_NAME = "rms-norm-model"
MODEL_ATOL = 1e-5
MODEL_RTOL = 1e-4
DEFAULT_INPUT_SHAPE: Tuple[int, int, int] = (4, 128, 512)


class RmsNormLayer(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        return x * torch.rsqrt(variance + self.eps) * self.weight


class RmsNormModel(nn.Module):
    def __init__(self, hidden_size: int = 512) -> None:
        super().__init__()
        self.encoder_norm = RmsNormLayer(hidden_size)
        self.decoder_norm = RmsNormLayer(hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder_norm(x)
        return self.decoder_norm(x)


MODEL_TARGETS = [
    {
        "name": "encoder_norm",
        "module_name": "encoder_norm",
        "kernel_type": "rms_norm",
        "kernel_path": "../kernels/rms_norm.py",
        "weight_attr": "weight",
        "eps_attr": "eps",
    },
    {
        "name": "decoder_norm",
        "module_name": "decoder_norm",
        "kernel_type": "rms_norm",
        "kernel_path": "../kernels/rms_norm.py",
        "weight_attr": "weight",
        "eps_attr": "eps",
    },
]


def _resolve_dtype(dtype: str) -> torch.dtype:
    if not hasattr(torch, dtype):
        raise ValueError(f"Unsupported torch dtype string: {dtype}")
    return getattr(torch, dtype)


def build_model(
    device: str = "cpu",
    dtype: str = "float32",
    class_name: Optional[str] = None,
):
    model_cls = globals().get(class_name or "RmsNormModel", RmsNormModel)
    if not issubclass(model_cls, nn.Module):
        raise TypeError(f"Expected an nn.Module class for class_name={class_name!r}")
    model = model_cls(hidden_size=DEFAULT_INPUT_SHAPE[-1])
    return model.to(device=device, dtype=_resolve_dtype(dtype))


def build_inputs(
    device: str = "cpu",
    input_shape: Optional[Tuple[int, ...]] = None,
    dtype: str = "float32",
):
    shape = tuple(input_shape or DEFAULT_INPUT_SHAPE)
    if len(shape) != 3:
        raise ValueError("RmsNormModel expects input_shape=batch,seq,hidden")
    if shape[-1] != DEFAULT_INPUT_SHAPE[-1]:
        raise ValueError(
            f"RmsNormModel expects hidden size {DEFAULT_INPUT_SHAPE[-1]}, got {shape[-1]}"
        )
    return torch.randn(*shape, device=device, dtype=_resolve_dtype(dtype))

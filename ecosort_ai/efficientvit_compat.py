"""Compatibility helpers for the optional laptop-only EfficientViT tools."""

from __future__ import annotations

import sys
import types
from typing import Any


def install_triton_rms_norm_fallback(torch: Any) -> bool:
    """Provide EfficientViT's RMSNorm symbol when Triton is unavailable.

    The upstream package imports its Triton RMSNorm module even for SAM
    backbones that do not use that layer. Native Windows PyTorch installations
    generally do not provide Triton, so register an inference-safe PyTorch
    implementation before importing EfficientViT.
    """
    try:
        import triton  # noqa: F401

        return False
    except ImportError:
        module_name = "efficientvit.models.nn.triton_rms_norm"
        if module_name in sys.modules:
            return True

        fallback_module = types.ModuleType(module_name)

        class TorchRMSNorm2dFunc:
            @staticmethod
            def apply(x: Any, weight: Any, bias: Any, eps: float) -> Any:
                output = x * torch.rsqrt(
                    torch.square(x).mean(dim=1, keepdim=True) + eps
                )
                if weight is not None:
                    output = output * weight.view(1, -1, 1, 1)
                if bias is not None:
                    output = output + bias.view(1, -1, 1, 1)
                return output

        fallback_module.TritonRMSNorm2dFunc = TorchRMSNorm2dFunc
        fallback_module.__all__ = ["TritonRMSNorm2dFunc"]
        sys.modules[module_name] = fallback_module
        return True

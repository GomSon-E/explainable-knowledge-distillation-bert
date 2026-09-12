"""PyTorch device selection and runtime reporting."""

from __future__ import annotations

import torch


def resolve_device(requested: str = "auto") -> torch.device:
    """Resolve auto/cpu/cuda without silently accepting unavailable CUDA."""

    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if requested not in {"cpu", "cuda"}:
        raise ValueError("device must be 'auto', 'cpu', or 'cuda'")
    return torch.device(requested)


def describe_device(device: torch.device) -> dict[str, object]:
    """Return serializable CUDA availability and selected-device details."""

    summary: dict[str, object] = {
        "cuda_available": torch.cuda.is_available(),
        "device": str(device),
        "cuda_device_count": torch.cuda.device_count(),
    }
    if device.type == "cuda":
        summary["cuda_device_name"] = torch.cuda.get_device_name(device)
    return summary

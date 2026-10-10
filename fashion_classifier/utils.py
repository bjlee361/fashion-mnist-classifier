"""Reproducibility, device selection and checkpoint helpers."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch

from .models import build_model


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device(preferred: str = "auto") -> torch.device:
    """Pick CUDA, then Apple Silicon (MPS), then CPU, unless one is requested."""
    if preferred != "auto":
        return torch.device(preferred)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def save_checkpoint(path: str | Path, model, model_name: str, extra: dict | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"model_name": model_name, "state_dict": model.state_dict(), **(extra or {})},
        path,
    )


def load_checkpoint(path: str | Path, device: torch.device):
    """Rebuild the model recorded in a checkpoint and load its weights."""
    checkpoint = torch.load(path, map_location=device)
    model = build_model(checkpoint["model_name"])
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device).eval()
    return model, checkpoint

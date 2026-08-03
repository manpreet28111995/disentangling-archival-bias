import os

import torch


def get_device():
    """Prefer Apple MPS, then CUDA, then CPU."""
    if torch.backends.mps.is_available():
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def move_inputs(inputs, device):
    return {key: value.to(device) for key, value in inputs.items()}

import torch
import intel_extension_for_pytorch as ipex
def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")

    if ipex.xpu.is_available():
        return torch.device("xpu")

    return torch.device("cpu")

device=get_device()
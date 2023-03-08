'''
Author: HU Zheng
Date: 2023-03-08 20:02:35
LastEditors: HU Zheng
LastEditTime: 2023-03-08 21:34:16
Description: file content
'''
import torch
import intel_extension_for_pytorch as ipex
def get_device():
    if torch.cuda.is_available():
        print("Error! CUDA is used.")
        return torch.device("cuda")

    if ipex.xpu.is_available():
        print("XPU is used.")
        return torch.device("xpu")
    print("Error! CPU is used.")
    return torch.device("cpu")

device=get_device()
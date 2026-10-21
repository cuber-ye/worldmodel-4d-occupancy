import torch


def get_device():
    """双轨设备选择：Colab / 云端走 CUDA，本地 Windows 走 CPU。"""
    if torch.cuda.is_available():          # Colab T4、云端 NVIDIA
        return torch.device("cuda")
    if torch.backends.mps.is_available():  # 将来换 Mac 也能跑
        return torch.device("mps")
    return torch.device("cpu")             # 你本地 RX 6500 XT（无 CUDA）


def device_info():
    d = get_device()
    if d.type == "cuda":
        return f"CUDA  {torch.cuda.get_device_name(0)}  " \
               f"({torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB)"
    return f"{d.type.upper()}  (no GPU acceleration)"
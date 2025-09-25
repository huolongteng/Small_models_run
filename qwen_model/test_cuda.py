import torch

# 检查 CUDA 是否可用
print(torch.cuda.is_available())  # 如果返回 True，表示 PyTorch 已检测到 GPU

# 查看当前 CUDA 设备
if torch.cuda.is_available():
    print("CUDA Device: ", torch.cuda.get_device_name(0))
else:
    print("CUDA is not available.")
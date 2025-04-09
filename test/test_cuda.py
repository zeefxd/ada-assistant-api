import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA dostępna: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA w PyTorch: {torch.version.cuda}")
    print(f"Urządzenie CUDA: {torch.cuda.get_device_name(0)}")
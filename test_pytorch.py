import torch
import torchvision
import matplotlib

print(f"Torch version: {torch.__version__}")
print(f"Torchvision version: {torchvision.__version__}")
print(f"Matplotlib version: {matplotlib.__version__}")

# Check if CUDA (GPU support) is available
print(f"CUDA available: {torch.cuda.is_available()}")
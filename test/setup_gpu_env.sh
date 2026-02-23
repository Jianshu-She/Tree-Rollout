#!/bin/bash

# Setup GPU environment for entropy evaluation
echo "Setting up GPU environment for 8x H200 GPUs..."

# Activate environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate entropy_eval

# Install CUDA-enabled PyTorch
echo "Installing PyTorch with CUDA support..."
pip uninstall -y torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install additional GPU acceleration libraries
pip install accelerate
pip install flash-attn --no-build-isolation

echo "Verifying GPU setup..."
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'CUDA version: {torch.version.cuda}')
print(f'Number of GPUs: {torch.cuda.device_count()}')
for i in range(torch.cuda.device_count()):
    print(f'GPU {i}: {torch.cuda.get_device_name(i)}')
"

echo "GPU environment setup complete!"
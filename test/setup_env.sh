#!/bin/bash

# Activate the new environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate entropy_eval

# Install required packages
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers datasets accelerate
pip install numpy pandas scipy matplotlib seaborn
pip install tqdm jupyter ipykernel
pip install huggingface_hub

# Verify installation
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import transformers; print(f'Transformers version: {transformers.__version__}')"
python -c "import datasets; print(f'Datasets version: {datasets.__version__}')"

echo "Environment setup complete! Use 'conda activate entropy_eval' to activate."
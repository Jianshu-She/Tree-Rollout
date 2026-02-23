#!/usr/bin/env python3
"""
Test script to verify environment setup and run a small entropy evaluation
"""
import sys
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

print("Testing environment setup...")

# Check imports
try:
    import torch
    print(f"✓ PyTorch version: {torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  CUDA devices: {torch.cuda.device_count()}")
except ImportError as e:
    print(f"✗ PyTorch not available: {e}")
    sys.exit(1)

try:
    import transformers
    print(f"✓ Transformers version: {transformers.__version__}")
except ImportError as e:
    print(f"✗ Transformers not available: {e}")
    sys.exit(1)

try:
    import datasets
    print(f"✓ Datasets version: {datasets.__version__}")
except ImportError as e:
    print(f"✗ Datasets not available: {e}")
    sys.exit(1)

try:
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import scipy
    print("✓ All required packages available")
except ImportError as e:
    print(f"✗ Missing package: {e}")
    sys.exit(1)

# Test small model loading (CPU version)
print("\nTesting model loading...")
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    
    # Use a smaller model for testing
    test_model_name = "microsoft/DialoGPT-small"  # Small model for testing
    print(f"Loading test model: {test_model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(test_model_name)
    model = AutoModelForCausalLM.from_pretrained(test_model_name)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print("✓ Model loaded successfully")
    
    # Test entropy calculation
    print("\nTesting entropy calculation...")
    
    def calculate_entropy(logits):
        probs = torch.softmax(logits, dim=-1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)
        return entropy.item()
    
    # Test with a simple input
    test_text = "What is 2 + 2?"
    inputs = tokenizer(test_text, return_tensors="pt", padding=True)
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits[:, -1, :]
        entropy = calculate_entropy(logits)
        print(f"✓ Entropy calculation works: {entropy:.3f}")
    
    print("\n✓ All tests passed! Environment is ready.")
    print("\nNext steps:")
    print("1. Install PyTorch with CUDA support if needed")
    print("2. Run the full evaluation script")
        
except Exception as e:
    print(f"✗ Error during testing: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
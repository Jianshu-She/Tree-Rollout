#!/usr/bin/env python3
"""
Quick test to verify GPU setup and model loading
"""
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
import logging
import warnings
warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    print("🔧 Quick GPU Test for Entropy Evaluation")
    print("=" * 50)
    
    # Check GPU status
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            gpu_name = torch.cuda.get_device_name(i)
            memory_gb = torch.cuda.get_device_properties(i).total_memory / 1024**3
            print(f"  GPU {i}: {gpu_name} ({memory_gb:.0f}GB)")
    
    # Test model loading
    print("\n🚀 Testing Model Loading...")
    model_name = "Qwen/Qwen2.5-7B-Instruct"
    
    try:
        print("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        
        print("Loading model (this may take a few minutes)...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        
        model.eval()
        print("✅ Model loaded successfully!")
        
        # Show GPU memory usage
        print("\n💾 GPU Memory Usage:")
        for i in range(torch.cuda.device_count()):
            if torch.cuda.get_device_properties(i).name:
                memory_used = torch.cuda.memory_allocated(i) / 1024**3
                memory_total = torch.cuda.get_device_properties(i).total_memory / 1024**3
                usage_pct = (memory_used / memory_total) * 100
                print(f"  GPU {i}: {memory_used:.1f}GB / {memory_total:.1f}GB ({usage_pct:.1f}%)")
        
        # Quick test generation
        print("\n🧮 Testing Generation...")
        test_prompt = "What is 15 + 27?"
        inputs = tokenizer(test_prompt, return_tensors="pt")
        
        # Move to appropriate device
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"Test prompt: {test_prompt}")
        print(f"Response: {response}")
        
        print("\n✅ All tests passed! Ready for full evaluation.")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    main()
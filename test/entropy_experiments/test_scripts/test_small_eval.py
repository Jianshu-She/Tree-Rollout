#!/usr/bin/env python3
"""
Test entropy evaluation with a small model first
"""
import torch
import numpy as np
import pandas as pd
import json
import re
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm
import logging
import warnings
warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def calculate_entropy(logits):
    """Calculate entropy from logits"""
    probs = torch.softmax(logits, dim=-1)
    entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)
    return entropy.item()

def generate_with_entropy(model, tokenizer, prompt, max_new_tokens=50):
    """Generate text and track entropy for first N tokens"""
    inputs = tokenizer(prompt, return_tensors="pt", padding=True)
    
    entropies = []
    generated_tokens = []
    
    with torch.no_grad():
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        
        for step in range(min(max_new_tokens, 50)):  # Small number for testing
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits[:, -1, :]
            
            # Calculate entropy
            entropy = calculate_entropy(logits)
            entropies.append(entropy)
            
            # Sample next token
            next_token_id = torch.multinomial(torch.softmax(logits, dim=-1), 1)
            generated_tokens.append(next_token_id.item())
            
            # Update input_ids and attention_mask
            input_ids = torch.cat([input_ids, next_token_id], dim=-1)
            attention_mask = torch.cat([attention_mask, torch.ones(1, 1)], dim=-1)
            
            # Stop if EOS token is generated
            if next_token_id.item() == tokenizer.eos_token_id:
                break
    
    # Decode generated text
    generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    
    return generated_text, entropies

def main():
    print("Testing entropy evaluation with small model...")
    
    # Use a small model for testing
    model_name = "microsoft/DialoGPT-small"
    print(f"Loading model: {model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Test with some simple math problems
    test_problems = [
        "What is 2 + 2?",
        "What is 5 * 3?", 
        "What is 10 - 7?",
        "What is 8 / 2?",
        "What is 3 + 5 * 2?"  # This should be harder
    ]
    
    # Simulate ground truth answers
    ground_truth = ["4", "15", "3", "4", "13"]
    
    results = []
    
    for i, (problem, answer) in enumerate(zip(test_problems, ground_truth)):
        print(f"\nProblem {i+1}: {problem}")
        
        prompt = f"Q: {problem}\nA:"
        generated_text, entropies = generate_with_entropy(model, tokenizer, prompt)
        
        print(f"Generated: {generated_text}")
        print(f"Entropies: {entropies[:5]}...")  # Show first 5
        
        # Simple correctness check (contains the right number)
        is_correct = answer in generated_text
        
        avg_entropy = np.mean(entropies) if entropies else 0.0
        
        results.append({
            'problem': problem,
            'generated': generated_text,
            'ground_truth': answer,
            'is_correct': is_correct,
            'avg_entropy': avg_entropy,
            'entropies': entropies
        })
        
        print(f"Average entropy: {avg_entropy:.3f}, Correct: {is_correct}")
    
    # Simple analysis
    print("\n" + "="*50)
    print("SIMPLE ENTROPY ANALYSIS")
    print("="*50)
    
    df = pd.DataFrame(results)
    
    if len(df) > 0:
        correct_df = df[df['is_correct'] == True]
        incorrect_df = df[df['is_correct'] == False]
        
        print(f"Total problems: {len(df)}")
        print(f"Correct: {len(correct_df)}")
        print(f"Incorrect: {len(incorrect_df)}")
        
        if len(correct_df) > 0:
            print(f"Average entropy (correct): {correct_df['avg_entropy'].mean():.3f}")
        if len(incorrect_df) > 0:
            print(f"Average entropy (incorrect): {incorrect_df['avg_entropy'].mean():.3f}")
        
        # Save results
        with open("test_entropy_results.json", 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print("\nTest completed! Results saved to test_entropy_results.json")
        print("If this works well, you can proceed with the full Qwen evaluation.")
    
    return results

if __name__ == "__main__":
    main()
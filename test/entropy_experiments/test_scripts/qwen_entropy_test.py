#!/usr/bin/env python3
"""
Test Qwen2.5-7B-Instruct with entropy calculation on a few math problems
"""
import torch
import numpy as np
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
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

def generate_with_entropy(model, tokenizer, prompt, max_new_tokens=100, device="cpu"):
    """Generate text and track entropy for first N tokens"""
    inputs = tokenizer(prompt, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    entropies = []
    generated_tokens = []
    
    with torch.no_grad():
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        
        for step in range(min(max_new_tokens, 100)):
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits[:, -1, :]
            
            # Calculate entropy
            entropy = calculate_entropy(logits)
            entropies.append(entropy)
            
            # Sample next token (using greedy for consistency)
            next_token_id = torch.argmax(logits, dim=-1, keepdim=True)
            generated_tokens.append(next_token_id.item())
            
            # Update input_ids and attention_mask
            input_ids = torch.cat([input_ids, next_token_id], dim=-1)
            attention_mask = torch.cat([attention_mask, torch.ones(1, 1).to(device)], dim=-1)
            
            # Stop if EOS token is generated
            if next_token_id.item() == tokenizer.eos_token.id if hasattr(tokenizer, 'eos_token') else tokenizer.eos_token_id:
                break
    
    # Decode generated text
    generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    
    return generated_text, entropies

def extract_answer(text):
    """Simple answer extraction"""
    import re
    
    # Look for numbers
    numbers = re.findall(r'-?\d+(?:\.\d+)?', text)
    if numbers:
        return numbers[-1]  # Return last number found
    
    return text.strip()

def main():
    print("Testing Qwen2.5-7B-Instruct with entropy calculation...")
    
    model_name = "Qwen/Qwen2.5-7B-Instruct"
    device = "cpu"  # Use CPU for now
    
    print(f"Loading model: {model_name}")
    print("This may take a few minutes...")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,  # Use float32 for CPU
            trust_remote_code=True
        ).to(device)
        
        model.eval()
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        print("✓ Model loaded successfully!")
        
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        print("This might be due to memory constraints or network issues.")
        print("Try using a GPU environment or a smaller model.")
        return
    
    # Test problems with known answers
    test_problems = [
        {
            "problem": "What is 15 + 27?",
            "answer": "42"
        },
        {
            "problem": "If a rectangle has length 8 and width 5, what is its area?",
            "answer": "40"
        },
        {
            "problem": "What is 144 divided by 12?", 
            "answer": "12"
        },
        {
            "problem": "What is the square root of 64?",
            "answer": "8"
        },
        {
            "problem": "If x + 5 = 12, what is x?",
            "answer": "7"
        }
    ]
    
    results = []
    
    for i, item in enumerate(test_problems):
        problem = item["problem"]
        ground_truth = item["answer"]
        
        print(f"\n--- Problem {i+1} ---")
        print(f"Q: {problem}")
        
        # Format as chat prompt
        chat_prompt = f"""Please solve this math problem step by step.

Problem: {problem}

Solution:"""
        
        try:
            generated_text, entropies = generate_with_entropy(
                model, tokenizer, chat_prompt, max_new_tokens=100, device=device
            )
            
            # Extract predicted answer
            predicted_answer = extract_answer(generated_text)
            
            # Check correctness (simple string matching)
            is_correct = ground_truth.lower() in generated_text.lower() or ground_truth == predicted_answer
            
            # Calculate entropy statistics
            avg_entropy = np.mean(entropies[:50]) if entropies else 0.0  # First 50 tokens
            max_entropy = np.max(entropies[:50]) if entropies else 0.0
            min_entropy = np.min(entropies[:50]) if entropies else 0.0
            
            result = {
                'problem': problem,
                'ground_truth': ground_truth,
                'generated_text': generated_text,
                'predicted_answer': predicted_answer,
                'is_correct': is_correct,
                'avg_entropy': avg_entropy,
                'max_entropy': max_entropy,
                'min_entropy': min_entropy,
                'entropies': entropies[:50],  # Store first 50 for analysis
                'num_tokens': len(entropies)
            }
            
            results.append(result)
            
            print(f"Generated: {generated_text[:100]}...")  # Show first 100 chars
            print(f"Predicted answer: {predicted_answer}")
            print(f"Correct: {is_correct}")
            print(f"Avg entropy (first 50 tokens): {avg_entropy:.3f}")
            
        except Exception as e:
            print(f"Error processing problem {i+1}: {e}")
            continue
    
    # Analysis
    if results:
        print("\n" + "="*60)
        print("ENTROPY vs CORRECTNESS ANALYSIS")
        print("="*60)
        
        correct_results = [r for r in results if r['is_correct']]
        incorrect_results = [r for r in results if not r['is_correct']]
        
        print(f"Total problems: {len(results)}")
        print(f"Correct answers: {len(correct_results)}")
        print(f"Incorrect answers: {len(incorrect_results)}")
        print(f"Accuracy: {len(correct_results)/len(results):.1%}")
        
        if correct_results:
            correct_avg_entropy = np.mean([r['avg_entropy'] for r in correct_results])
            print(f"Average entropy (correct): {correct_avg_entropy:.3f}")
        
        if incorrect_results:
            incorrect_avg_entropy = np.mean([r['avg_entropy'] for r in incorrect_results])
            print(f"Average entropy (incorrect): {incorrect_avg_entropy:.3f}")
        
        if correct_results and incorrect_results:
            entropy_diff = incorrect_avg_entropy - correct_avg_entropy
            print(f"Entropy difference (incorrect - correct): {entropy_diff:.3f}")
            
            if entropy_diff > 0:
                print("→ Higher entropy associated with incorrect answers")
            else:
                print("→ Lower entropy associated with incorrect answers")
        
        # Save results
        with open("qwen_entropy_test_results.json", 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\nResults saved to qwen_entropy_test_results.json")
        print("This demonstrates the basic functionality!")
        print("For the full evaluation, run the main entropy_eval.py script with a larger dataset.")
    
    else:
        print("No results to analyze.")

if __name__ == "__main__":
    main()
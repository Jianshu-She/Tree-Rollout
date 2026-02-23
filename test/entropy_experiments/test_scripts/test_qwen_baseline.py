#!/usr/bin/env python3
"""
Test Qwen2.5-7B-Instruct baseline performance on GSM8K without entropy tracking
"""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import json
import re

def extract_answer(text):
    """Simple but effective answer extraction"""
    # Look for boxed answers
    boxed_match = re.search(r'\\boxed\{([^}]+)\}', text)
    if boxed_match:
        return boxed_match.group(1).strip()
    
    # Look for explicit statements
    patterns = [
        r'(?:the answer is|final answer is|therefore)[:\s]*\$?([+-]?\d+(?:,\d{3})*(?:\.\d+)?)',
        r'(?:total|makes?|profit)[:\s]*\$?([+-]?\d+(?:,\d{3})*(?:\.\d+)?)\s*(?:dollars?|per day|total)?$',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text.lower())
        if match:
            return match.group(1).replace(',', '')
    
    # Find the last calculation result
    lines = text.split('\n')
    for line in reversed(lines):
        if '=' in line:
            parts = line.split('=')
            if len(parts) >= 2:
                result_text = parts[-1].strip()
                numbers = re.findall(r'\$?([+-]?\d+(?:,\d{3})*(?:\.\d+)?)', result_text)
                if numbers:
                    return numbers[0].replace(',', '')
    
    # Get last substantial number
    all_numbers = re.findall(r'([+-]?\d+(?:,\d{3})*(?:\.\d+)?)', text)
    if all_numbers:
        # Filter out small numbers (likely step numbers)
        substantial = [n for n in all_numbers if float(n.replace(',', '')) > 10]
        if substantial:
            return substantial[-1].replace(',', '')
        return all_numbers[-1].replace(',', '')
    
    return ""

def main():
    print("🎯 Testing Qwen2.5-7B-Instruct BASELINE performance on GSM8K")
    print("(Without entropy tracking, using standard generation)")
    
    # Load model
    model_name = "Qwen/Qwen2.5-7B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load GSM8K problems
    dataset = load_dataset("gsm8k", "main", split="test")
    test_problems = list(dataset.select(range(50)))  # Test on 50 problems
    
    results = []
    correct_count = 0
    
    for i, problem in enumerate(test_problems):
        question = problem['question']
        expected_answer = problem['answer'].split('#### ')[-1].strip()
        
        # Use a clear, direct prompt
        prompt = f"""Solve this math problem step by step and provide the final numerical answer.

Problem: {question}

Solution:"""
        
        # Standard generation (no entropy tracking)
        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=300,
                do_sample=False,  # Greedy decoding for consistency
                temperature=1.0,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        # Decode only the generated part
        generated_tokens = outputs[0][inputs['input_ids'].shape[1]:]
        generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        # Extract answer
        predicted_answer = extract_answer(generated_text)
        
        # Check correctness
        try:
            is_correct = abs(float(predicted_answer) - float(expected_answer)) < 1e-6
        except:
            is_correct = predicted_answer.strip() == expected_answer.strip()
        
        if is_correct:
            correct_count += 1
        
        result = {
            'problem_id': i,
            'question': question,
            'expected': expected_answer,
            'generated': generated_text,
            'predicted': predicted_answer,
            'correct': is_correct
        }
        results.append(result)
        
        # Progress update
        if (i + 1) % 10 == 0:
            current_accuracy = correct_count / (i + 1)
            print(f"Processed {i+1}/50 problems. Current accuracy: {current_accuracy:.1%}")
    
    # Final results
    final_accuracy = correct_count / len(results)
    print(f"\n{'='*60}")
    print(f"BASELINE GSM8K PERFORMANCE")
    print(f"{'='*60}")
    print(f"Total problems: {len(results)}")
    print(f"Correct: {correct_count}")
    print(f"Accuracy: {final_accuracy:.1%}")
    
    # Save results
    with open('baseline_gsm8k_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Show some examples
    print(f"\n📋 Sample Results:")
    for i in range(min(5, len(results))):
        r = results[i]
        print(f"\nExample {i+1}:")
        print(f"  Q: {r['question'][:60]}...")
        print(f"  Expected: {r['expected']}")
        print(f"  Generated: {r['generated'][:100]}...")
        print(f"  Predicted: {r['predicted']}")
        print(f"  Correct: {r['correct']}")
    
    # Show failures to understand what's happening
    failures = [r for r in results if not r['correct']]
    if failures:
        print(f"\n❌ Some Failure Examples:")
        for i, r in enumerate(failures[:3]):
            print(f"\nFailure {i+1}:")
            print(f"  Q: {r['question'][:60]}...")
            print(f"  Expected: {r['expected']}")
            print(f"  Generated: {r['generated'][:150]}...")
            print(f"  Predicted: {r['predicted']}")

if __name__ == "__main__":
    main()
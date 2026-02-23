#!/usr/bin/env python3
"""
Simple test to check Qwen2.5-7B-Instruct actual performance on GSM8K
"""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

def main():
    print("🔍 Testing Qwen2.5-7B-Instruct on a few GSM8K problems to check actual capability...")
    
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
    
    # Load a few GSM8K problems
    dataset = load_dataset("gsm8k", "main", split="test")
    test_problems = list(dataset.select(range(5)))
    
    for i, problem in enumerate(test_problems):
        question = problem['question']
        answer = problem['answer'].split('#### ')[-1]  # Extract final answer
        
        print(f"\n{'='*60}")
        print(f"Problem {i+1}:")
        print(f"Q: {question}")
        print(f"Expected Answer: {answer}")
        
        # Try different prompting approaches
        prompts = [
            f"Solve this step by step and give the final numerical answer:\n\n{question}\n\nSolution:",
            f"Question: {question}\n\nLet me solve this step by step and provide the final answer as a number:\n\nStep-by-step solution:",
            f"Math problem: {question}\n\nI need to find the numerical answer. Let me work through this:\n\nSolution:"
        ]
        
        for j, prompt in enumerate(prompts):
            print(f"\nPrompt Style {j+1}:")
            inputs = tokenizer(prompt, return_tensors="pt")
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=200,
                    do_sample=False,
                    temperature=0.1,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )
            
            # Decode only the generated part
            generated_tokens = outputs[0][inputs['input_ids'].shape[1]:]
            generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
            
            print(f"Generated: {generated_text}")
            
            # Simple answer extraction - look for the last number
            import re
            numbers = re.findall(r'\b\d+(?:\.\d+)?\b', generated_text)
            if numbers:
                extracted = numbers[-1]
                is_correct = str(extracted) == str(answer)
                print(f"Extracted: {extracted} | Expected: {answer} | Correct: {is_correct}")
            else:
                print(f"No number found | Expected: {answer}")
        
        if i >= 2:  # Just test first 3 problems
            break

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Quick test of multi-sample entropy evaluation on a few questions
"""
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import json
import re

def calculate_entropy(logits):
    """Calculate entropy from logits"""
    with torch.no_grad():
        probs = torch.softmax(logits, dim=-1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)
        return entropy.cpu().item()

def extract_answer(text):
    """Simple answer extraction"""
    # Look for explicit statements
    patterns = [
        r'(?:the answer is|final answer is|therefore)[:\s]*\$?([+-]?\d+(?:,\d{3})*(?:\.\d+)?)',
        r'(?:total|makes?|profit)[:\s]*\$?([+-]?\d+(?:,\d{3})*(?:\.\d+)?)\s*(?:dollars?|per day|total)?$',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text.lower())
        if match:
            return match.group(1).replace(',', '')
    
    # Get last substantial number
    all_numbers = re.findall(r'([+-]?\d+(?:,\d{3})*(?:\.\d+)?)', text)
    if all_numbers:
        substantial = [n for n in all_numbers if float(n.replace(',', '')) > 10]
        if substantial:
            return substantial[-1].replace(',', '')
        return all_numbers[-1].replace(',', '')
    
    return ""

def check_correctness(predicted, expected):
    """Check if predicted matches expected"""
    try:
        pred_clean = predicted.replace('$', '').replace(',', '').strip()
        exp_clean = expected.replace('$', '').replace(',', '').strip()
        
        if not pred_clean or not exp_clean:
            return False
        
        return abs(float(pred_clean) - float(exp_clean)) < 1e-6
    except:
        return pred_clean.lower() == exp_clean.lower()

def main():
    print("🔍 Testing multi-sample entropy evaluation on 3 questions with 8 samples each...")
    
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
    test_problems = list(dataset.select(range(3)))
    
    results = []
    
    for q_idx, problem in enumerate(test_problems):
        question = problem['question']
        expected_answer = problem['answer'].split('#### ')[-1].strip()
        
        prompt = f"""Solve this math problem step by step and provide the final numerical answer.

Problem: {question}

Solution:"""
        
        print(f"\n{'='*60}")
        print(f"Question {q_idx+1}: {question[:80]}...")
        print(f"Expected: {expected_answer}")
        
        # Generate 8 samples
        question_results = []
        correct_count = 0
        
        for sample_idx in range(8):
            inputs = tokenizer(prompt, return_tensors="pt")
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=200,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=tokenizer.eos_token_id,
                    output_scores=True,
                    return_dict_in_generate=True
                )
            
            # Extract text and entropy
            generated_tokens = outputs.sequences[0][inputs['input_ids'].shape[1]:]
            generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
            
            # Calculate entropies for different ranges
            entropies = []
            if hasattr(outputs, 'scores') and outputs.scores:
                for score in outputs.scores:
                    entropy = calculate_entropy(score[0])
                    entropies.append(entropy)
            
            # Extract answer
            predicted_answer = extract_answer(generated_text)
            is_correct = check_correctness(predicted_answer, expected_answer)
            
            if is_correct:
                correct_count += 1
            
            # Calculate entropy stats for different ranges
            entropy_stats = {}
            for token_range, range_name in [(50, "first_50"), (100, "first_100"), (200, "first_200"), (-1, "all")]:
                if token_range == -1:
                    tokens_subset = entropies
                else:
                    tokens_subset = entropies[:token_range]
                
                if tokens_subset:
                    entropy_stats[f"avg_entropy_{range_name}"] = np.mean(tokens_subset)
                    entropy_stats[f"max_entropy_{range_name}"] = np.max(tokens_subset)
                    entropy_stats[f"std_entropy_{range_name}"] = np.std(tokens_subset)
            
            question_results.append({
                'sample_id': sample_idx,
                'predicted_answer': predicted_answer,
                'is_correct': is_correct,
                'generated_text': generated_text[:100] + "...",
                **entropy_stats
            })
            
            print(f"  Sample {sample_idx+1}: {predicted_answer} ({'✓' if is_correct else '✗'}) - Entropy: {entropy_stats.get('avg_entropy_first_100', 0):.3f}")
        
        question_accuracy = correct_count / 8
        print(f"\nQuestion Accuracy: {correct_count}/8 = {question_accuracy:.1%}")
        
        # Calculate question-level entropy stats
        question_entropy_stats = {}
        for range_name in ["first_50", "first_100", "first_200", "all"]:
            avg_entropies = [s[f"avg_entropy_{range_name}"] for s in question_results if f"avg_entropy_{range_name}" in s]
            if avg_entropies:
                question_entropy_stats[f"question_avg_entropy_{range_name}"] = np.mean(avg_entropies)
                question_entropy_stats[f"question_entropy_variance_{range_name}"] = np.var(avg_entropies)
        
        results.append({
            'question_id': q_idx,
            'question': question,
            'expected_answer': expected_answer,
            'question_accuracy': question_accuracy,
            'correct_count': correct_count,
            'samples': question_results,
            **question_entropy_stats
        })
    
    # Simple analysis
    print(f"\n{'='*60}")
    print("SUMMARY ANALYSIS")
    print(f"{'='*60}")
    
    accuracies = [r['question_accuracy'] for r in results]
    entropies_100 = [r['question_avg_entropy_first_100'] for r in results if 'question_avg_entropy_first_100' in r]
    
    print(f"Question accuracies: {[f'{acc:.1%}' for acc in accuracies]}")
    print(f"Average entropies (first 100): {[f'{ent:.3f}' for ent in entropies_100]}")
    
    if len(accuracies) >= 2 and len(entropies_100) >= 2:
        from scipy.stats import pearsonr
        corr, p_val = pearsonr(entropies_100, accuracies)
        print(f"Correlation (entropy vs accuracy): r={corr:.3f}, p={p_val:.3f}")
    
    # Save results
    with open('test_multi_sample_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n✅ Test completed! Results saved to test_multi_sample_results.json")

if __name__ == "__main__":
    main()
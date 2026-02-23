#!/usr/bin/env python3
"""
Debug the evaluation to check for issues
"""
import json
import numpy as np

# Load results
with open('/mnt/weka/home/jianshu.she/MCTS/entropy_results_efficient_20250730_011406/all_results.json') as f:
    results = json.load(f)

print("🔍 Debugging Multi-Sample Evaluation Results")
print("=" * 60)

# Check overall statistics
total_questions = len(results)
total_samples = sum(len(r['samples']) for r in results)
print(f"Total questions: {total_questions}")
print(f"Total samples: {total_samples}")

# Check entropy statistics
all_entropies = []
zero_entropy_count = 0

for r in results:
    for sample in r['samples']:
        entropies = sample['entropies']
        all_entropies.extend(entropies)
        zero_entropy_count += sum(1 for e in entropies if e == 0.0)

print(f"\nEntropy Statistics:")
print(f"Total entropy values: {len(all_entropies)}")
print(f"Zero entropy count: {zero_entropy_count} ({zero_entropy_count/len(all_entropies)*100:.1f}%)")
print(f"Entropy range: {min(all_entropies):.3f} to {max(all_entropies):.3f}")
print(f"Average entropy: {np.mean(all_entropies):.3f}")

# Check a few specific examples
print(f"\n📋 Detailed Sample Analysis:")
for i in range(min(3, len(results))):
    r = results[i]
    print(f"\nQuestion {i+1}: {r['question'][:80]}...")
    print(f"Expected: {r['expected_answer']}")
    print(f"Question accuracy: {r['correct_count']}/{r['samples_per_question']} = {r['question_accuracy']:.1%}")
    
    # Check individual samples
    for j, sample in enumerate(r['samples'][:3]):  # First 3 samples
        print(f"  Sample {j+1}:")
        print(f"    Generated: {sample['generated_text'][:100]}...")
        print(f"    Predicted: '{sample['predicted_answer']}'")
        print(f"    Correct: {sample['is_correct']}")
        
        # Check entropy pattern
        entropies = sample['entropies'][:20]  # First 20 tokens
        zero_count = sum(1 for e in entropies if e == 0.0)
        print(f"    Entropies (first 20): {[f'{e:.3f}' for e in entropies]}")
        print(f"    Zero entropies in first 20: {zero_count}/20")

# Check accuracy calculation manually for first question
print(f"\n🔍 Manual Accuracy Verification for Question 1:")
q1 = results[0]
expected = q1['expected_answer']
print(f"Expected answer: '{expected}'")

correct_manual = 0
for i, sample in enumerate(q1['samples']):
    predicted = sample['predicted_answer']
    is_correct = sample['is_correct']
    
    # Manual check
    try:
        manual_correct = abs(float(predicted) - float(expected)) < 1e-6
    except:
        manual_correct = predicted.strip().lower() == expected.strip().lower()
    
    print(f"Sample {i+1}: predicted='{predicted}', correct={is_correct}, manual_check={manual_correct}")
    
    if manual_correct:
        correct_manual += 1

print(f"Manual accuracy: {correct_manual}/{len(q1['samples'])} = {correct_manual/len(q1['samples']):.1%}")
print(f"Recorded accuracy: {q1['correct_count']}/{q1['samples_per_question']} = {q1['question_accuracy']:.1%}")

# Check token range calculations
print(f"\n📊 Token Range Analysis:")
for token_range in [50, 100, 200]:
    range_name = f"first_{token_range}"
    avg_key = f"question_avg_entropy_{range_name}"
    
    if avg_key in q1:
        print(f"{range_name}: {q1[avg_key]:.3f}")
        
        # Manual calculation for first sample
        sample_entropies = q1['samples'][0]['entropies'][:token_range]
        manual_avg = np.mean(sample_entropies) if sample_entropies else 0.0
        print(f"  Manual calc for sample 1: {manual_avg:.3f}")
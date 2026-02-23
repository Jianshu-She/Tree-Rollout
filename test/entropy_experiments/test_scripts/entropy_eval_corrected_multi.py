#!/usr/bin/env python3
"""
Corrected multi-sample entropy evaluation fixing the major issues
"""
import torch
import numpy as np
import pandas as pd
import json
import re
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
import argparse
from tqdm import tqdm
import logging
from typing import List, Dict, Tuple, Optional
import warnings
warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CorrectedMultiSampleEvaluator:
    def __init__(self, model_name: str = "Qwen/Qwen2.5-7B-Instruct", device_map: str = "auto"):
        self.model_name = model_name
        
        torch.cuda.empty_cache()
        
        logger.info(f"Loading model {model_name}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, 
            trust_remote_code=True,
            padding_side="left"
        )
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map=device_map,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        
        self.model.eval()
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def calculate_entropy(self, logits: torch.Tensor) -> float:
        """Calculate entropy from logits - fixed version"""
        with torch.no_grad():
            # Add small epsilon to avoid log(0)
            probs = torch.softmax(logits, dim=-1)
            
            # Only calculate entropy for non-zero probabilities
            mask = probs > 1e-10
            entropy = -torch.sum(probs[mask] * torch.log(probs[mask] + 1e-10))
            
            return entropy.cpu().item()

    @torch.no_grad()
    def generate_single_sample(self, prompt: str, max_new_tokens: int = 200) -> Tuple[str, List[float]]:
        """
        Generate a single sample with proper entropy tracking
        Fixed to use standard generation with output_scores
        """
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,  # Enable sampling
                temperature=0.8,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                output_scores=True,  # Get logits for entropy calculation
                return_dict_in_generate=True
            )
        
        # Extract the generated text
        generated_tokens = outputs.sequences[0][inputs['input_ids'].shape[1]:]
        generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        # Extract entropy from scores - FIXED
        entropies = []
        if hasattr(outputs, 'scores') and outputs.scores:
            for score in outputs.scores:
                entropy = self.calculate_entropy(score[0])  # [0] because batch_size=1
                entropies.append(entropy)
        
        return generated_text, entropies

    def generate_multiple_samples(self, prompt: str, num_samples: int = 12, max_new_tokens: int = 200) -> List[Tuple[str, List[float]]]:
        """
        Generate multiple samples one by one to ensure diversity
        """
        results = []
        
        for sample_idx in range(num_samples):
            generated_text, entropies = self.generate_single_sample(prompt, max_new_tokens)
            results.append((generated_text, entropies))
            
            # Memory cleanup between samples
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        return results

    def extract_answer(self, text: str) -> str:
        """Extract answer using improved method"""
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

    def check_answer_correctness(self, predicted: str, expected: str) -> bool:
        """Check if predicted answer matches expected"""
        try:
            pred_clean = predicted.replace('$', '').replace(',', '').strip()
            exp_clean = expected.replace('$', '').replace(',', '').strip()
            
            if not pred_clean or not exp_clean:
                return False
            
            # Exact string match
            if pred_clean.lower() == exp_clean.lower():
                return True
            
            # Numerical comparison
            pred_num = float(pred_clean)
            exp_num = float(exp_clean)
            
            return abs(pred_num - exp_num) < 1e-6
            
        except (ValueError, TypeError):
            return False

    def format_prompt(self, question: str) -> str:
        """Format prompt for mathematical reasoning"""
        return f"""Solve this math problem step by step and provide the final numerical answer.

Problem: {question}

Solution:"""

    def load_gsm8k(self) -> List[Dict]:
        """Load GSM8K dataset"""
        try:
            logger.info("Loading GSM8K test set...")
            dataset = load_dataset("gsm8k", "main", split="test")
            problems = list(dataset)
            logger.info(f"Loaded {len(problems)} problems from GSM8K")
            return problems
        except Exception as e:
            logger.error(f"Failed to load GSM8K: {e}")
            return []

    def calculate_entropy_stats(self, entropies: List[float], token_ranges: List[int]) -> Dict:
        """Calculate entropy statistics for different token ranges"""
        stats = {}
        
        for token_range in token_ranges:
            if token_range == -1:  # All tokens
                tokens_subset = entropies
                range_name = "all"
            else:
                tokens_subset = entropies[:token_range]
                range_name = f"first_{token_range}"
            
            if tokens_subset:
                stats[f"avg_entropy_{range_name}"] = np.mean(tokens_subset)
                stats[f"max_entropy_{range_name}"] = np.max(tokens_subset)
                stats[f"min_entropy_{range_name}"] = np.min(tokens_subset)
                stats[f"std_entropy_{range_name}"] = np.std(tokens_subset)
                stats[f"num_tokens_{range_name}"] = len(tokens_subset)
            else:
                stats[f"avg_entropy_{range_name}"] = 0.0
                stats[f"max_entropy_{range_name}"] = 0.0
                stats[f"min_entropy_{range_name}"] = 0.0
                stats[f"std_entropy_{range_name}"] = 0.0
                stats[f"num_tokens_{range_name}"] = 0
        
        return stats

    def evaluate_dataset(self, problems: List[Dict], max_questions: int = 30, samples_per_question: int = 10) -> List[Dict]:
        """Evaluate model with multiple samples per question - CORRECTED VERSION"""
        results = []
        token_ranges = [50, 100, 200, -1]  # -1 means all tokens
        
        problems_subset = problems[:max_questions]
        
        for q_idx, problem in enumerate(tqdm(problems_subset, desc="Processing questions")):
            question = problem['question']
            expected_answer = problem['answer'].split('#### ')[-1].strip()
            
            prompt = self.format_prompt(question)
            
            try:
                logger.info(f"Question {q_idx+1}/{max_questions}: Generating {samples_per_question} samples...")
                
                # Generate multiple samples for this question
                sample_results = self.generate_multiple_samples(prompt, samples_per_question)
                
                # Process each sample
                question_samples = []
                correct_count = 0
                
                for sample_idx, (generated_text, entropies) in enumerate(sample_results):
                    # Extract predicted answer
                    predicted_answer = self.extract_answer(generated_text)
                    
                    # Check correctness
                    is_correct = self.check_answer_correctness(predicted_answer, expected_answer)
                    if is_correct:
                        correct_count += 1
                    
                    # Calculate entropy statistics for different token ranges
                    entropy_stats = self.calculate_entropy_stats(entropies, token_ranges)
                    
                    sample_result = {
                        'question_id': q_idx,
                        'sample_id': sample_idx,
                        'question': question,
                        'expected_answer': expected_answer,
                        'generated_text': generated_text,
                        'predicted_answer': predicted_answer,
                        'is_correct': is_correct,
                        'entropies': entropies,
                        **entropy_stats
                    }
                    
                    question_samples.append(sample_result)
                
                # Calculate per-question accuracy
                question_accuracy = correct_count / samples_per_question
                
                # Calculate average entropy statistics across all samples for this question
                question_entropy_stats = {}
                for token_range in token_ranges:
                    range_name = "all" if token_range == -1 else f"first_{token_range}"
                    
                    # Average entropy stats across samples
                    avg_entropies = [s[f"avg_entropy_{range_name}"] for s in question_samples if s[f"num_tokens_{range_name}"] > 0]
                    max_entropies = [s[f"max_entropy_{range_name}"] for s in question_samples if s[f"num_tokens_{range_name}"] > 0]
                    std_entropies = [s[f"std_entropy_{range_name}"] for s in question_samples if s[f"num_tokens_{range_name}"] > 0]
                    
                    if avg_entropies:
                        question_entropy_stats[f"question_avg_entropy_{range_name}"] = np.mean(avg_entropies)
                        question_entropy_stats[f"question_max_entropy_{range_name}"] = np.mean(max_entropies)
                        question_entropy_stats[f"question_std_entropy_{range_name}"] = np.mean(std_entropies)
                        question_entropy_stats[f"question_entropy_variance_{range_name}"] = np.var(avg_entropies)  # Variance across samples
                    else:
                        question_entropy_stats[f"question_avg_entropy_{range_name}"] = 0.0
                        question_entropy_stats[f"question_max_entropy_{range_name}"] = 0.0
                        question_entropy_stats[f"question_std_entropy_{range_name}"] = 0.0
                        question_entropy_stats[f"question_entropy_variance_{range_name}"] = 0.0
                
                # Store question-level result
                question_result = {
                    'question_id': q_idx,
                    'question': question,
                    'expected_answer': expected_answer,
                    'samples_per_question': samples_per_question,
                    'correct_count': correct_count,
                    'question_accuracy': question_accuracy,
                    'samples': question_samples,
                    **question_entropy_stats
                }
                
                results.append(question_result)
                
                # Progress update
                overall_accuracy = np.mean([r['question_accuracy'] for r in results])
                logger.info(f"Question {q_idx+1}: {correct_count}/{samples_per_question} correct ({question_accuracy:.1%}). Overall avg: {overall_accuracy:.1%}")
                
            except Exception as e:
                logger.error(f"Error processing question {q_idx}: {e}")
                continue
                
            # Memory cleanup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        return results

    def analyze_results(self, results: List[Dict]) -> Dict:
        """Analyze correlation between entropy and per-question accuracy"""
        if not results:
            return {}
        
        # Create DataFrame with question-level data
        question_data = []
        for r in results:
            question_data.append({
                'question_id': r['question_id'],
                'question_accuracy': r['question_accuracy'],
                'correct_count': r['correct_count'],
                **{k: v for k, v in r.items() if k.startswith('question_')}
            })
        
        df = pd.DataFrame(question_data)
        token_ranges = ["first_50", "first_100", "first_200", "all"]
        
        analysis = {
            'total_questions': len(df),
            'total_samples': sum(r['samples_per_question'] for r in results),
            'overall_accuracy': df['question_accuracy'].mean(),
            'accuracy_std': df['question_accuracy'].std(),
            'perfect_questions': len(df[df['question_accuracy'] == 1.0]),
            'zero_questions': len(df[df['question_accuracy'] == 0.0]),
        }
        
        # Check entropy validity
        all_entropies = []
        zero_entropy_count = 0
        
        for r in results:
            for sample in r['samples']:
                entropies = sample['entropies']
                all_entropies.extend(entropies)
                zero_entropy_count += sum(1 for e in entropies if abs(e) < 1e-6)
        
        analysis['entropy_stats'] = {
            'total_entropy_values': len(all_entropies),
            'zero_entropy_percentage': zero_entropy_count / len(all_entropies) * 100 if all_entropies else 0,
            'entropy_range': [min(all_entropies), max(all_entropies)] if all_entropies else [0, 0],
            'avg_entropy': np.mean(all_entropies) if all_entropies else 0
        }
        
        # Correlation analysis for each token range
        correlations = {}
        for token_range in token_ranges:
            avg_entropy_col = f"question_avg_entropy_{token_range}"
            max_entropy_col = f"question_max_entropy_{token_range}"
            std_entropy_col = f"question_std_entropy_{token_range}"
            var_entropy_col = f"question_entropy_variance_{token_range}"
            
            if avg_entropy_col in df.columns and len(df) > 5:
                # Correlations with question accuracy
                pearson_avg, p_avg = pearsonr(df[avg_entropy_col], df['question_accuracy'])
                pearson_max, p_max = pearsonr(df[max_entropy_col], df['question_accuracy'])
                pearson_std, p_std = pearsonr(df[std_entropy_col], df['question_accuracy'])
                pearson_var, p_var = pearsonr(df[var_entropy_col], df['question_accuracy'])
                
                spearman_avg, sp_avg = spearmanr(df[avg_entropy_col], df['question_accuracy'])
                
                correlations[token_range] = {
                    'avg_entropy_mean': df[avg_entropy_col].mean(),
                    'max_entropy_mean': df[max_entropy_col].mean(),
                    'std_entropy_mean': df[std_entropy_col].mean(),
                    'entropy_variance_mean': df[var_entropy_col].mean(),
                    
                    'pearson_avg_entropy': pearson_avg,
                    'p_value_avg_entropy': p_avg,
                    'pearson_max_entropy': pearson_max,
                    'p_value_max_entropy': p_max,
                    'pearson_std_entropy': pearson_std,
                    'p_value_std_entropy': p_std,
                    'pearson_entropy_variance': pearson_var,
                    'p_value_entropy_variance': p_var,
                    
                    'spearman_avg_entropy': spearman_avg,
                    'spearman_p_value': sp_avg,
                }
        
        analysis['correlations'] = correlations
        
        return analysis


def main():
    parser = argparse.ArgumentParser(description="CORRECTED multi-sample entropy evaluation")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct", help="Model name")
    parser.add_argument("--max_questions", type=int, default=30, help="Max questions to evaluate")
    parser.add_argument("--samples_per_question", type=int, default=10, help="Samples per question")
    parser.add_argument("--output_dir", default="./entropy_results_corrected_multi", help="Output directory")
    
    args = parser.parse_args()
    
    if not torch.cuda.is_available():
        logger.error("No GPUs available!")
        return
    
    # Create output directory
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"{args.output_dir}_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    evaluator = CorrectedMultiSampleEvaluator(args.model)
    
    # Load GSM8K dataset
    gsm8k_problems = evaluator.load_gsm8k()
    if not gsm8k_problems:
        logger.error("Failed to load GSM8K dataset")
        return
    
    logger.info(f"Starting CORRECTED multi-sample evaluation: {args.max_questions} questions × {args.samples_per_question} samples")
    results = evaluator.evaluate_dataset(gsm8k_problems, args.max_questions, args.samples_per_question)
    
    if results:
        logger.info("Analyzing results...")
        analysis = evaluator.analyze_results(results)
        
        print(f"\n{'='*80}")
        print("CORRECTED MULTI-SAMPLE ENTROPY vs QUESTION ACCURACY ANALYSIS")
        print(f"{'='*80}")
        print(f"Model: {args.model}")
        print(f"Questions evaluated: {analysis['total_questions']}")
        print(f"Total samples generated: {analysis['total_samples']}")
        print(f"Samples per question: {args.samples_per_question}")
        print(f"Overall average accuracy: {analysis['overall_accuracy']:.1%} ± {analysis['accuracy_std']:.1%}")
        print(f"Perfect questions (100% accuracy): {analysis['perfect_questions']}")
        print(f"Zero questions (0% accuracy): {analysis['zero_questions']}")
        
        # Show entropy validation
        entropy_stats = analysis['entropy_stats']
        print(f"\n📊 ENTROPY VALIDATION:")
        print(f"Total entropy values: {entropy_stats['total_entropy_values']}")
        print(f"Zero entropy percentage: {entropy_stats['zero_entropy_percentage']:.1f}%")
        print(f"Entropy range: {entropy_stats['entropy_range'][0]:.3f} to {entropy_stats['entropy_range'][1]:.3f}")
        print(f"Average entropy: {entropy_stats['avg_entropy']:.3f}")
        
        # Show correlations for each token range
        if 'correlations' in analysis:
            print(f"\n📈 CORRELATION ANALYSIS:")
            print(f"{'Token Range':<15} {'Avg Entropy':<15} {'Max Entropy':<15} {'Entropy Var':<15}")
            print("-" * 60)
            
            for token_range, corr_data in analysis['correlations'].items():
                avg_corr = corr_data['pearson_avg_entropy']
                avg_p = corr_data['p_value_avg_entropy']
                max_corr = corr_data['pearson_max_entropy']
                max_p = corr_data['p_value_max_entropy']
                var_corr = corr_data['pearson_entropy_variance']
                var_p = corr_data['p_value_entropy_variance']
                
                print(f"{token_range:<15} r={avg_corr:>6.3f} p={avg_p:.3f} r={max_corr:>6.3f} p={max_p:.3f} r={var_corr:>6.3f} p={var_p:.3f}")
            
            # Highlight significant findings
            significant_findings = []
            for token_range, corr_data in analysis['correlations'].items():
                for measure in ['avg_entropy', 'max_entropy', 'entropy_variance']:
                    corr = corr_data[f'pearson_{measure}']
                    p_val = corr_data[f'p_value_{measure}']
                    if p_val < 0.05 and abs(corr) > 0.3:
                        significant_findings.append(f"{measure} ({token_range}): r={corr:.3f}, p={p_val:.3f}")
            
            if significant_findings:
                print(f"\n🎯 SIGNIFICANT FINDINGS (p < 0.05, |r| > 0.3):")
                for finding in significant_findings:
                    print(f"  ✓ {finding}")
            else:
                print(f"\n📈 No strong significant correlations found.")
        
        # Save results
        with open(f"{output_dir}/all_results.json", 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        with open(f"{output_dir}/analysis.json", 'w') as f:
            json.dump(analysis, f, indent=2, default=str)
        
        print(f"\n📁 Results saved to: {output_dir}/")
        
        # Show some example questions
        print(f"\n📋 Sample Question Results:")
        for i in range(min(3, len(results))):
            r = results[i]
            print(f"\nQuestion {i+1}: {r['question'][:80]}...")
            print(f"  Expected: {r['expected_answer']}")
            print(f"  Accuracy: {r['correct_count']}/{r['samples_per_question']} = {r['question_accuracy']:.1%}")
            print(f"  Avg entropy (first 100): {r.get('question_avg_entropy_first_100', 0):.3f}")
            print(f"  Sample answers: {[s['predicted_answer'] for s in r['samples'][:3]]}")
        
    else:
        logger.error("No results to analyze.")

if __name__ == "__main__":
    main()
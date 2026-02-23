#!/usr/bin/env python3
"""
Corrected entropy evaluation that preserves model performance while tracking entropy
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

class CorrectedEntropyEvaluator:
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
        """Calculate entropy from logits"""
        with torch.no_grad():
            probs = torch.softmax(logits, dim=-1)
            entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)
            return entropy.cpu().item()

    @torch.no_grad()
    def generate_with_entropy_tracking(self, prompt: str, max_new_tokens: int = 300) -> Tuple[str, List[float]]:
        """
        Generate text using model's standard generation but extract entropy from first 100 tokens
        This preserves the model's performance while still tracking entropy
        """
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        # First, do standard generation to get the full response (for performance)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,  # Greedy for consistency
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                output_scores=True,  # This gives us the logits!
                return_dict_in_generate=True
            )
        
        # Extract the generated text
        generated_tokens = outputs.sequences[0][inputs['input_ids'].shape[1]:]
        generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        # Extract entropy from the first 100 tokens using the scores
        entropies = []
        if hasattr(outputs, 'scores') and outputs.scores:
            # outputs.scores contains logits for each generated token
            for i, score in enumerate(outputs.scores[:100]):  # First 100 tokens
                entropy = self.calculate_entropy(score[0])  # [0] because batch_size=1
                entropies.append(entropy)
        
        return generated_text, entropies

    def extract_answer(self, text: str) -> str:
        """Extract answer using the same method as baseline"""
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
        """Format prompt the same way as baseline"""
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

    def evaluate_dataset(self, problems: List[Dict], max_samples: int = 100) -> List[Dict]:
        """Evaluate model with entropy tracking"""
        results = []
        
        problems_subset = problems[:max_samples]
        
        for i, problem in enumerate(tqdm(problems_subset, desc="Processing problems")):
            question = problem['question']
            expected_answer = problem['answer'].split('#### ')[-1].strip()
            
            prompt = self.format_prompt(question)
            
            try:
                # Generate with entropy tracking
                generated_text, entropies = self.generate_with_entropy_tracking(prompt)
                
                # Extract predicted answer
                predicted_answer = self.extract_answer(generated_text)
                
                # Check correctness
                is_correct = self.check_answer_correctness(predicted_answer, expected_answer)
                
                # Calculate entropy statistics for first 100 tokens
                if entropies:
                    avg_entropy = np.mean(entropies)
                    max_entropy = np.max(entropies)
                    min_entropy = np.min(entropies)
                    std_entropy = np.std(entropies)
                else:
                    avg_entropy = max_entropy = min_entropy = std_entropy = 0.0
                
                result = {
                    'problem_id': i,
                    'question': question,
                    'expected_answer': expected_answer,
                    'generated_text': generated_text,
                    'predicted_answer': predicted_answer,
                    'is_correct': is_correct,
                    'entropies': entropies,
                    'avg_entropy': avg_entropy,
                    'max_entropy': max_entropy,
                    'min_entropy': min_entropy,
                    'std_entropy': std_entropy,
                    'num_entropy_tokens': len(entropies)
                }
                
                results.append(result)
                
                # Progress update every 10 problems
                if (i + 1) % 10 == 0:
                    accuracy = sum(1 for r in results if r['is_correct']) / len(results)
                    logger.info(f"Processed {i+1}/{max_samples} problems. Current accuracy: {accuracy:.1%}")
                
            except Exception as e:
                logger.error(f"Error processing problem {i}: {e}")
                continue
                
            # Memory cleanup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        return results

    def analyze_results(self, results: List[Dict]) -> Dict:
        """Analyze correlation between entropy and correctness"""
        if not results:
            return {}
        
        df = pd.DataFrame(results)
        
        correct_results = df[df['is_correct'] == True]
        incorrect_results = df[df['is_correct'] == False]
        
        analysis = {
            'total_problems': len(df),
            'correct_count': len(correct_results),
            'incorrect_count': len(incorrect_results),
            'accuracy': len(correct_results) / len(df) if len(df) > 0 else 0.0,
        }
        
        if len(correct_results) > 0 and len(incorrect_results) > 0:
            analysis.update({
                'correct_avg_entropy': correct_results['avg_entropy'].mean(),
                'incorrect_avg_entropy': incorrect_results['avg_entropy'].mean(),
                'correct_entropy_std': correct_results['avg_entropy'].std(),
                'incorrect_entropy_std': incorrect_results['avg_entropy'].std(),
                'entropy_difference': incorrect_results['avg_entropy'].mean() - correct_results['avg_entropy'].mean()
            })
            
            # Additional entropy statistics
            analysis.update({
                'correct_max_entropy': correct_results['max_entropy'].mean(),
                'incorrect_max_entropy': incorrect_results['max_entropy'].mean(),
                'correct_min_entropy': correct_results['min_entropy'].mean(),
                'incorrect_min_entropy': incorrect_results['min_entropy'].mean(),
            })
            
            if len(df) > 10:
                # Correlation with average entropy
                pearson_corr, pearson_p = pearsonr(df['avg_entropy'], df['is_correct'].astype(int))
                spearman_corr, spearman_p = spearmanr(df['avg_entropy'], df['is_correct'].astype(int))
                
                # Correlation with max entropy
                pearson_max_corr, pearson_max_p = pearsonr(df['max_entropy'], df['is_correct'].astype(int))
                
                # Correlation with entropy std
                pearson_std_corr, pearson_std_p = pearsonr(df['std_entropy'], df['is_correct'].astype(int))
                
                analysis.update({
                    'pearson_correlation_avg': pearson_corr,
                    'pearson_p_value_avg': pearson_p,
                    'spearman_correlation_avg': spearman_corr,
                    'spearman_p_value_avg': spearman_p,
                    'pearson_correlation_max': pearson_max_corr,
                    'pearson_p_value_max': pearson_max_p,
                    'pearson_correlation_std': pearson_std_corr,
                    'pearson_p_value_std': pearson_std_p,
                })
        
        return analysis

    def plot_results(self, results: List[Dict], output_dir: str = "."):
        """Create comprehensive visualizations"""
        if not results:
            return
        
        df = pd.DataFrame(results)
        
        plt.figure(figsize=(20, 12))
        
        correct_entropies = df[df['is_correct'] == True]['avg_entropy']
        incorrect_entropies = df[df['is_correct'] == False]['avg_entropy']
        
        # Plot 1: Entropy distribution by correctness
        plt.subplot(2, 4, 1)
        if len(correct_entropies) > 0:
            plt.hist(correct_entropies, alpha=0.7, label=f'Correct ({len(correct_entropies)})', bins=15, color='green')
        if len(incorrect_entropies) > 0:
            plt.hist(incorrect_entropies, alpha=0.7, label=f'Incorrect ({len(incorrect_entropies)})', bins=15, color='red')
        plt.xlabel('Average Entropy')
        plt.ylabel('Frequency')
        plt.title('Entropy Distribution by Correctness')
        plt.legend()
        
        # Plot 2: Scatter plot
        plt.subplot(2, 4, 2)
        colors = ['green' if correct else 'red' for correct in df['is_correct']]
        plt.scatter(df['avg_entropy'], df['is_correct'].astype(int), c=colors, alpha=0.6)
        plt.xlabel('Average Entropy')
        plt.ylabel('Correctness (0/1)')
        plt.title('Entropy vs Correctness')
        
        # Plot 3: Box plot
        plt.subplot(2, 4, 3)
        data_to_plot = []
        labels = []
        if len(correct_entropies) > 0:
            data_to_plot.append(correct_entropies)
            labels.append('Correct')
        if len(incorrect_entropies) > 0:
            data_to_plot.append(incorrect_entropies)
            labels.append('Incorrect')
        
        if data_to_plot:
            plt.boxplot(data_to_plot, labels=labels)
            plt.ylabel('Average Entropy')
            plt.title('Entropy Distribution Box Plot')
        
        # Plot 4: Token-level entropy evolution
        plt.subplot(2, 4, 4)
        if 'entropies' in df.columns:
            max_tokens = min(50, max(len(e) for e in df['entropies'] if e) if len(df) > 0 else 0)
            if max_tokens > 0:
                correct_evolution = []
                incorrect_evolution = []
                
                for token_idx in range(max_tokens):
                    if len(correct_entropies) > 0:
                        correct_at_token = [e[token_idx] for e in df[df['is_correct'] == True]['entropies'] 
                                          if len(e) > token_idx]
                        if correct_at_token:
                            correct_evolution.append(np.mean(correct_at_token))
                    
                    if len(incorrect_entropies) > 0:
                        incorrect_at_token = [e[token_idx] for e in df[df['is_correct'] == False]['entropies'] 
                                            if len(e) > token_idx]
                        if incorrect_at_token:
                            incorrect_evolution.append(np.mean(incorrect_at_token))
                
                if correct_evolution:
                    plt.plot(range(len(correct_evolution)), correct_evolution, 'g-', label='Correct', alpha=0.8)
                if incorrect_evolution:
                    plt.plot(range(len(incorrect_evolution)), incorrect_evolution, 'r-', label='Incorrect', alpha=0.8)
                plt.xlabel('Token Position')
                plt.ylabel('Average Entropy')
                plt.title('Entropy Evolution (First 50 Tokens)')
                plt.legend()
        
        # Plot 5: Overall accuracy
        plt.subplot(2, 4, 5)
        accuracy = df['is_correct'].mean()
        plt.bar(['Incorrect', 'Correct'], [1-accuracy, accuracy], color=['red', 'green'], alpha=0.7)
        plt.ylabel('Proportion')
        plt.title(f'Overall Accuracy: {accuracy:.1%}')
        plt.ylim(0, 1)
        
        # Plot 6: Entropy quartile analysis
        plt.subplot(2, 4, 6)
        if len(df) > 4:
            df['entropy_quartile'] = pd.qcut(df['avg_entropy'], 4, labels=['Q1\n(Low)', 'Q2', 'Q3', 'Q4\n(High)'])
            quartile_accuracy = df.groupby('entropy_quartile')['is_correct'].mean()
            bars = plt.bar(quartile_accuracy.index, quartile_accuracy.values, 
                          color=['darkgreen', 'lightgreen', 'orange', 'red'], alpha=0.7)
            plt.xlabel('Entropy Quartile')
            plt.ylabel('Accuracy')
            plt.title('Accuracy by Entropy Quartile')
            plt.ylim(0, 1)
            
            # Add value labels on bars
            for bar, value in zip(bars, quartile_accuracy.values):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                        f'{value:.2f}', ha='center', va='bottom')
        
        # Plot 7: Max entropy comparison
        plt.subplot(2, 4, 7)
        if len(df) > 0:
            correct_max = df[df['is_correct'] == True]['max_entropy']
            incorrect_max = df[df['is_correct'] == False]['max_entropy']
            
            data_to_plot = []
            labels = []
            if len(correct_max) > 0:
                data_to_plot.append(correct_max)
                labels.append('Correct')
            if len(incorrect_max) > 0:
                data_to_plot.append(incorrect_max)
                labels.append('Incorrect')
            
            if data_to_plot:
                plt.boxplot(data_to_plot, labels=labels)
                plt.ylabel('Max Entropy')
                plt.title('Max Entropy by Correctness')
        
        # Plot 8: Entropy standard deviation
        plt.subplot(2, 4, 8)
        if len(df) > 0:
            correct_std = df[df['is_correct'] == True]['std_entropy']
            incorrect_std = df[df['is_correct'] == False]['std_entropy']
            
            data_to_plot = []
            labels = []
            if len(correct_std) > 0:
                data_to_plot.append(correct_std)
                labels.append('Correct')
            if len(incorrect_std) > 0:
                data_to_plot.append(incorrect_std)
                labels.append('Incorrect')
            
            if data_to_plot:
                plt.boxplot(data_to_plot, labels=labels)
                plt.ylabel('Entropy Std Dev')
                plt.title('Entropy Variability by Correctness')
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/corrected_entropy_analysis.png", dpi=300, bbox_inches='tight')
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Corrected entropy evaluation preserving model performance")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct", help="Model name")
    parser.add_argument("--max_samples", type=int, default=100, help="Max samples to evaluate")
    parser.add_argument("--output_dir", default="./entropy_results_corrected", help="Output directory")
    
    args = parser.parse_args()
    
    if not torch.cuda.is_available():
        logger.error("No GPUs available!")
        return
    
    # Create output directory
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"{args.output_dir}_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    evaluator = CorrectedEntropyEvaluator(args.model)
    
    # Load GSM8K dataset
    gsm8k_problems = evaluator.load_gsm8k()
    if not gsm8k_problems:
        logger.error("Failed to load GSM8K dataset")
        return
    
    logger.info(f"Starting corrected entropy evaluation on {args.max_samples} GSM8K problems")
    results = evaluator.evaluate_dataset(gsm8k_problems, args.max_samples)
    
    if results:
        logger.info("Analyzing results...")
        analysis = evaluator.analyze_results(results)
        
        print("\n" + "="*80)
        print("CORRECTED ENTROPY vs CORRECTNESS ANALYSIS")
        print("="*80)
        print(f"Model: {args.model}")
        print(f"Total problems evaluated: {analysis.get('total_problems', 0)}")
        print(f"Correct answers: {analysis.get('correct_count', 0)}")
        print(f"Incorrect answers: {analysis.get('incorrect_count', 0)}")
        print(f"Overall accuracy: {analysis.get('accuracy', 0):.1%}")
        
        if analysis.get('correct_count', 0) > 0 and analysis.get('incorrect_count', 0) > 0:
            print(f"\n📊 Entropy Statistics:")
            print(f"  Correct answers:")
            print(f"    - Average entropy: {analysis['correct_avg_entropy']:.3f} ± {analysis['correct_entropy_std']:.3f}")
            print(f"    - Max entropy: {analysis['correct_max_entropy']:.3f}")
            print(f"    - Min entropy: {analysis['correct_min_entropy']:.3f}")
            
            print(f"  Incorrect answers:")
            print(f"    - Average entropy: {analysis['incorrect_avg_entropy']:.3f} ± {analysis['incorrect_entropy_std']:.3f}")
            print(f"    - Max entropy: {analysis['incorrect_max_entropy']:.3f}")
            print(f"    - Min entropy: {analysis['incorrect_min_entropy']:.3f}")
            
            print(f"\n  Entropy differences (incorrect - correct):")
            print(f"    - Average: {analysis['entropy_difference']:.3f}")
            print(f"    - Max: {analysis['incorrect_max_entropy'] - analysis['correct_max_entropy']:.3f}")
            
            if 'pearson_correlation_avg' in analysis:
                print(f"\n🔍 Correlation Analysis:")
                print(f"  Average Entropy vs Correctness:")
                print(f"    - Pearson: {analysis['pearson_correlation_avg']:.3f} (p={analysis['pearson_p_value_avg']:.3f})")
                print(f"    - Spearman: {analysis['spearman_correlation_avg']:.3f} (p={analysis['spearman_p_value_avg']:.3f})")
                
                print(f"  Max Entropy vs Correctness:")
                print(f"    - Pearson: {analysis['pearson_correlation_max']:.3f} (p={analysis['pearson_p_value_max']:.3f})")
                
                print(f"  Entropy Std Dev vs Correctness:")
                print(f"    - Pearson: {analysis['pearson_correlation_std']:.3f} (p={analysis['pearson_p_value_std']:.3f})")
                
                # Determine significance
                significant_correlations = []
                if analysis['pearson_p_value_avg'] < 0.05:
                    significant_correlations.append(f"Average entropy (r={analysis['pearson_correlation_avg']:.3f})")
                if analysis['pearson_p_value_max'] < 0.05:
                    significant_correlations.append(f"Max entropy (r={analysis['pearson_correlation_max']:.3f})")
                if analysis['pearson_p_value_std'] < 0.05:
                    significant_correlations.append(f"Entropy variability (r={analysis['pearson_correlation_std']:.3f})")
                
                if significant_correlations:
                    print(f"\n🎯 SIGNIFICANT FINDINGS:")
                    for corr in significant_correlations:
                        print(f"    ✓ {corr}")
                    
                    if analysis['pearson_correlation_avg'] < -0.1 and analysis['pearson_p_value_avg'] < 0.05:
                        print(f"\n💡 INTERPRETATION: Lower entropy (higher confidence) correlates with correct answers!")
                        print(f"   → Entropy IS a meaningful uncertainty measure for mathematical reasoning.")
                    elif analysis['pearson_correlation_avg'] > 0.1 and analysis['pearson_p_value_avg'] < 0.05:
                        print(f"\n💡 INTERPRETATION: Higher entropy correlates with correct answers!")
                        print(f"   → This suggests complex problems require more 'uncertainty' to solve correctly.")
                else:
                    print(f"\n📊 FINDING: No significant correlations found between entropy measures and correctness.")
        
        # Save results
        with open(f"{output_dir}/all_results.json", 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        with open(f"{output_dir}/analysis.json", 'w') as f:
            json.dump(analysis, f, indent=2, default=str)
        
        logger.info("Creating visualizations...")
        evaluator.plot_results(results, output_dir)
        
        print(f"\n📁 Results saved to: {output_dir}/")
        
        # Verify we maintained performance
        if analysis.get('accuracy', 0) > 0.5:
            print(f"✅ SUCCESS: Maintained good model performance ({analysis['accuracy']:.1%})")
        else:
            print(f"⚠️  WARNING: Lower than expected accuracy ({analysis['accuracy']:.1%})")
        
        # Show some examples
        print(f"\n📋 Sample Results:")
        for i in range(min(3, len(results))):
            r = results[i]
            print(f"\nExample {i+1}:")
            print(f"  Q: {r['question'][:60]}...")
            print(f"  Expected: {r['expected_answer']}")
            print(f"  Predicted: {r['predicted_answer']}")
            print(f"  Correct: {r['is_correct']}")
            print(f"  Avg Entropy: {r['avg_entropy']:.3f}")
            print(f"  Entropy tokens: {r['num_entropy_tokens']}")
        
    else:
        logger.error("No results to analyze.")

if __name__ == "__main__":
    main()
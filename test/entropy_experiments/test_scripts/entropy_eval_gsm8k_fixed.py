#!/usr/bin/env python3
"""
Fixed entropy evaluation for GSM8K with proper answer extraction
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

class GSM8KEntropyEvaluator:
    def __init__(self, model_name: str = "Qwen/Qwen2.5-7B-Instruct", device_map: str = "auto"):
        self.model_name = model_name
        self.device_map = device_map
        
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
    def generate_with_entropy_batch(self, prompts: List[str], max_new_tokens: int = 150) -> List[Tuple[str, List[float]]]:
        """Generate text for multiple prompts and track entropy"""
        batch_size = min(len(prompts), 4)
        results = []
        
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i+batch_size]
            batch_results = self._process_batch(batch_prompts, max_new_tokens)
            results.extend(batch_results)
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        return results

    @torch.no_grad()
    def _process_batch(self, prompts: List[str], max_new_tokens: int) -> List[Tuple[str, List[float]]]:
        """Process a single batch"""
        inputs = self.tokenizer(
            prompts, 
            return_tensors="pt", 
            padding=True, 
            truncation=True,
            max_length=1024
        )
        
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        batch_size = inputs['input_ids'].shape[0]
        
        all_entropies = [[] for _ in range(batch_size)]
        all_generated_tokens = [[] for _ in range(batch_size)]
        finished = [False] * batch_size
        
        current_input_ids = inputs['input_ids'].clone()
        current_attention_mask = inputs['attention_mask'].clone()
        
        for step in range(min(max_new_tokens, 100)):
            if all(finished):
                break
                
            outputs = self.model(
                input_ids=current_input_ids,
                attention_mask=current_attention_mask,
                use_cache=True
            )
            
            logits = outputs.logits[:, -1, :]
            
            for b in range(batch_size):
                if not finished[b] and len(all_entropies[b]) < 100:
                    entropy = self.calculate_entropy(logits[b])
                    all_entropies[b].append(entropy)
            
            next_token_ids = torch.argmax(logits, dim=-1, keepdim=True)
            
            current_input_ids = torch.cat([current_input_ids, next_token_ids], dim=-1)
            current_attention_mask = torch.cat([
                current_attention_mask, 
                torch.ones(batch_size, 1, device=device)
            ], dim=-1)
            
            for b in range(batch_size):
                if not finished[b]:
                    token_id = next_token_ids[b].item()
                    all_generated_tokens[b].append(token_id)
                    
                    if token_id == self.tokenizer.eos_token_id:
                        finished[b] = True
        
        results = []
        for b in range(batch_size):
            generated_text = self.tokenizer.decode(all_generated_tokens[b], skip_special_tokens=True)
            results.append((generated_text, all_entropies[b]))
        
        return results

    def extract_gsm8k_answer(self, ground_truth: str) -> str:
        """Extract the final answer from GSM8K format"""
        # GSM8K answers end with #### [number]
        match = re.search(r'#### (.+)', ground_truth.strip())
        if match:
            return match.group(1).strip()
        return ground_truth.strip()

    def extract_model_answer(self, text: str) -> str:
        """Extract numerical answer from model output"""
        # Clean the text
        text = text.strip()
        
        # Look for common answer patterns at the end
        patterns = [
            r'(?:the answer is|answer:|final answer:|therefore)[:\s]*\$?([+-]?\d+(?:,\d{3})*(?:\.\d+)?)',
            r'(?:equals?|=)[:\s]*\$?([+-]?\d+(?:,\d{3})*(?:\.\d+)?)\s*$',
            r'\$?([+-]?\d+(?:,\d{3})*(?:\.\d+)?)\s*(?:dollars?|cents?|is the answer)\s*$',
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text.lower())
            if matches:
                # Return the last match, remove commas
                return matches[-1].replace(',', '')
        
        # Look for numbers near the end of the text
        sentences = text.split('.')
        for sentence in reversed(sentences[-3:]):  # Check last 3 sentences
            numbers = re.findall(r'\$?([+-]?\d+(?:,\d{3})*(?:\.\d+)?)', sentence)
            if numbers:
                return numbers[-1].replace(',', '')
        
        # Fallback: find all numbers and return the last one
        all_numbers = re.findall(r'([+-]?\d+(?:,\d{3})*(?:\.\d+)?)', text)
        if all_numbers:
            return all_numbers[-1].replace(',', '')
        
        return ""

    def check_answer_correctness(self, predicted: str, ground_truth: str) -> bool:
        """Check if predicted answer matches ground truth"""
        try:
            # Remove dollar signs and commas
            pred_clean = predicted.replace('$', '').replace(',', '').strip()
            truth_clean = ground_truth.replace('$', '').replace(',', '').strip()
            
            # Try exact string match first
            if pred_clean.lower() == truth_clean.lower():
                return True
            
            # Try numerical comparison
            pred_num = float(pred_clean)
            truth_num = float(truth_clean)
            
            # Allow small floating point errors
            return abs(pred_num - truth_num) < 1e-6
            
        except (ValueError, TypeError):
            # Fallback to string comparison
            return predicted.strip().lower() == ground_truth.strip().lower()

    def format_gsm8k_prompt(self, question: str) -> str:
        """Format GSM8K problem with chain-of-thought prompting"""
        return f"""Solve this math problem step by step. Show your work and give the final numerical answer.

Question: {question}

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
        """Evaluate model on GSM8K"""
        results = []
        batch_size = 4  # Smaller batch for longer sequences
        
        problems_subset = problems[:max_samples]
        
        for i in range(0, len(problems_subset), batch_size):
            batch_problems = problems_subset[i:i+batch_size]
            
            prompts = []
            batch_data = []
            
            for problem in batch_problems:
                question = problem['question']
                answer_raw = problem['answer']
                answer_clean = self.extract_gsm8k_answer(answer_raw)
                
                prompt = self.format_gsm8k_prompt(question)
                prompts.append(prompt)
                batch_data.append({
                    'question': question,
                    'answer_raw': answer_raw,
                    'answer_clean': answer_clean,
                    'problem_id': len(results) + len(batch_data)
                })
            
            logger.info(f"Processing batch {i//batch_size + 1} ({len(prompts)} problems)")
            batch_results = self.generate_with_entropy_batch(prompts, max_new_tokens=200)
            
            for (generated_text, entropies), data in zip(batch_results, batch_data):
                predicted_answer = self.extract_model_answer(generated_text)
                is_correct = self.check_answer_correctness(predicted_answer, data['answer_clean'])
                
                entropies_100 = entropies[:100]
                avg_entropy = np.mean(entropies_100) if entropies_100 else 0.0
                max_entropy = np.max(entropies_100) if entropies_100 else 0.0
                min_entropy = np.min(entropies_100) if entropies_100 else 0.0
                
                result = {
                    'dataset': 'GSM8K',
                    'problem_id': data['problem_id'],
                    'problem': data['question'],
                    'ground_truth_raw': data['answer_raw'],
                    'ground_truth': data['answer_clean'],
                    'predicted_answer': predicted_answer,
                    'generated_text': generated_text,
                    'is_correct': is_correct,
                    'entropies': entropies_100,
                    'avg_entropy': avg_entropy,
                    'max_entropy': max_entropy,
                    'min_entropy': min_entropy,
                    'num_tokens': len(entropies)
                }
                
                results.append(result)
            
            accuracy = sum(1 for r in results if r['is_correct']) / len(results) if results else 0
            logger.info(f"Processed {len(results)}/{len(problems_subset)} problems. Current accuracy: {accuracy:.1%}")
            
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
            
            if len(df) > 10:
                pearson_corr, pearson_p = pearsonr(df['avg_entropy'], df['is_correct'].astype(int))
                spearman_corr, spearman_p = spearmanr(df['avg_entropy'], df['is_correct'].astype(int))
                
                analysis.update({
                    'pearson_correlation': pearson_corr,
                    'pearson_p_value': pearson_p,
                    'spearman_correlation': spearman_corr,
                    'spearman_p_value': spearman_p
                })
        
        return analysis

    def plot_results(self, results: List[Dict], output_dir: str = "."):
        """Create visualizations"""
        if not results:
            return
        
        df = pd.DataFrame(results)
        
        plt.figure(figsize=(16, 10))
        
        # Plot 1: Entropy distribution
        plt.subplot(2, 3, 1)
        correct_entropies = df[df['is_correct'] == True]['avg_entropy']
        incorrect_entropies = df[df['is_correct'] == False]['avg_entropy']
        
        if len(correct_entropies) > 0:
            plt.hist(correct_entropies, alpha=0.7, label=f'Correct ({len(correct_entropies)})', bins=15, color='green')
        if len(incorrect_entropies) > 0:
            plt.hist(incorrect_entropies, alpha=0.7, label=f'Incorrect ({len(incorrect_entropies)})', bins=15, color='red')
        plt.xlabel('Average Entropy')
        plt.ylabel('Frequency')
        plt.title('Entropy Distribution by Correctness')
        plt.legend()
        
        # Plot 2: Scatter plot
        plt.subplot(2, 3, 2)
        colors = ['green' if correct else 'red' for correct in df['is_correct']]
        plt.scatter(df['avg_entropy'], df['is_correct'].astype(int), c=colors, alpha=0.6)
        plt.xlabel('Average Entropy')
        plt.ylabel('Correctness (0/1)')
        plt.title('Entropy vs Correctness')
        
        # Plot 3: Box plot
        plt.subplot(2, 3, 3)
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
            plt.title('Entropy Distribution')
        
        # Plot 4: Token evolution
        plt.subplot(2, 3, 4)
        max_tokens = min(50, max(len(e) for e in df['entropies'] if e) if len(df) > 0 else 0)
        if max_tokens > 0:
            correct_evolution = []
            incorrect_evolution = []
            
            for token_idx in range(max_tokens):
                if len(correct_entropies) > 0:
                    correct_at_token = [e[token_idx] for e in df[df['is_correct'] == True]['entropies'] if len(e) > token_idx]
                    if correct_at_token:
                        correct_evolution.append(np.mean(correct_at_token))
                
                if len(incorrect_entropies) > 0:
                    incorrect_at_token = [e[token_idx] for e in df[df['is_correct'] == False]['entropies'] if len(e) > token_idx]
                    if incorrect_at_token:
                        incorrect_evolution.append(np.mean(incorrect_at_token))
            
            if correct_evolution:
                plt.plot(range(len(correct_evolution)), correct_evolution, 'g-', label='Correct', alpha=0.8)
            if incorrect_evolution:
                plt.plot(range(len(incorrect_evolution)), incorrect_evolution, 'r-', label='Incorrect', alpha=0.8)
            plt.xlabel('Token Position')
            plt.ylabel('Average Entropy')
            plt.title('Entropy Evolution')
            plt.legend()
        
        # Plot 5: Accuracy histogram
        plt.subplot(2, 3, 5)
        accuracy = df['is_correct'].mean()
        plt.bar(['Incorrect', 'Correct'], [1-accuracy, accuracy], color=['red', 'green'], alpha=0.7)
        plt.ylabel('Proportion')
        plt.title(f'Overall Accuracy: {accuracy:.1%}')
        
        # Plot 6: Entropy quartile analysis
        plt.subplot(2, 3, 6)
        if len(df) > 4:
            df['entropy_quartile'] = pd.qcut(df['avg_entropy'], 4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
            quartile_accuracy = df.groupby('entropy_quartile')['is_correct'].mean()
            quartile_accuracy.plot(kind='bar', color=['skyblue', 'lightgreen', 'orange', 'red'])
            plt.xlabel('Entropy Quartile')
            plt.ylabel('Accuracy')
            plt.title('Accuracy by Entropy Quartile')
            plt.xticks(rotation=0)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/gsm8k_entropy_analysis.png", dpi=300, bbox_inches='tight')
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="GSM8K entropy evaluation with fixed answer extraction")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct", help="Model name")
    parser.add_argument("--max_samples", type=int, default=200, help="Max samples to evaluate")
    parser.add_argument("--output_dir", default="./entropy_results_gsm8k_fixed", help="Output directory")
    
    args = parser.parse_args()
    
    if not torch.cuda.is_available():
        logger.error("No GPUs available!")
        return
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    evaluator = GSM8KEntropyEvaluator(args.model)
    
    # Load GSM8K dataset
    gsm8k_problems = evaluator.load_gsm8k()
    if not gsm8k_problems:
        logger.error("Failed to load GSM8K dataset")
        return
    
    logger.info(f"Evaluating GSM8K ({len(gsm8k_problems)} problems available)")
    results = evaluator.evaluate_dataset(gsm8k_problems, args.max_samples)
    
    if results:
        logger.info("Analyzing results...")
        analysis = evaluator.analyze_results(results)
        
        print("\n" + "="*80)
        print("GSM8K ENTROPY vs CORRECTNESS ANALYSIS")
        print("="*80)
        print(f"Model: {args.model}")
        print(f"Total problems evaluated: {analysis.get('total_problems', 0)}")
        print(f"Correct answers: {analysis.get('correct_count', 0)}")
        print(f"Incorrect answers: {analysis.get('incorrect_count', 0)}")
        print(f"Overall accuracy: {analysis.get('accuracy', 0):.1%}")
        
        if analysis.get('correct_count', 0) > 0 and analysis.get('incorrect_count', 0) > 0:
            print(f"\nEntropy Statistics:")
            print(f"  Correct answers   - avg entropy: {analysis['correct_avg_entropy']:.3f} ± {analysis['correct_entropy_std']:.3f}")
            print(f"  Incorrect answers - avg entropy: {analysis['incorrect_avg_entropy']:.3f} ± {analysis['incorrect_entropy_std']:.3f}")
            print(f"  Entropy difference (incorrect - correct): {analysis['entropy_difference']:.3f}")
            
            if 'pearson_correlation' in analysis:
                print(f"\nCorrelation Analysis:")
                print(f"  Pearson correlation:  {analysis['pearson_correlation']:.3f} (p={analysis['pearson_p_value']:.3f})")
                print(f"  Spearman correlation: {analysis['spearman_correlation']:.3f} (p={analysis['spearman_p_value']:.3f})")
                
                if analysis['pearson_p_value'] < 0.05:
                    if analysis['pearson_correlation'] < 0:
                        print(f"\n🔍 SIGNIFICANT FINDING: Lower entropy (higher confidence) correlates with correctness!")
                        print("   → This suggests entropy is a useful uncertainty measure for math problems.")
                    else:
                        print(f"\n🔍 SIGNIFICANT FINDING: Higher entropy correlates with correctness!")
                else:
                    print(f"\n📊 FINDING: No significant correlation between entropy and correctness.")
        else:
            print(f"\n⚠️  Need both correct and incorrect answers to analyze correlation.")
        
        # Save results
        with open(f"{args.output_dir}/all_results.json", 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        with open(f"{args.output_dir}/analysis.json", 'w') as f:
            json.dump(analysis, f, indent=2, default=str)
        
        logger.info("Creating visualizations...")
        evaluator.plot_results(results, args.output_dir)
        
        print(f"\n📁 Results saved to: {args.output_dir}/")
        
        # Show some examples
        print(f"\n📋 Sample Results:")
        for i in range(min(3, len(results))):
            r = results[i]
            print(f"\nExample {i+1}:")
            print(f"  Q: {r['problem'][:80]}...")
            print(f"  Expected: {r['ground_truth']}")
            print(f"  Predicted: {r['predicted_answer']}")
            print(f"  Correct: {r['is_correct']}")
            print(f"  Entropy: {r['avg_entropy']:.3f}")
        
    else:
        logger.error("No results to analyze.")

if __name__ == "__main__":
    main()
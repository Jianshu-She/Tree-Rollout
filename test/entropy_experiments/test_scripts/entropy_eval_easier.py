#!/usr/bin/env python3
"""
Entropy evaluation with easier math datasets for better accuracy
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

class EasierEntropyEvaluator:
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
    def generate_with_entropy_batch(self, prompts: List[str], max_new_tokens: int = 100) -> List[Tuple[str, List[float]]]:
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

    def load_gsm8k(self) -> List[Dict]:
        """Load GSM8K dataset (easier math problems)"""
        try:
            logger.info("Loading GSM8K dataset...")
            dataset = load_dataset("gsm8k", "main", split="test")
            problems = list(dataset)
            logger.info(f"Loaded {len(problems)} problems from GSM8K")
            return problems
        except Exception as e:
            logger.error(f"Failed to load GSM8K: {e}")
            return []

    def load_math_qa(self) -> List[Dict]:
        """Load MathQA dataset"""
        try:
            logger.info("Loading simple math problems...")
            # Create simple arithmetic problems
            problems = []
            
            # Addition problems
            for i in range(50):
                a, b = np.random.randint(1, 100, 2)
                problems.append({
                    'question': f"What is {a} + {b}?",
                    'answer': str(a + b),
                    'type': 'addition'
                })
            
            # Subtraction problems  
            for i in range(30):
                a = np.random.randint(50, 200)
                b = np.random.randint(1, a)
                problems.append({
                    'question': f"What is {a} - {b}?",
                    'answer': str(a - b),
                    'type': 'subtraction'
                })
            
            # Multiplication problems
            for i in range(30):
                a = np.random.randint(2, 20)
                b = np.random.randint(2, 20)
                problems.append({
                    'question': f"What is {a} × {b}?",
                    'answer': str(a * b),
                    'type': 'multiplication'
                })
            
            # Division problems
            for i in range(20):
                b = np.random.randint(2, 12)
                a = b * np.random.randint(2, 15)
                problems.append({
                    'question': f"What is {a} ÷ {b}?",
                    'answer': str(a // b),
                    'type': 'division'
                })
            
            logger.info(f"Generated {len(problems)} simple math problems")
            return problems
            
        except Exception as e:
            logger.error(f"Failed to generate math problems: {e}")
            return []

    def extract_answer(self, text: str) -> str:
        """Extract numerical answer"""
        # Look for numbers in the text
        numbers = re.findall(r'-?\d+(?:\.\d+)?', text)
        if numbers:
            return numbers[-1]  # Return last number found
        
        # Look for common answer patterns
        patterns = [
            r"(?:the answer is|answer:|equals?|=)\s*([+-]?\d+(?:\.\d+)?)",
            r"([+-]?\d+(?:\.\d+)?)\s*(?:is the answer|is correct)",
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text.lower())
            if match:
                return match.group(1)
        
        return ""

    def check_answer_correctness(self, predicted: str, ground_truth: str) -> bool:
        """Check if predicted answer matches ground truth"""
        try:
            pred_num = float(predicted.strip())
            truth_num = float(ground_truth.strip())
            return abs(pred_num - truth_num) < 1e-6
        except:
            return predicted.strip().lower() == ground_truth.strip().lower()

    def format_math_prompt(self, question: str) -> str:
        """Format as simple math prompt"""
        return f"Solve this math problem step by step:\n\nQuestion: {question}\n\nAnswer:"

    def evaluate_dataset(self, dataset_name: str, problems: List[Dict], max_samples: int = 100) -> List[Dict]:
        """Evaluate model on dataset"""
        results = []
        batch_size = 8
        
        problems_subset = problems[:max_samples]
        
        for i in range(0, len(problems_subset), batch_size):
            batch_problems = problems_subset[i:i+batch_size]
            
            prompts = []
            batch_data = []
            
            for problem in batch_problems:
                if 'question' in problem:
                    question = problem['question']
                    answer = problem['answer']
                elif 'problem' in problem:
                    question = problem['problem']
                    answer = problem.get('answer', problem.get('solution', ''))
                else:
                    continue
                
                prompt = self.format_math_prompt(question)
                prompts.append(prompt)
                batch_data.append({
                    'question': question,
                    'answer': answer,
                    'problem_id': len(results) + len(batch_data),
                    'type': problem.get('type', 'unknown')
                })
            
            if not prompts:
                continue
            
            logger.info(f"Processing batch {i//batch_size + 1} ({len(prompts)} problems)")
            batch_results = self.generate_with_entropy_batch(prompts, max_new_tokens=50)
            
            for (generated_text, entropies), data in zip(batch_results, batch_data):
                predicted_answer = self.extract_answer(generated_text)
                is_correct = self.check_answer_correctness(predicted_answer, data['answer'])
                
                entropies_100 = entropies[:100]
                avg_entropy = np.mean(entropies_100) if entropies_100 else 0.0
                max_entropy = np.max(entropies_100) if entropies_100 else 0.0
                min_entropy = np.min(entropies_100) if entropies_100 else 0.0
                
                result = {
                    'dataset': dataset_name,
                    'problem_id': data['problem_id'],
                    'problem_type': data['type'],
                    'problem': data['question'],
                    'ground_truth': data['answer'],
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
            logger.info(f"Processed {len(results)}/{len(problems_subset)} problems. Accuracy: {accuracy:.1%}")
            
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
        
        # Add problem type breakdown
        if 'problem_type' in df.columns:
            type_accuracy = df.groupby('problem_type')['is_correct'].agg(['count', 'sum', 'mean']).round(3)
            analysis['accuracy_by_type'] = type_accuracy.to_dict()
        
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
        
        plt.figure(figsize=(18, 12))
        
        # Plot 1: Entropy distribution
        plt.subplot(2, 4, 1)
        correct_entropies = df[df['is_correct'] == True]['avg_entropy']
        incorrect_entropies = df[df['is_correct'] == False]['avg_entropy']
        
        plt.hist(correct_entropies, alpha=0.7, label='Correct', bins=20, color='green')
        plt.hist(incorrect_entropies, alpha=0.7, label='Incorrect', bins=20, color='red')
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
        data_to_plot = [correct_entropies, incorrect_entropies]
        plt.boxplot(data_to_plot, labels=['Correct', 'Incorrect'])
        plt.ylabel('Average Entropy')
        plt.title('Entropy Distribution')
        
        # Plot 4: Accuracy by problem type
        plt.subplot(2, 4, 4)
        if 'problem_type' in df.columns:
            type_accuracy = df.groupby('problem_type')['is_correct'].mean()
            type_accuracy.plot(kind='bar', color='skyblue')
            plt.xlabel('Problem Type')
            plt.ylabel('Accuracy')
            plt.title('Accuracy by Problem Type')
            plt.xticks(rotation=45)
        
        # Plot 5: Token evolution
        plt.subplot(2, 4, 5)
        max_tokens = min(30, max(len(e) for e in df['entropies'] if e))
        correct_evolution = []
        incorrect_evolution = []
        
        for token_idx in range(max_tokens):
            correct_at_token = [e[token_idx] for e in df[df['is_correct'] == True]['entropies'] if len(e) > token_idx]
            incorrect_at_token = [e[token_idx] for e in df[df['is_correct'] == False]['entropies'] if len(e) > token_idx]
            
            if correct_at_token:
                correct_evolution.append(np.mean(correct_at_token))
            if incorrect_at_token:
                incorrect_evolution.append(np.mean(incorrect_at_token))
        
        plt.plot(range(len(correct_evolution)), correct_evolution, 'g-', label='Correct', alpha=0.8)
        plt.plot(range(len(incorrect_evolution)), incorrect_evolution, 'r-', label='Incorrect', alpha=0.8)
        plt.xlabel('Token Position')
        plt.ylabel('Average Entropy')
        plt.title('Entropy Evolution')
        plt.legend()
        
        # Plot 6: Accuracy vs Entropy quartiles
        plt.subplot(2, 4, 6)
        df['entropy_quartile'] = pd.qcut(df['avg_entropy'], 4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
        quartile_accuracy = df.groupby('entropy_quartile')['is_correct'].mean()
        quartile_accuracy.plot(kind='bar', color=['skyblue', 'lightgreen', 'orange', 'red'])
        plt.xlabel('Entropy Quartile')
        plt.ylabel('Accuracy')
        plt.title('Accuracy by Entropy Quartile')
        plt.xticks(rotation=0)
        
        # Plot 7: Problem difficulty vs entropy
        plt.subplot(2, 4, 7)
        if 'problem_type' in df.columns:
            type_entropy = df.groupby('problem_type')['avg_entropy'].mean()
            type_entropy.plot(kind='bar', color='orange')
            plt.xlabel('Problem Type')
            plt.ylabel('Average Entropy')
            plt.title('Average Entropy by Problem Type')
            plt.xticks(rotation=45)
        
        # Plot 8: Confusion matrix style
        plt.subplot(2, 4, 8)
        # Bin by entropy and show accuracy
        df['entropy_bin'] = pd.cut(df['avg_entropy'], bins=5)
        bin_stats = df.groupby('entropy_bin')['is_correct'].agg(['count', 'mean']).fillna(0)
        bin_stats['mean'].plot(kind='line', marker='o', color='purple')
        plt.xlabel('Entropy Range')
        plt.ylabel('Accuracy')
        plt.title('Accuracy vs Entropy Range')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/entropy_analysis_easier.png", dpi=300, bbox_inches='tight')
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Easier math entropy evaluation")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct", help="Model name")
    parser.add_argument("--max_samples", type=int, default=200, help="Max samples per dataset")
    parser.add_argument("--output_dir", default="./entropy_results_easier", help="Output directory")
    parser.add_argument("--use_gsm8k", action="store_true", help="Use GSM8K instead of simple math")
    
    args = parser.parse_args()
    
    if not torch.cuda.is_available():
        logger.error("No GPUs available!")
        return
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    evaluator = EasierEntropyEvaluator(args.model)
    
    all_results = []
    
    if args.use_gsm8k:
        # Use GSM8K dataset
        gsm8k_problems = evaluator.load_gsm8k()
        if gsm8k_problems:
            logger.info(f"Evaluating GSM8K ({len(gsm8k_problems)} problems available)")
            gsm8k_results = evaluator.evaluate_dataset("GSM8K", gsm8k_problems, args.max_samples)
            all_results.extend(gsm8k_results)
    else:
        # Use simple math problems
        simple_problems = evaluator.load_math_qa()
        if simple_problems:
            logger.info(f"Evaluating Simple Math ({len(simple_problems)} problems)")
            simple_results = evaluator.evaluate_dataset("Simple-Math", simple_problems, args.max_samples)
            all_results.extend(simple_results)
    
    if all_results:
        logger.info("Analyzing results...")
        analysis = evaluator.analyze_results(all_results)
        
        print("\n" + "="*80)
        print("ENTROPY vs CORRECTNESS ANALYSIS (Easier Math)")
        print("="*80)
        print(f"Model: {args.model}")
        print(f"Total problems evaluated: {analysis.get('total_problems', 0)}")
        print(f"Correct answers: {analysis.get('correct_count', 0)}")
        print(f"Incorrect answers: {analysis.get('incorrect_count', 0)}")
        print(f"Overall accuracy: {analysis.get('accuracy', 0):.1%}")
        
        if 'accuracy_by_type' in analysis:
            print(f"\nAccuracy by Problem Type:")
            for ptype, stats in analysis['accuracy_by_type']['mean'].items():
                count = analysis['accuracy_by_type']['count'][ptype]
                print(f"  {ptype}: {stats:.1%} ({count} problems)")
        
        if 'correct_avg_entropy' in analysis:
            print(f"\nEntropy Statistics:")
            print(f"  Correct answers   - avg entropy: {analysis['correct_avg_entropy']:.3f} ± {analysis['correct_entropy_std']:.3f}")
            print(f"  Incorrect answers - avg entropy: {analysis['incorrect_avg_entropy']:.3f} ± {analysis['incorrect_entropy_std']:.3f}")
            print(f"  Entropy difference (incorrect - correct): {analysis['entropy_difference']:.3f}")
            
            if 'pearson_correlation' in analysis:
                print(f"\nCorrelation Analysis:")
                print(f"  Pearson correlation:  {analysis['pearson_correlation']:.3f} (p={analysis['pearson_p_value']:.3f})")
                print(f"  Spearman correlation: {analysis['spearman_correlation']:.3f} (p={analysis['spearman_p_value']:.3f})")
                
                if analysis['pearson_p_value'] < 0.05:
                    direction = 'higher' if analysis['pearson_correlation'] > 0 else 'lower'
                    print(f"\n🔍 SIGNIFICANT FINDING: {direction.title()} entropy correlates with correctness!")
                    if analysis['pearson_correlation'] < 0:
                        print("   → Lower entropy (higher confidence) → Better accuracy")
                    else:
                        print("   → Higher entropy (lower confidence) → Better accuracy")
                else:
                    print(f"\n📊 FINDING: No significant correlation between entropy and correctness.")
        
        # Save results
        with open(f"{args.output_dir}/all_results.json", 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
        
        with open(f"{args.output_dir}/analysis.json", 'w') as f:
            json.dump(analysis, f, indent=2, default=str)
        
        logger.info("Creating visualizations...")
        evaluator.plot_results(all_results, args.output_dir)
        
        print(f"\n📁 Results saved to: {args.output_dir}/")
        
    else:
        logger.error("No results to analyze.")

if __name__ == "__main__":
    main()
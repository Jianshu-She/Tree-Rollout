#!/usr/bin/env python3
"""
GPU-optimized entropy evaluation script for Qwen2.5-7B-Instruct
Designed for multiple H200 GPUs with memory optimization
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
import gc
warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GPUEntropyEvaluator:
    def __init__(self, model_name: str = "Qwen/Qwen2.5-7B-Instruct", device_map: str = "auto"):
        self.model_name = model_name
        self.device_map = device_map
        
        # GPU memory optimization
        torch.cuda.empty_cache()
        gc.collect()
        
        logger.info(f"Loading model {model_name} with device_map={device_map}")
        logger.info(f"Available GPUs: {torch.cuda.device_count()}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, 
            trust_remote_code=True,
            padding_side="left"
        )
        
        # Load model with optimizations for H200
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,  # H200 optimized
            device_map=device_map,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            max_memory={i: "22GiB" for i in range(torch.cuda.device_count())},  # H200 has 32GB, leave some headroom
            load_in_8bit=False,  # H200 has enough memory
            load_in_4bit=False,
            # attn_implementation="flash_attention_2" if torch.cuda.is_available() else None
        )
        
        self.model.eval()
        
        # Add pad token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # Print memory usage
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                memory_used = torch.cuda.memory_allocated(i) / 1024**3
                memory_total = torch.cuda.get_device_properties(i).total_memory / 1024**3
                logger.info(f"GPU {i}: {memory_used:.1f}GB / {memory_total:.1f}GB used")

    def calculate_entropy(self, logits: torch.Tensor) -> float:
        """Calculate entropy from logits"""
        with torch.no_grad():
            probs = torch.softmax(logits, dim=-1)
            entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)
            return entropy.cpu().item()

    @torch.no_grad()
    def generate_with_entropy_batch(self, prompts: List[str], max_new_tokens: int = 100) -> List[Tuple[str, List[float]]]:
        """Generate text for multiple prompts and track entropy (batched for efficiency)"""
        batch_size = min(len(prompts), 4)  # Process in smaller batches to avoid OOM
        results = []
        
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i+batch_size]
            batch_results = self._process_batch(batch_prompts, max_new_tokens)
            results.extend(batch_results)
            
            # Clear cache between batches
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        return results

    @torch.no_grad()
    def _process_batch(self, prompts: List[str], max_new_tokens: int) -> List[Tuple[str, List[float]]]:
        """Process a single batch"""
        # Tokenize batch
        inputs = self.tokenizer(
            prompts, 
            return_tensors="pt", 
            padding=True, 
            truncation=True,
            max_length=2048
        )
        
        # Move to appropriate device
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        batch_size = inputs['input_ids'].shape[0]
        input_length = inputs['input_ids'].shape[1]
        
        # Storage for results
        all_entropies = [[] for _ in range(batch_size)]
        all_generated_tokens = [[] for _ in range(batch_size)]
        finished = [False] * batch_size
        
        current_input_ids = inputs['input_ids'].clone()
        current_attention_mask = inputs['attention_mask'].clone()
        
        for step in range(min(max_new_tokens, 100)):
            if all(finished):
                break
                
            # Forward pass
            outputs = self.model(
                input_ids=current_input_ids,
                attention_mask=current_attention_mask,
                use_cache=True
            )
            
            # Get logits for last position
            logits = outputs.logits[:, -1, :]  # [batch_size, vocab_size]
            
            # Calculate entropy for each sample in batch
            for b in range(batch_size):
                if not finished[b] and len(all_entropies[b]) < 100:
                    entropy = self.calculate_entropy(logits[b])
                    all_entropies[b].append(entropy)
            
            # Generate next tokens (greedy decoding for consistency)
            next_token_ids = torch.argmax(logits, dim=-1, keepdim=True)
            
            # Update sequences
            current_input_ids = torch.cat([current_input_ids, next_token_ids], dim=-1)
            current_attention_mask = torch.cat([
                current_attention_mask, 
                torch.ones(batch_size, 1, device=device)
            ], dim=-1)
            
            # Store generated tokens and check for EOS
            for b in range(batch_size):
                if not finished[b]:
                    token_id = next_token_ids[b].item()
                    all_generated_tokens[b].append(token_id)
                    
                    if token_id == self.tokenizer.eos_token_id:
                        finished[b] = True
        
        # Decode generated texts
        results = []
        for b in range(batch_size):
            generated_text = self.tokenizer.decode(all_generated_tokens[b], skip_special_tokens=True)
            results.append((generated_text, all_entropies[b]))
        
        return results

    def extract_answer(self, text: str) -> str:
        """Extract final answer from generated text"""
        patterns = [
            r"The answer is\s*([^.\n]+)",
            r"Answer:\s*([^.\n]+)", 
            r"Final answer:\s*([^.\n]+)",
            r"\\boxed\{([^}]+)\}",
            r"Therefore,?\s*([^.\n]+)",
            r"So,?\s*([^.\n]+)",
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        # Extract last number if no pattern matches
        numbers = re.findall(r'-?\d+(?:\.\d+)?', text)
        if numbers:
            return numbers[-1]
            
        # Return last line as fallback
        lines = [line.strip() for line in text.strip().split('\n') if line.strip()]
        return lines[-1] if lines else text.strip()

    def load_math_500(self) -> List[Dict]:
        """Load MATH-500 dataset"""
        try:
            logger.info("Loading MATH dataset...")
            dataset = load_dataset("hendrycks/competition_math", split="test", trust_remote_code=True)
            # Take first 500 examples
            problems = list(dataset.select(range(min(500, len(dataset)))))
            logger.info(f"Loaded {len(problems)} problems from MATH dataset")
            return problems
        except Exception as e:
            logger.error(f"Failed to load MATH dataset: {e}")
            return []

    def load_polymath_en(self) -> List[Dict]:
        """Load PolyMath-en dataset or alternatives"""
        datasets_to_try = [
            ("math_qa", "train"),
            ("microsoft/PolyMath", "train"),  
            ("AI-MO/aimo-validation-math-level-4", "train"),
            ("gsm8k", "test")  # Fallback to GSM8K
        ]
        
        for dataset_name, split in datasets_to_try:
            try:
                logger.info(f"Trying to load {dataset_name}...")
                dataset = load_dataset(dataset_name, split=split, trust_remote_code=True)
                problems = list(dataset.select(range(min(500, len(dataset)))))
                logger.info(f"Loaded {len(problems)} problems from {dataset_name}")
                return problems
            except Exception as e:
                logger.warning(f"Failed to load {dataset_name}: {e}")
                continue
        
        logger.warning("Could not load any PolyMath-like dataset")
        return []

    def format_math_prompt(self, problem: str) -> str:
        """Format math problem as chat prompt"""
        return f"""Please solve this math problem step by step and provide the final answer.

Problem: {problem}

Solution:"""

    def check_answer_correctness(self, predicted: str, ground_truth: str) -> bool:
        """Check if predicted answer matches ground truth"""
        def normalize_answer(ans):
            ans = str(ans).lower().strip()
            ans = re.sub(r'^(the answer is|answer:|final answer:)\s*', '', ans, flags=re.IGNORECASE)
            ans = re.sub(r'\s*(dollars?|cents?|percent|%|\$)\s*$', '', ans, flags=re.IGNORECASE)
            
            # Extract numbers
            numbers = re.findall(r'-?\d+\.?\d*', ans)
            if numbers:
                try:
                    return float(numbers[-1])
                except:
                    pass
            return ans.strip()
        
        pred_norm = normalize_answer(predicted)
        truth_norm = normalize_answer(ground_truth)
        
        # Direct comparison
        if pred_norm == truth_norm:
            return True
        
        # Numerical comparison
        try:
            return abs(float(pred_norm) - float(truth_norm)) < 1e-6
        except:
            pass
        
        # Substring matching
        return str(pred_norm) in str(truth_norm) or str(truth_norm) in str(pred_norm)

    def evaluate_dataset(self, dataset_name: str, problems: List[Dict], max_samples: int = 100) -> List[Dict]:
        """Evaluate model on dataset with GPU optimization"""
        results = []
        batch_size = 8  # Process in batches
        
        # Prepare problems in batches
        for i in range(0, min(max_samples, len(problems)), batch_size):
            batch_problems = problems[i:i+batch_size]
            
            # Prepare prompts
            prompts = []
            batch_data = []
            
            for problem in batch_problems:
                if 'problem' in problem:
                    problem_text = problem['problem']
                    answer = problem.get('solution', problem.get('answer', ''))
                elif 'question' in problem:
                    problem_text = problem['question']
                    answer = problem.get('answer', '')
                else:
                    continue
                
                prompt = self.format_math_prompt(problem_text)
                prompts.append(prompt)
                batch_data.append({
                    'problem_text': problem_text,
                    'answer': answer,
                    'problem_id': len(results) + len(batch_data)
                })
            
            if not prompts:
                continue
            
            # Generate responses with entropy tracking
            logger.info(f"Processing batch {i//batch_size + 1} ({len(prompts)} problems)")
            batch_results = self.generate_with_entropy_batch(prompts)
            
            # Process results
            for (generated_text, entropies), data in zip(batch_results, batch_data):
                predicted_answer = self.extract_answer(generated_text)
                is_correct = self.check_answer_correctness(predicted_answer, data['answer'])
                
                # Calculate entropy statistics for first 100 tokens
                entropies_100 = entropies[:100]
                avg_entropy = np.mean(entropies_100) if entropies_100 else 0.0
                max_entropy = np.max(entropies_100) if entropies_100 else 0.0
                min_entropy = np.min(entropies_100) if entropies_100 else 0.0
                
                result = {
                    'dataset': dataset_name,
                    'problem_id': data['problem_id'],
                    'problem': data['problem_text'],
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
            
            # Progress update
            accuracy = sum(1 for r in results if r['is_correct']) / len(results) if results else 0
            logger.info(f"Processed {len(results)}/{min(max_samples, len(problems))} problems. Current accuracy: {accuracy:.1%}")
            
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
            
            # Correlation analysis
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
        
        plt.figure(figsize=(15, 10))
        
        # Plot 1: Entropy distribution
        plt.subplot(2, 3, 1)
        correct_entropies = df[df['is_correct'] == True]['avg_entropy']
        incorrect_entropies = df[df['is_correct'] == False]['avg_entropy']
        
        plt.hist(correct_entropies, alpha=0.7, label='Correct', bins=20, color='green')
        plt.hist(incorrect_entropies, alpha=0.7, label='Incorrect', bins=20, color='red')
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
        data_to_plot = [correct_entropies, incorrect_entropies]
        box_plot = plt.boxplot(data_to_plot, labels=['Correct', 'Incorrect'])
        plt.ylabel('Average Entropy')
        plt.title('Entropy Distribution')
        
        # Plot 4: Token evolution
        plt.subplot(2, 3, 4)
        max_tokens = min(50, max(len(e) for e in df['entropies'] if e))
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
        
        # Plot 5: Accuracy by entropy quartiles
        plt.subplot(2, 3, 5)
        df['entropy_quartile'] = pd.qcut(df['avg_entropy'], 4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
        quartile_accuracy = df.groupby('entropy_quartile')['is_correct'].mean()
        quartile_accuracy.plot(kind='bar', color=['skyblue', 'lightgreen', 'orange', 'red'])
        plt.xlabel('Entropy Quartile')
        plt.ylabel('Accuracy')
        plt.title('Accuracy by Entropy Quartile')
        plt.xticks(rotation=0)
        
        # Plot 6: Dataset comparison
        plt.subplot(2, 3, 6)
        if 'dataset' in df.columns:
            dataset_stats = df.groupby('dataset').agg({
                'is_correct': 'mean',
                'avg_entropy': 'mean'
            }).reset_index()
            
            plt.scatter(dataset_stats['avg_entropy'], dataset_stats['is_correct'], s=100)
            for _, row in dataset_stats.iterrows():
                plt.annotate(row['dataset'], (row['avg_entropy'], row['is_correct']))
            plt.xlabel('Average Entropy')
            plt.ylabel('Accuracy')
            plt.title('Dataset Comparison')
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/entropy_analysis_gpu.png", dpi=300, bbox_inches='tight')
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="GPU-optimized entropy evaluation")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct", help="Model name")
    parser.add_argument("--device_map", default="auto", help="Device mapping strategy")
    parser.add_argument("--max_samples", type=int, default=200, help="Max samples per dataset")
    parser.add_argument("--output_dir", default="./entropy_results_gpu", help="Output directory")
    
    args = parser.parse_args()
    
    # Check GPU availability
    if not torch.cuda.is_available():
        logger.error("No GPUs available! This script requires CUDA.")
        return
    
    logger.info(f"Found {torch.cuda.device_count()} GPUs")
    for i in range(torch.cuda.device_count()):
        gpu_name = torch.cuda.get_device_name(i)
        gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
        logger.info(f"GPU {i}: {gpu_name} ({gpu_memory:.1f}GB)")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize evaluator 
    evaluator = GPUEntropyEvaluator(args.model, args.device_map)
    
    # Load datasets
    logger.info("Loading datasets...")
    math_problems = evaluator.load_math_500()
    polymath_problems = evaluator.load_polymath_en()
    
    all_results = []
    
    # Evaluate MATH-500
    if math_problems:
        logger.info(f"Evaluating MATH-500 ({len(math_problems)} problems available)")
        math_results = evaluator.evaluate_dataset("MATH-500", math_problems, args.max_samples)
        all_results.extend(math_results)
        
        with open(f"{args.output_dir}/math_500_results.json", 'w') as f:
            json.dump(math_results, f, indent=2, default=str)
    
    # Evaluate PolyMath-en
    if polymath_problems:
        logger.info(f"Evaluating PolyMath-en ({len(polymath_problems)} problems available)")
        polymath_results = evaluator.evaluate_dataset("PolyMath-en", polymath_problems, args.max_samples)
        all_results.extend(polymath_results)
        
        with open(f"{args.output_dir}/polymath_results.json", 'w') as f:
            json.dump(polymath_results, f, indent=2, default=str)
    
    if all_results:
        # Analyze results
        logger.info("Analyzing results...")
        analysis = evaluator.analyze_results(all_results)
        
        # Print comprehensive results
        print("\n" + "="*70)
        print("ENTROPY vs CORRECTNESS ANALYSIS (GPU-Optimized)")
        print("="*70)
        print(f"Model: {args.model}")
        print(f"Total problems evaluated: {analysis.get('total_problems', 0)}")
        print(f"Correct answers: {analysis.get('correct_count', 0)}")
        print(f"Incorrect answers: {analysis.get('incorrect_count', 0)}")
        print(f"Overall accuracy: {analysis.get('accuracy', 0):.1%}")
        
        if 'correct_avg_entropy' in analysis:
            print(f"\nEntropy Statistics:")
            print(f"  Correct answers   - avg entropy: {analysis['correct_avg_entropy']:.3f} ± {analysis['correct_entropy_std']:.3f}")
            print(f"  Incorrect answers - avg entropy: {analysis['incorrect_avg_entropy']:.3f} ± {analysis['incorrect_entropy_std']:.3f}")
            print(f"  Entropy difference (incorrect - correct): {analysis['entropy_difference']:.3f}")
            
            if 'pearson_correlation' in analysis:
                print(f"\nCorrelation Analysis:")
                print(f"  Pearson correlation:  {analysis['pearson_correlation']:.3f} (p={analysis['pearson_p_value']:.3f})")
                print(f"  Spearman correlation: {analysis['spearman_correlation']:.3f} (p={analysis['spearman_p_value']:.3f})")
                
                # Interpretation
                if analysis['pearson_p_value'] < 0.05:
                    direction = 'higher' if analysis['pearson_correlation'] > 0 else 'lower'
                    correctness = 'correct' if analysis['pearson_correlation'] > 0 else 'incorrect'
                    print(f"\n🔍 FINDING: Significant correlation detected!")
                    print(f"   {direction.title()} entropy is associated with {correctness} answers.")
                    print(f"   This suggests entropy could be used as an uncertainty measure.")
                else:
                    print(f"\n📊 FINDING: No significant correlation between entropy and correctness.")
                    print(f"   The model appears well-calibrated in terms of confidence.")
        
        # Save results
        with open(f"{args.output_dir}/all_results.json", 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
        
        with open(f"{args.output_dir}/analysis.json", 'w') as f:
            json.dump(analysis, f, indent=2, default=str)
        
        # Create visualizations
        logger.info("Creating visualizations...")
        evaluator.plot_results(all_results, args.output_dir)
        
        logger.info(f"Results saved to {args.output_dir}/")
        print(f"\n📁 All results and visualizations saved to: {args.output_dir}/")
        
        # Final GPU memory report
        if torch.cuda.is_available():
            print(f"\n💾 Final GPU Memory Usage:")
            for i in range(torch.cuda.device_count()):
                memory_used = torch.cuda.memory_allocated(i) / 1024**3
                memory_total = torch.cuda.get_device_properties(i).total_memory / 1024**3
                print(f"   GPU {i}: {memory_used:.1f}GB / {memory_total:.1f}GB ({memory_used/memory_total:.1%})")
    
    else:
        logger.error("No results to analyze. Check dataset loading and model setup.")

if __name__ == "__main__":
    main()
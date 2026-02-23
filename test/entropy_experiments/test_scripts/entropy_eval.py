#!/usr/bin/env python3
import torch
import numpy as np
import pandas as pd
import json
import re
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

class EntropyEvaluator:
    def __init__(self, model_name: str = "Qwen/Qwen2.5-7B-Instruct", device: str = "auto"):
        self.model_name = model_name
        self.device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        
        logger.info(f"Loading model {model_name} on {self.device}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None,
            trust_remote_code=True
        )
        
        if self.device != "cuda":
            self.model = self.model.to(self.device)
        
        self.model.eval()
        
        # Add pad token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def calculate_entropy(self, logits: torch.Tensor) -> float:
        """Calculate entropy from logits"""
        probs = torch.softmax(logits, dim=-1)
        # Add small epsilon to avoid log(0)
        entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)
        return entropy.item()

    def generate_with_entropy(self, prompt: str, max_new_tokens: int = 100) -> Tuple[str, List[float]]:
        """Generate text and track entropy for first 100 tokens"""
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        entropies = []
        generated_tokens = []
        
        with torch.no_grad():
            # Get initial prompt embeddings
            input_ids = inputs["input_ids"]
            attention_mask = inputs["attention_mask"]
            
            for step in range(min(max_new_tokens, 100)):  # Only track first 100 tokens
                outputs = self.model(input_ids, attention_mask=attention_mask)
                logits = outputs.logits[:, -1, :]  # Get logits for last token
                
                # Calculate entropy
                entropy = self.calculate_entropy(logits)
                entropies.append(entropy)
                
                # Sample next token
                next_token_id = torch.multinomial(torch.softmax(logits, dim=-1), 1)
                generated_tokens.append(next_token_id.item())
                
                # Update input_ids and attention_mask
                input_ids = torch.cat([input_ids, next_token_id], dim=-1)
                attention_mask = torch.cat([attention_mask, torch.ones(1, 1).to(self.device)], dim=-1)
                
                # Stop if EOS token is generated
                if next_token_id.item() == self.tokenizer.eos_token_id:
                    break
        
        # Decode generated text
        generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        return generated_text, entropies

    def extract_answer(self, text: str) -> str:
        """Extract final answer from generated text"""
        # Look for common answer patterns
        patterns = [
            r"The answer is\s*([^.\n]+)",
            r"Answer:\s*([^.\n]+)",
            r"Final answer:\s*([^.\n]+)",
            r"\$\\boxed\{([^}]+)\}",
            r"Therefore,?\s*([^.\n]+)",
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        # If no pattern matches, return last line or sentence
        lines = text.strip().split('\n')
        return lines[-1].strip() if lines else text.strip()

    def load_math_500(self) -> List[Dict]:
        """Load MATH-500 dataset"""
        try:
            dataset = load_dataset("hendrycks/competition_math", split="test")
            # Take first 500 examples
            return list(dataset.select(range(min(500, len(dataset)))))
        except Exception as e:
            logger.error(f"Failed to load MATH dataset: {e}")
            return []

    def load_polymath_en(self) -> List[Dict]:
        """Load PolyMath-en dataset"""
        try:
            # Try to load from common polymath dataset names
            possible_names = [
                "math_qa", 
                "microsoft/PolyMath",
                "AI-MO/aimo-validation-math-level-4"
            ]
            
            for name in possible_names:
                try:
                    dataset = load_dataset(name, split="validation" if "validation" in name else "test")
                    return list(dataset.select(range(min(500, len(dataset)))))
                except:
                    continue
            
            logger.warning("Could not load PolyMath-en dataset, using only MATH-500")
            return []
        except Exception as e:
            logger.error(f"Failed to load PolyMath dataset: {e}")
            return []

    def format_math_prompt(self, problem: str) -> str:
        """Format math problem as chat prompt"""
        return f"""Please solve this math problem step by step and provide the final answer.

Problem: {problem}

Solution:"""

    def check_answer_correctness(self, predicted: str, ground_truth: str) -> bool:
        """Check if predicted answer matches ground truth"""
        # Normalize both answers
        def normalize_answer(ans):
            ans = str(ans).lower().strip()
            # Remove common prefixes/suffixes
            ans = re.sub(r'^(the answer is|answer:|final answer:)\s*', '', ans, flags=re.IGNORECASE)
            ans = re.sub(r'\s*(dollars?|cents?|percent|%|\$)\s*$', '', ans, flags=re.IGNORECASE)
            # Extract numbers if present
            numbers = re.findall(r'-?\d+\.?\d*', ans)
            if numbers:
                try:
                    return float(numbers[-1])  # Take last number
                except:
                    pass
            return ans.strip()
        
        pred_norm = normalize_answer(predicted)
        truth_norm = normalize_answer(ground_truth)
        
        # Direct comparison
        if pred_norm == truth_norm:
            return True
        
        # Numerical comparison if both are numbers
        try:
            return abs(float(pred_norm) - float(truth_norm)) < 1e-6
        except:
            pass
        
        # Substring matching for partial credit
        return str(pred_norm) in str(truth_norm) or str(truth_norm) in str(pred_norm)

    def evaluate_dataset(self, dataset_name: str, problems: List[Dict], max_samples: int = 100) -> List[Dict]:
        """Evaluate model on dataset and collect entropy/correctness data"""
        results = []
        
        for i, problem in enumerate(tqdm(problems[:max_samples], desc=f"Evaluating {dataset_name}")):
            try:
                # Format problem based on dataset structure
                if 'problem' in problem:
                    problem_text = problem['problem']
                    answer = problem.get('solution', problem.get('answer', ''))
                elif 'question' in problem:
                    problem_text = problem['question']
                    answer = problem.get('answer', '')
                else:
                    continue
                
                # Generate solution with entropy tracking
                prompt = self.format_math_prompt(problem_text)
                generated_text, entropies = self.generate_with_entropy(prompt)
                
                # Extract predicted answer
                predicted_answer = self.extract_answer(generated_text)
                
                # Check correctness
                is_correct = self.check_answer_correctness(predicted_answer, answer)
                
                # Calculate average entropy for first 100 tokens
                avg_entropy = np.mean(entropies) if entropies else 0.0
                max_entropy = np.max(entropies) if entropies else 0.0
                min_entropy = np.min(entropies) if entropies else 0.0
                
                results.append({
                    'dataset': dataset_name,
                    'problem_id': i,
                    'problem': problem_text,
                    'ground_truth': answer,
                    'predicted_answer': predicted_answer,
                    'generated_text': generated_text,
                    'is_correct': is_correct,
                    'entropies': entropies,
                    'avg_entropy': avg_entropy,
                    'max_entropy': max_entropy,
                    'min_entropy': min_entropy,
                    'num_tokens': len(entropies)
                })
                
                if (i + 1) % 10 == 0:
                    logger.info(f"Processed {i+1}/{min(max_samples, len(problems))} problems from {dataset_name}")
                    
            except Exception as e:
                logger.error(f"Error processing problem {i}: {e}")
                continue
        
        return results

    def analyze_results(self, results: List[Dict]) -> Dict:
        """Analyze correlation between entropy and correctness"""
        if not results:
            return {}
        
        # Convert to DataFrame for easier analysis
        df = pd.DataFrame(results)
        
        # Separate correct and incorrect answers
        correct_results = df[df['is_correct'] == True]
        incorrect_results = df[df['is_correct'] == False]
        
        analysis = {
            'total_problems': len(df),
            'correct_count': len(correct_results),
            'incorrect_count': len(incorrect_results),
            'accuracy': len(correct_results) / len(df) if len(df) > 0 else 0.0,
        }
        
        if len(correct_results) > 0 and len(incorrect_results) > 0:
            # Calculate statistics
            analysis.update({
                'correct_avg_entropy': correct_results['avg_entropy'].mean(),
                'incorrect_avg_entropy': incorrect_results['avg_entropy'].mean(),
                'correct_entropy_std': correct_results['avg_entropy'].std(),
                'incorrect_entropy_std': incorrect_results['avg_entropy'].std(),
                'entropy_difference': incorrect_results['avg_entropy'].mean() - correct_results['avg_entropy'].mean()
            })
            
            # Correlation analysis
            if len(df) > 10:  # Need sufficient data for correlation
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
        """Create visualizations of entropy vs correctness"""
        if not results:
            return
        
        df = pd.DataFrame(results)
        
        # Plot 1: Entropy distribution by correctness
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 2, 1)
        correct_entropies = df[df['is_correct'] == True]['avg_entropy']
        incorrect_entropies = df[df['is_correct'] == False]['avg_entropy']
        
        plt.hist(correct_entropies, alpha=0.7, label='Correct', bins=20, color='green')
        plt.hist(incorrect_entropies, alpha=0.7, label='Incorrect', bins=20, color='red')
        plt.xlabel('Average Entropy')
        plt.ylabel('Frequency')
        plt.title('Entropy Distribution by Correctness')
        plt.legend()
        
        # Plot 2: Scatter plot
        plt.subplot(2, 2, 2)
        colors = ['green' if correct else 'red' for correct in df['is_correct']]
        plt.scatter(df['avg_entropy'], df['is_correct'].astype(int), c=colors, alpha=0.6)
        plt.xlabel('Average Entropy')
        plt.ylabel('Correctness (0/1)')
        plt.title('Entropy vs Correctness')
        
        # Plot 3: Box plot
        plt.subplot(2, 2, 3)
        data_to_plot = [correct_entropies, incorrect_entropies]
        plt.boxplot(data_to_plot, labels=['Correct', 'Incorrect'])
        plt.ylabel('Average Entropy')
        plt.title('Entropy Distribution by Correctness')
        
        # Plot 4: Token-level entropy evolution
        plt.subplot(2, 2, 4)
        if 'entropies' in df.columns:
            # Average entropy across first tokens for correct vs incorrect
            max_tokens = min(50, max(len(e) for e in df['entropies'] if e))
            correct_entropy_evolution = []
            incorrect_entropy_evolution = []
            
            for token_idx in range(max_tokens):
                correct_entropies_at_token = [e[token_idx] for e in df[df['is_correct'] == True]['entropies'] if len(e) > token_idx]
                incorrect_entropies_at_token = [e[token_idx] for e in df[df['is_correct'] == False]['entropies'] if len(e) > token_idx]
                
                if correct_entropies_at_token:
                    correct_entropy_evolution.append(np.mean(correct_entropies_at_token))
                if incorrect_entropies_at_token:
                    incorrect_entropy_evolution.append(np.mean(incorrect_entropies_at_token))
            
            plt.plot(range(len(correct_entropy_evolution)), correct_entropy_evolution, 'g-', label='Correct', alpha=0.8)
            plt.plot(range(len(incorrect_entropy_evolution)), incorrect_entropy_evolution, 'r-', label='Incorrect', alpha=0.8)
            plt.xlabel('Token Position')
            plt.ylabel('Average Entropy')
            plt.title('Entropy Evolution by Token Position')
            plt.legend()
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/entropy_analysis.png", dpi=300, bbox_inches='tight')
        plt.show()

def main():
    parser = argparse.ArgumentParser(description="Evaluate entropy vs correctness on math datasets")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct", help="Model name")
    parser.add_argument("--device", default="auto", help="Device to use")
    parser.add_argument("--max_samples", type=int, default=100, help="Max samples per dataset")
    parser.add_argument("--output_dir", default="./entropy_results", help="Output directory")
    
    args = parser.parse_args()
    
    # Create output directory
    import os
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize evaluator
    evaluator = EntropyEvaluator(args.model, args.device)
    
    # Load datasets
    logger.info("Loading MATH-500 dataset...")
    math_problems = evaluator.load_math_500()
    
    logger.info("Loading PolyMath-en dataset...")
    polymath_problems = evaluator.load_polymath_en()
    
    all_results = []
    
    # Evaluate MATH-500
    if math_problems:
        logger.info(f"Evaluating on MATH-500 ({len(math_problems)} problems)")
        math_results = evaluator.evaluate_dataset("MATH-500", math_problems, args.max_samples)
        all_results.extend(math_results)
        
        # Save intermediate results
        with open(f"{args.output_dir}/math_500_results.json", 'w') as f:
            json.dump(math_results, f, indent=2, default=str)
    
    # Evaluate PolyMath-en
    if polymath_problems:
        logger.info(f"Evaluating on PolyMath-en ({len(polymath_problems)} problems)")
        polymath_results = evaluator.evaluate_dataset("PolyMath-en", polymath_problems, args.max_samples)
        all_results.extend(polymath_results)
        
        # Save intermediate results
        with open(f"{args.output_dir}/polymath_results.json", 'w') as f:
            json.dump(polymath_results, f, indent=2, default=str)
    
    if all_results:
        # Analyze results
        logger.info("Analyzing results...")
        analysis = evaluator.analyze_results(all_results)
        
        # Print summary
        print("\n" + "="*50)
        print("ENTROPY vs CORRECTNESS ANALYSIS")
        print("="*50)
        print(f"Total problems evaluated: {analysis.get('total_problems', 0)}")
        print(f"Correct answers: {analysis.get('correct_count', 0)}")
        print(f"Incorrect answers: {analysis.get('incorrect_count', 0)}")
        print(f"Accuracy: {analysis.get('accuracy', 0):.3f}")
        
        if 'correct_avg_entropy' in analysis:
            print(f"\nEntropy Statistics:")
            print(f"Correct answers - avg entropy: {analysis['correct_avg_entropy']:.3f} ± {analysis['correct_entropy_std']:.3f}")
            print(f"Incorrect answers - avg entropy: {analysis['incorrect_avg_entropy']:.3f} ± {analysis['incorrect_entropy_std']:.3f}")
            print(f"Entropy difference (incorrect - correct): {analysis['entropy_difference']:.3f}")
            
            if 'pearson_correlation' in analysis:
                print(f"\nCorrelation Analysis:")
                print(f"Pearson correlation: {analysis['pearson_correlation']:.3f} (p={analysis['pearson_p_value']:.3f})")
                print(f"Spearman correlation: {analysis['spearman_correlation']:.3f} (p={analysis['spearman_p_value']:.3f})")
                
                # Interpret results
                if analysis['pearson_p_value'] < 0.05:
                    print(f"Significant correlation found! Higher entropy is associated with {'correct' if analysis['pearson_correlation'] > 0 else 'incorrect'} answers.")
                else:
                    print("No significant correlation found between entropy and correctness.")
        
        # Save all results and analysis
        with open(f"{args.output_dir}/all_results.json", 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
        
        with open(f"{args.output_dir}/analysis.json", 'w') as f:
            json.dump(analysis, f, indent=2, default=str)
        
        # Create visualizations
        evaluator.plot_results(all_results, args.output_dir)
        
        logger.info(f"Results saved to {args.output_dir}/")
    else:
        logger.error("No results to analyze. Check dataset loading.")

if __name__ == "__main__":
    main()
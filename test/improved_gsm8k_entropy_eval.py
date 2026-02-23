#!/usr/bin/env python3
"""
Improved GSM8K entropy evaluation with proper chat formatting and 16-sample analysis
Fixed to achieve proper ~80% accuracy for Qwen2.5-7B-Instruct
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

class ImprovedGSM8KEvaluator:
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
        
        # Check if model has chat template
        self.has_chat_template = hasattr(self.tokenizer, 'chat_template') and self.tokenizer.chat_template is not None
        logger.info(f"Model has chat template: {self.has_chat_template}")

    def format_chat_prompt(self, question: str) -> str:
        """Format prompt using proper chat template if available"""
        
        # Standard mathematical reasoning prompt
        system_message = "You are a helpful assistant that solves math problems step by step."
        user_message = f"Please solve this math problem step by step and provide the final numerical answer.\n\nProblem: {question}\n\nPlease show your work and clearly state your final answer as a number."
        
        if self.has_chat_template:
            # Use chat template
            messages = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
            ]
            
            try:
                prompt = self.tokenizer.apply_chat_template(
                    messages, 
                    tokenize=False, 
                    add_generation_prompt=True
                )
                return prompt
            except Exception as e:
                logger.warning(f"Chat template failed: {e}, falling back to manual format")
        
        # Fallback to manual Qwen format
        prompt = f"<|im_start|>system\n{system_message}<|im_end|>\n<|im_start|>user\n{user_message}<|im_end|>\n<|im_start|>assistant\n"
        return prompt

    def calculate_entropy(self, logits: torch.Tensor) -> float:
        """Calculate entropy from logits"""
        with torch.no_grad():
            probs = torch.softmax(logits, dim=-1)
            entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)
            return entropy.cpu().item()

    @torch.no_grad()
    def generate_multiple_samples_with_entropy(self, prompt: str, num_samples: int = 16, max_new_tokens: int = 512) -> List[Dict]:
        """
        Generate multiple samples with entropy tracking
        """
        results = []
        
        # Process in batches for memory efficiency
        batch_size = 4
        for batch_start in range(0, num_samples, batch_size):
            batch_end = min(batch_start + batch_size, num_samples)
            batch_size_actual = batch_end - batch_start
            
            # Tokenize prompt
            inputs = self.tokenizer(prompt, return_tensors="pt")
            device = next(self.model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Repeat for batch
            batch_inputs = {k: v.repeat(batch_size_actual, 1) for k, v in inputs.items()}
            
            # Generate with proper sampling parameters for mathematical reasoning
            outputs = self.model.generate(
                **batch_inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,  # Moderate temperature for some diversity
                top_p=0.95,
                top_k=50,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                output_scores=True,
                return_dict_in_generate=True,
                repetition_penalty=1.1
            )
            
            # Process each sample in the batch
            for sample_idx in range(batch_size_actual):
                # Extract generated tokens (remove input tokens)
                input_length = inputs['input_ids'].shape[1]
                generated_tokens = outputs.sequences[sample_idx][input_length:]
                generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
                
                # Calculate entropy for different token ranges
                entropies = []
                if hasattr(outputs, 'scores') and outputs.scores:
                    for i, score in enumerate(outputs.scores):
                        if i < len(outputs.scores):  # Make sure we don't go out of bounds
                            entropy = self.calculate_entropy(score[sample_idx])
                            entropies.append(entropy)
                
                # Calculate entropy statistics for different ranges
                entropy_stats = self._calculate_entropy_ranges(entropies)
                
                result = {
                    'generated_text': generated_text,
                    'all_entropies': entropies,
                    **entropy_stats
                }
                
                results.append(result)
            
            # Memory cleanup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        return results

    def _calculate_entropy_ranges(self, entropies: List[float]) -> Dict:
        """Calculate entropy statistics for different token ranges"""
        ranges = {
            'first_50': entropies[:50] if len(entropies) > 0 else [],
            'first_100': entropies[:100] if len(entropies) > 0 else [],
            'first_200': entropies[:200] if len(entropies) > 0 else [],
            'all': entropies
        }
        
        stats = {}
        for range_name, entropy_subset in ranges.items():
            if entropy_subset:
                stats[f'avg_entropy_{range_name}'] = np.mean(entropy_subset)
                stats[f'max_entropy_{range_name}'] = np.max(entropy_subset)
                stats[f'min_entropy_{range_name}'] = np.min(entropy_subset)
                stats[f'std_entropy_{range_name}'] = np.std(entropy_subset)
                stats[f'num_tokens_{range_name}'] = len(entropy_subset)
            else:
                stats[f'avg_entropy_{range_name}'] = 0.0
                stats[f'max_entropy_{range_name}'] = 0.0
                stats[f'min_entropy_{range_name}'] = 0.0
                stats[f'std_entropy_{range_name}'] = 0.0
                stats[f'num_tokens_{range_name}'] = 0
        
        return stats

    def extract_numerical_answer(self, text: str) -> str:
        """Improved numerical answer extraction"""
        
        # Clean the text
        text = text.strip()
        
        # Strategy 1: Look for explicit final answer patterns
        final_answer_patterns = [
            r'(?:the final answer is|final answer:|answer:|therefore,?\s*the answer is)[:\s]*\$?([+-]?\d+(?:,\d{3})*(?:\.\d+)?)',
            r'(?:the answer is|answer is)[:\s]*\$?([+-]?\d+(?:,\d{3})*(?:\.\d+)?)',
            r'(?:so|thus|therefore)[,\s]+(?:the answer is)?[:\s]*\$?([+-]?\d+(?:,\d{3})*(?:\.\d+)?)',
        ]
        
        for pattern in final_answer_patterns:
            matches = re.findall(pattern, text.lower())
            if matches:
                return matches[-1].replace(',', '')  # Take the last match, remove commas
        
        # Strategy 2: Look for boxed answers (LaTeX style)
        boxed_patterns = [
            r'\\boxed\{([^}]+)\}',
            r'boxed\{([^}]+)\}',
            r'\$\\boxed\{([^}]+)\}\$',
        ]
        
        for pattern in boxed_patterns:
            matches = re.findall(pattern, text)
            if matches:
                # Extract number from the boxed content
                boxed_content = matches[-1]
                numbers = re.findall(r'([+-]?\d+(?:,\d{3})*(?:\.\d+)?)', boxed_content)
                if numbers:
                    return numbers[-1].replace(',', '')
        
        # Strategy 3: Look for calculation results (equations with =)
        lines = text.split('\n')
        for line in reversed(lines[-10:]):  # Check last 10 lines
            if '=' in line and not any(word in line.lower() for word in ['let', 'step', 'first', 'next', 'then']):
                # Find the part after the last equals sign
                parts = line.split('=')
                if len(parts) >= 2:
                    result_part = parts[-1].strip()
                    # Extract number from this part
                    numbers = re.findall(r'\$?([+-]?\d+(?:,\d{3})*(?:\.\d+)?)', result_part)
                    if numbers:
                        return numbers[0].replace(',', '')
        
        # Strategy 4: Look for the last substantial number in the text
        # Find all numbers and take the last substantial one
        all_numbers = re.findall(r'([+-]?\d+(?:,\d{3})*(?:\.\d+)?)', text)
        if all_numbers:
            # Filter out likely non-answer numbers (very small, very common)
            substantial_numbers = []
            for num_str in reversed(all_numbers):
                try:
                    num = float(num_str.replace(',', ''))
                    # Skip very small numbers (likely step numbers) and very large numbers (likely intermediate calculations)
                    if 0.01 <= abs(num) <= 1000000:
                        substantial_numbers.append(num_str.replace(',', ''))
                    if len(substantial_numbers) >= 5:  # Don't check too many
                        break
                except:
                    continue
            
            if substantial_numbers:
                return substantial_numbers[0]  # Return the last substantial number
            elif all_numbers:
                return all_numbers[-1].replace(',', '')  # Fallback to very last number
        
        return ""

    def check_answer_correctness(self, predicted: str, expected: str) -> bool:
        """Check if predicted answer matches expected answer"""
        try:
            # Clean both answers
            pred_clean = predicted.replace('$', '').replace(',', '').strip()
            exp_clean = expected.replace('$', '').replace(',', '').strip()
            
            if not pred_clean or not exp_clean:
                return False
            
            # Try exact string match first
            if pred_clean.lower() == exp_clean.lower():
                return True
            
            # Try numerical comparison
            pred_num = float(pred_clean)
            exp_num = float(exp_clean)
            
            # Use relative tolerance for larger numbers
            tolerance = max(1e-6, abs(exp_num) * 1e-9)
            return abs(pred_num - exp_num) <= tolerance
            
        except (ValueError, TypeError):
            return False

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

    def evaluate_subset(self, problems: List[Dict], max_questions: int = 100, samples_per_question: int = 16) -> List[Dict]:
        """Evaluate model on GSM8K subset with multiple samples per question"""
        results = []
        
        problems_subset = problems[:max_questions]
        
        for q_idx, problem in enumerate(tqdm(problems_subset, desc="Processing questions")):
            question = problem['question']
            # Extract expected answer from GSM8K format (after ####)
            expected_answer = problem['answer'].split('#### ')[-1].strip()
            
            # Format with proper chat template
            prompt = self.format_chat_prompt(question)
            
            try:
                # Generate multiple samples
                samples = self.generate_multiple_samples_with_entropy(prompt, samples_per_question)
                
                # Process each sample
                sample_results = []
                correct_count = 0
                
                for sample_idx, sample_data in enumerate(samples):
                    generated_text = sample_data['generated_text']
                    predicted_answer = self.extract_numerical_answer(generated_text)
                    is_correct = self.check_answer_correctness(predicted_answer, expected_answer)
                    
                    if is_correct:
                        correct_count += 1
                    
                    sample_result = {
                        'sample_id': sample_idx,
                        'generated_text': generated_text,
                        'predicted_answer': predicted_answer,
                        'is_correct': is_correct,
                        **{k: v for k, v in sample_data.items() if k != 'generated_text'}
                    }
                    sample_results.append(sample_result)
                
                # Calculate question-level statistics
                question_accuracy = correct_count / samples_per_question
                
                # Aggregate entropy statistics across samples
                question_entropy_stats = self._aggregate_question_entropy_stats(sample_results)
                
                question_result = {
                    'question_id': q_idx,
                    'question': question,
                    'expected_answer': expected_answer,
                    'samples_per_question': samples_per_question,
                    'correct_count': correct_count,
                    'question_accuracy': question_accuracy,
                    'samples': sample_results,
                    **question_entropy_stats
                }
                
                results.append(question_result)
                
                # Progress update
                overall_accuracy = np.mean([r['question_accuracy'] for r in results])
                logger.info(f"Q{q_idx+1}: {correct_count}/{samples_per_question} correct ({question_accuracy:.1%}). Overall: {overall_accuracy:.1%}")
                
                # Show some examples for debugging
                if q_idx < 3:
                    logger.info(f"Example answers: Expected='{expected_answer}', Predicted={[s['predicted_answer'] for s in sample_results[:3]]}")
                
            except Exception as e:
                logger.error(f"Error processing question {q_idx}: {e}")
                continue
            
            # Memory cleanup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        return results

    def _aggregate_question_entropy_stats(self, sample_results: List[Dict]) -> Dict:
        """Aggregate entropy statistics across all samples for a question"""
        ranges = ['first_50', 'first_100', 'first_200', 'all']
        question_stats = {}
        
        for range_name in ranges:
            # Collect entropy stats from all samples
            avg_entropies = [s[f'avg_entropy_{range_name}'] for s in sample_results if s[f'num_tokens_{range_name}'] > 0]
            max_entropies = [s[f'max_entropy_{range_name}'] for s in sample_results if s[f'num_tokens_{range_name}'] > 0]
            std_entropies = [s[f'std_entropy_{range_name}'] for s in sample_results if s[f'num_tokens_{range_name}'] > 0]
            
            if avg_entropies:
                question_stats[f'question_avg_entropy_{range_name}'] = np.mean(avg_entropies)
                question_stats[f'question_max_entropy_{range_name}'] = np.mean(max_entropies)
                question_stats[f'question_std_entropy_{range_name}'] = np.mean(std_entropies)
                question_stats[f'question_entropy_variance_{range_name}'] = np.var(avg_entropies)
            else:
                question_stats[f'question_avg_entropy_{range_name}'] = 0.0
                question_stats[f'question_max_entropy_{range_name}'] = 0.0
                question_stats[f'question_std_entropy_{range_name}'] = 0.0
                question_stats[f'question_entropy_variance_{range_name}'] = 0.0
        
        return question_stats

    def analyze_entropy_accuracy_correlation(self, results: List[Dict]) -> Dict:
        """Analyze correlation between entropy measures and question accuracy"""
        if not results:
            return {}
        
        # Create DataFrame with question-level data
        question_data = []
        for r in results:
            row = {
                'question_id': r['question_id'],
                'question_accuracy': r['question_accuracy'],
                'correct_count': r['correct_count'],
            }
            # Add entropy statistics
            for key, value in r.items():
                if key.startswith('question_'):
                    row[key] = value
            question_data.append(row)
        
        df = pd.DataFrame(question_data)
        
        analysis = {
            'total_questions': len(df),
            'total_samples': sum(r['samples_per_question'] for r in results),
            'overall_accuracy': df['question_accuracy'].mean(),
            'accuracy_std': df['question_accuracy'].std(),
            'perfect_questions': len(df[df['question_accuracy'] == 1.0]),
            'zero_questions': len(df[df['question_accuracy'] == 0.0]),
            'partial_questions': len(df[(df['question_accuracy'] > 0) & (df['question_accuracy'] < 1.0)]),
        }
        
        # Correlation analysis for each token range
        ranges = ['first_50', 'first_100', 'first_200', 'all']
        correlations = {}
        
        for range_name in ranges:
            avg_entropy_col = f'question_avg_entropy_{range_name}'
            max_entropy_col = f'question_max_entropy_{range_name}'
            var_entropy_col = f'question_entropy_variance_{range_name}'
            
            if avg_entropy_col in df.columns and len(df) > 10:
                # Calculate correlations
                pearson_avg, p_avg = pearsonr(df[avg_entropy_col], df['question_accuracy'])
                pearson_max, p_max = pearsonr(df[max_entropy_col], df['question_accuracy'])
                pearson_var, p_var = pearsonr(df[var_entropy_col], df['question_accuracy'])
                
                spearman_avg, sp_avg = spearmanr(df[avg_entropy_col], df['question_accuracy'])
                
                correlations[range_name] = {
                    'avg_entropy_mean': df[avg_entropy_col].mean(),
                    'max_entropy_mean': df[max_entropy_col].mean(),
                    'entropy_variance_mean': df[var_entropy_col].mean(),
                    
                    'pearson_avg_entropy': pearson_avg,
                    'p_value_avg_entropy': p_avg,
                    'pearson_max_entropy': pearson_max,
                    'p_value_max_entropy': p_max,
                    'pearson_entropy_variance': pearson_var,
                    'p_value_entropy_variance': p_var,
                    
                    'spearman_avg_entropy': spearman_avg,
                    'spearman_p_value': sp_avg,
                }
        
        analysis['correlations'] = correlations
        
        return analysis

    def create_visualizations(self, results: List[Dict], analysis: Dict, output_dir: str):
        """Create comprehensive visualizations"""
        if not results:
            return
        
        # Prepare data
        question_data = []
        for r in results:
            row = {'question_accuracy': r['question_accuracy']}
            for key, value in r.items():
                if key.startswith('question_'):
                    row[key] = value
            question_data.append(row)
        
        df = pd.DataFrame(question_data)
        ranges = ['first_50', 'first_100', 'first_200', 'all']
        
        # Create main correlation plot
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        fig.suptitle(f'GSM8K: Entropy vs Question Accuracy Analysis\nOverall Accuracy: {analysis["overall_accuracy"]:.1%}', fontsize=16)
        
        for i, range_name in enumerate(ranges):
            avg_entropy_col = f'question_avg_entropy_{range_name}'
            var_entropy_col = f'question_entropy_variance_{range_name}'
            
            if avg_entropy_col not in df.columns:
                continue
            
            # Top row: Average entropy vs accuracy
            ax = axes[0, i]
            ax.scatter(df[avg_entropy_col], df['question_accuracy'], alpha=0.6, color='blue', s=30)
            ax.set_xlabel(f'Average Entropy ({range_name})')
            ax.set_ylabel('Question Accuracy')
            ax.set_title(f'Avg Entropy vs Accuracy\n({range_name} tokens)')
            ax.grid(True, alpha=0.3)
            
            # Add correlation info
            if range_name in analysis.get('correlations', {}):
                corr = analysis['correlations'][range_name]['pearson_avg_entropy']
                p_val = analysis['correlations'][range_name]['p_value_avg_entropy']
                significance = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
                ax.text(0.05, 0.95, f'r = {corr:.3f}{significance}\np = {p_val:.3f}', 
                       transform=ax.transAxes, bbox=dict(boxstyle="round", facecolor='wheat', alpha=0.8))
            
            # Bottom row: Entropy variance vs accuracy
            ax = axes[1, i]
            ax.scatter(df[var_entropy_col], df['question_accuracy'], alpha=0.6, color='red', s=30)
            ax.set_xlabel(f'Entropy Variance ({range_name})')
            ax.set_ylabel('Question Accuracy')
            ax.set_title(f'Entropy Variance vs Accuracy\n({range_name} tokens)')
            ax.grid(True, alpha=0.3)
            
            if range_name in analysis.get('correlations', {}):
                corr = analysis['correlations'][range_name]['pearson_entropy_variance']
                p_val = analysis['correlations'][range_name]['p_value_entropy_variance']
                significance = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
                ax.text(0.05, 0.95, f'r = {corr:.3f}{significance}\np = {p_val:.3f}', 
                       transform=ax.transAxes, bbox=dict(boxstyle="round", facecolor='lightcoral', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/entropy_accuracy_correlations.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        # Create accuracy distribution plot
        self._plot_accuracy_distribution(df, analysis, output_dir)

    def _plot_accuracy_distribution(self, df: pd.DataFrame, analysis: Dict, output_dir: str):
        """Plot accuracy distribution and summary statistics"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('GSM8K Question Accuracy Distribution', fontsize=16)
        
        # Histogram of accuracies
        ax = axes[0, 0]
        ax.hist(df['question_accuracy'], bins=17, alpha=0.7, edgecolor='black', color='skyblue')
        ax.set_xlabel('Question Accuracy')
        ax.set_ylabel('Number of Questions')
        ax.set_title('Distribution of Question Accuracy')
        ax.axvline(analysis['overall_accuracy'], color='red', linestyle='--', 
                  label=f'Mean: {analysis["overall_accuracy"]:.1%}')
        ax.legend()
        
        # Pie chart of performance categories
        ax = axes[0, 1]
        perfect = analysis['perfect_questions']
        partial = analysis['partial_questions']
        zero = analysis['zero_questions']
        
        sizes = [perfect, partial, zero]
        labels = [f'Perfect (100%)\n{perfect} questions', 
                 f'Partial (1-99%)\n{partial} questions', 
                 f'Zero (0%)\n{zero} questions']
        colors = ['green', 'orange', 'red']
        
        ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        ax.set_title('Question Performance Categories')
        
        # Accuracy by entropy quartiles
        ax = axes[1, 0]
        if 'question_avg_entropy_first_100' in df.columns:
            df['entropy_quartile'] = pd.qcut(df['question_avg_entropy_first_100'], 4, 
                                           labels=['Q1 (Low)', 'Q2', 'Q3', 'Q4 (High)'])
            quartile_acc = df.groupby('entropy_quartile')['question_accuracy'].mean()
            quartile_acc.plot(kind='bar', ax=ax, color=['green', 'yellowgreen', 'orange', 'red'])
            ax.set_xlabel('Entropy Quartile (First 100 tokens)')
            ax.set_ylabel('Average Accuracy')
            ax.set_title('Accuracy by Entropy Quartile')
            ax.tick_params(axis='x', rotation=45)
        
        # Summary statistics
        ax = axes[1, 1]
        ax.axis('off')
        summary_text = f"""
        Summary Statistics:
        
        Total Questions: {analysis['total_questions']}
        Total Samples: {analysis['total_samples']}
        
        Overall Accuracy: {analysis['overall_accuracy']:.1%} ± {analysis['accuracy_std']:.1%}
        
        Perfect Questions: {analysis['perfect_questions']} ({analysis['perfect_questions']/analysis['total_questions']*100:.1f}%)
        Partial Questions: {analysis['partial_questions']} ({analysis['partial_questions']/analysis['total_questions']*100:.1f}%)
        Zero Questions: {analysis['zero_questions']} ({analysis['zero_questions']/analysis['total_questions']*100:.1f}%)
        """
        ax.text(0.1, 0.9, summary_text, transform=ax.transAxes, fontsize=12, 
               verticalalignment='top', bbox=dict(boxstyle="round", facecolor='lightgray', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/accuracy_distribution.png", dpi=300, bbox_inches='tight')
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Improved GSM8K entropy evaluation with 16 samples")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct", help="Model name")
    parser.add_argument("--max_questions", type=int, default=100, help="Max questions to evaluate")
    parser.add_argument("--samples_per_question", type=int, default=16, help="Samples per question")
    parser.add_argument("--output_dir", default="./improved_gsm8k_results", help="Output directory")
    
    args = parser.parse_args()
    
    if not torch.cuda.is_available():
        logger.error("No GPUs available!")
        return
    
    # Create output directory with timestamp
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"{args.output_dir}_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    evaluator = ImprovedGSM8KEvaluator(args.model)
    
    # Load GSM8K dataset
    gsm8k_problems = evaluator.load_gsm8k()
    if not gsm8k_problems:
        logger.error("Failed to load GSM8K dataset")
        return
    
    logger.info(f"Starting improved evaluation: {args.max_questions} questions × {args.samples_per_question} samples")
    
    # Run evaluation
    results = evaluator.evaluate_subset(gsm8k_problems, args.max_questions, args.samples_per_question)
    
    if results:
        logger.info("Analyzing results...")
        analysis = evaluator.analyze_entropy_accuracy_correlation(results)
        
        # Print comprehensive results
        print(f"\n{'='*80}")
        print("IMPROVED GSM8K ENTROPY vs ACCURACY ANALYSIS")
        print(f"{'='*80}")
        print(f"Model: {args.model}")
        print(f"Questions evaluated: {analysis['total_questions']}")
        print(f"Total samples generated: {analysis['total_samples']}")
        print(f"Samples per question: {args.samples_per_question}")
        print(f"\nOVERALL ACCURACY: {analysis['overall_accuracy']:.1%} ± {analysis['accuracy_std']:.1%}")
        print(f"Perfect questions (100%): {analysis['perfect_questions']}")
        print(f"Partial questions (1-99%): {analysis['partial_questions']}")
        print(f"Zero questions (0%): {analysis['zero_questions']}")
        
        # Show correlation results
        if 'correlations' in analysis:
            print(f"\n📊 ENTROPY-ACCURACY CORRELATION ANALYSIS:")
            print(f"{'Token Range':<15} {'Avg Entropy':<15} {'Max Entropy':<15} {'Entropy Variance':<15}")
            print("-" * 75)
            
            significant_findings = []
            for range_name, corr_data in analysis['correlations'].items():
                avg_corr = corr_data['pearson_avg_entropy']
                avg_p = corr_data['p_value_avg_entropy']
                max_corr = corr_data['pearson_max_entropy']
                max_p = corr_data['p_value_max_entropy']
                var_corr = corr_data['pearson_entropy_variance']
                var_p = corr_data['p_value_entropy_variance']
                
                # Format with significance indicators
                avg_sig = "***" if avg_p < 0.001 else "**" if avg_p < 0.01 else "*" if avg_p < 0.05 else ""
                max_sig = "***" if max_p < 0.001 else "**" if max_p < 0.01 else "*" if max_p < 0.05 else ""
                var_sig = "***" if var_p < 0.001 else "**" if var_p < 0.01 else "*" if var_p < 0.05 else ""
                
                print(f"{range_name:<15} {avg_corr:>6.3f}{avg_sig:<3} {max_corr:>6.3f}{max_sig:<3} {var_corr:>6.3f}{var_sig:<3}")
                
                # Collect significant findings
                if avg_p < 0.05 and abs(avg_corr) > 0.1:
                    significant_findings.append(f"Average entropy ({range_name}): r={avg_corr:.3f}, p={avg_p:.3f}")
                if max_p < 0.05 and abs(max_corr) > 0.1:
                    significant_findings.append(f"Max entropy ({range_name}): r={max_corr:.3f}, p={max_p:.3f}")
                if var_p < 0.05 and abs(var_corr) > 0.1:
                    significant_findings.append(f"Entropy variance ({range_name}): r={var_corr:.3f}, p={var_p:.3f}")
            
            print(f"\nSignificance: *** p<0.001, ** p<0.01, * p<0.05")
            
            if significant_findings:
                print(f"\n🎯 SIGNIFICANT FINDINGS (p < 0.05, |r| > 0.1):")
                for finding in significant_findings:
                    print(f"  ✓ {finding}")
            else:
                print(f"\n📈 No significant correlations found between entropy measures and question accuracy.")
        
        # Save results
        with open(f"{output_dir}/results.json", 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        with open(f"{output_dir}/analysis.json", 'w') as f:
            json.dump(analysis, f, indent=2, default=str)
        
        logger.info("Creating visualizations...")
        evaluator.create_visualizations(results, analysis, output_dir)
        
        print(f"\n📁 Results saved to: {output_dir}/")
        
        # Show some sample results for verification
        print(f"\n📋 Sample Results (first 3 questions):")
        for i in range(min(3, len(results))):
            r = results[i]
            print(f"\nQuestion {i+1}: {r['question'][:80]}...")
            print(f"  Expected: {r['expected_answer']}")
            print(f"  Accuracy: {r['correct_count']}/{r['samples_per_question']} = {r['question_accuracy']:.1%}")
            sample_answers = [s['predicted_answer'] for s in r['samples'][:5]]
            print(f"  Sample answers: {sample_answers}")
        
    else:
        logger.error("No results to analyze.")

if __name__ == "__main__":
    main()
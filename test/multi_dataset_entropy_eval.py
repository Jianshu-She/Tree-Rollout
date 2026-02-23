#!/usr/bin/env python3
"""
Multi-dataset entropy evaluation for GSM8K and MATH-500
Supports multiple samples per question with comprehensive entropy tracking
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

class MultiDatasetEntropyEvaluator:
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
        
        self.has_chat_template = hasattr(self.tokenizer, 'chat_template') and self.tokenizer.chat_template is not None
        logger.info(f"Model has chat template: {self.has_chat_template}")

    def format_chat_prompt(self, question: str, dataset_type: str = "gsm8k") -> str:
        """Format prompt using proper chat template for different datasets"""
        
        if dataset_type == "math":
            # MATH dataset requires more careful problem-solving
            system_message = "You are a helpful mathematics assistant. Solve the problem step by step, showing all your work. Provide your final answer in the simplest form."
            user_message = f"Problem: {question}\n\nPlease solve this step by step and provide your final answer."
        else:
            # GSM8K style
            system_message = "You are a helpful assistant that solves math problems step by step."
            user_message = f"Please solve this math problem step by step and provide the final numerical answer.\n\nProblem: {question}\n\nPlease show your work and clearly state your final answer as a number."
        
        if self.has_chat_template:
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
        """Generate multiple samples with entropy tracking"""
        results = []
        
        # Process in batches for memory efficiency
        batch_size = 4
        for batch_start in range(0, num_samples, batch_size):
            batch_end = min(batch_start + batch_size, num_samples)
            batch_size_actual = batch_end - batch_start
            
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
            device = next(self.model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            batch_inputs = {k: v.repeat(batch_size_actual, 1) for k, v in inputs.items()}
            
            outputs = self.model.generate(
                **batch_inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.95,
                top_k=50,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                output_scores=True,
                return_dict_in_generate=True,
                repetition_penalty=1.1
            )
            
            for sample_idx in range(batch_size_actual):
                input_length = inputs['input_ids'].shape[1]
                generated_tokens = outputs.sequences[sample_idx][input_length:]
                generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
                
                entropies = []
                if hasattr(outputs, 'scores') and outputs.scores:
                    for i, score in enumerate(outputs.scores):
                        if i < len(outputs.scores):
                            entropy = self.calculate_entropy(score[sample_idx])
                            entropies.append(entropy)
                
                entropy_stats = self._calculate_entropy_ranges(entropies)
                
                result = {
                    'generated_text': generated_text,
                    'all_entropies': entropies,
                    **entropy_stats
                }
                
                results.append(result)
            
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

    def extract_answer_gsm8k(self, text: str, expected_answer: str = None) -> str:
        """Extract numerical answer from GSM8K-style response"""
        
        # Clean the text
        text = text.strip()
        
        # Look for explicit final answer patterns
        final_answer_patterns = [
            r'(?:the final answer is|final answer:|answer:|therefore,?\s*the answer is)[:\s]*\$?([+-]?\d+(?:,\d{3})*(?:\.\d+)?)',
            r'(?:the answer is|answer is)[:\s]*\$?([+-]?\d+(?:,\d{3})*(?:\.\d+)?)',
            r'(?:so|thus|therefore)[,\s]+(?:the answer is)?[:\s]*\$?([+-]?\d+(?:,\d{3})*(?:\.\d+)?)',
        ]
        
        for pattern in final_answer_patterns:
            matches = re.findall(pattern, text.lower())
            if matches:
                return matches[-1].replace(',', '')
        
        # Look for boxed answers
        boxed_patterns = [
            r'\\boxed\{([^}]+)\}',
            r'boxed\{([^}]+)\}',
        ]
        
        for pattern in boxed_patterns:
            matches = re.findall(pattern, text)
            if matches:
                boxed_content = matches[-1]
                numbers = re.findall(r'([+-]?\d+(?:,\d{3})*(?:\.\d+)?)', boxed_content)
                if numbers:
                    return numbers[-1].replace(',', '')
        
        # Look for calculation results
        lines = text.split('\n')
        for line in reversed(lines[-10:]):
            if '=' in line and not any(word in line.lower() for word in ['let', 'step', 'first', 'next', 'then']):
                parts = line.split('=')
                if len(parts) >= 2:
                    result_part = parts[-1].strip()
                    numbers = re.findall(r'\$?([+-]?\d+(?:,\d{3})*(?:\.\d+)?)', result_part)
                    if numbers:
                        return numbers[0].replace(',', '')
        
        # Last resort: find the last substantial number
        all_numbers = re.findall(r'([+-]?\d+(?:,\d{3})*(?:\.\d+)?)', text)
        if all_numbers:
            # Filter out likely non-answer numbers
            substantial_numbers = []
            for num_str in reversed(all_numbers):
                try:
                    num = float(num_str.replace(',', ''))
                    if 0.01 <= abs(num) <= 1000000:
                        substantial_numbers.append(num_str.replace(',', ''))
                    if len(substantial_numbers) >= 5:
                        break
                except:
                    continue
            
            if substantial_numbers:
                return substantial_numbers[0]
            elif all_numbers:
                return all_numbers[-1].replace(',', '')
        
        return ""

    def extract_answer_math(self, text: str, expected_answer: str = None) -> str:
        """Extract answer from MATH dataset response - handles various mathematical formats"""
        
        text = text.strip()
        
        # MATH dataset often uses \boxed{} for final answers
        boxed_patterns = [
            r'\\boxed\{([^}]+)\}',
            r'\\boxed\s*\{([^}]+)\}',
            r'boxed\{([^}]+)\}',
        ]
        
        for pattern in boxed_patterns:
            matches = re.findall(pattern, text, re.DOTALL)
            if matches:
                # Take the last boxed answer
                answer = matches[-1].strip()
                # Clean up LaTeX formatting
                answer = answer.replace('\\$', '$').replace('$', '').strip()
                answer = answer.replace('\\,', '').replace(',', '')
                answer = answer.replace('\\frac', 'frac')
                return answer
        
        # Look for explicit final answer statements
        final_patterns = [
            r'(?:the final answer is|final answer:|answer:|therefore)[:\s]*([^\n.]+)',
            r'(?:the answer is|answer is)[:\s]*([^\n.]+)',
        ]
        
        for pattern in final_patterns:
            matches = re.findall(pattern, text.lower())
            if matches:
                answer = matches[-1].strip()
                # Clean common endings
                answer = re.sub(r'\.$', '', answer)
                answer = answer.replace('$', '').strip()
                return answer
        
        # For numerical answers, find the last number
        if expected_answer and re.match(r'^[+-]?\d+(?:\.\d+)?$', str(expected_answer).replace(',', '')):
            numbers = re.findall(r'([+-]?\d+(?:,\d{3})*(?:\.\d+)?)', text)
            if numbers:
                return numbers[-1].replace(',', '')
        
        # For fraction answers
        fractions = re.findall(r'\\frac\{([^}]+)\}\{([^}]+)\}', text)
        if fractions:
            num, den = fractions[-1]
            return f"\\frac{{{num}}}{{{den}}}"
        
        # Simple fractions
        simple_fractions = re.findall(r'(\d+)/(\d+)', text)
        if simple_fractions:
            return f"{simple_fractions[-1][0]}/{simple_fractions[-1][1]}"
        
        return ""

    def normalize_answer_math(self, answer: str) -> str:
        """Normalize MATH dataset answers for comparison"""
        if not answer:
            return ""
        
        # Remove spaces and lowercase
        answer = answer.strip().lower()
        
        # Handle fractions
        if '\\frac' in answer:
            # Extract numerator and denominator
            frac_match = re.match(r'\\frac\{([^}]+)\}\{([^}]+)\}', answer)
            if frac_match:
                num, den = frac_match.groups()
                try:
                    # Try to simplify
                    num_val = float(num)
                    den_val = float(den)
                    if den_val != 0:
                        return str(num_val / den_val)
                except:
                    pass
        
        # Handle simple fractions
        if '/' in answer:
            parts = answer.split('/')
            if len(parts) == 2:
                try:
                    num = float(parts[0])
                    den = float(parts[1])
                    if den != 0:
                        return str(num / den)
                except:
                    pass
        
        # Try to convert to float for numerical comparison
        try:
            return str(float(answer))
        except:
            return answer

    def check_answer_correctness(self, predicted: str, expected: str, dataset_type: str = "gsm8k") -> bool:
        """Check if predicted answer matches expected answer"""
        if not predicted or not expected:
            return False
        
        # Clean both answers
        pred_clean = predicted.replace('$', '').replace(',', '').strip()
        exp_clean = expected.replace('$', '').replace(',', '').strip()
        
        # Exact match
        if pred_clean.lower() == exp_clean.lower():
            return True
        
        if dataset_type == "math":
            # Normalize both answers for MATH dataset
            pred_norm = self.normalize_answer_math(pred_clean)
            exp_norm = self.normalize_answer_math(exp_clean)
            
            if pred_norm == exp_norm:
                return True
            
            # Try numerical comparison if both can be converted to float
            try:
                pred_num = float(pred_norm)
                exp_num = float(exp_norm)
                tolerance = max(1e-6, abs(exp_num) * 1e-9)
                return abs(pred_num - exp_num) <= tolerance
            except:
                pass
            
            # Check fraction equivalence
            if '/' in pred_clean and '/' in exp_clean:
                return pred_clean == exp_clean
            
        else:
            # GSM8K - numerical comparison
            try:
                pred_num = float(pred_clean)
                exp_num = float(exp_clean)
                tolerance = max(1e-6, abs(exp_num) * 1e-9)
                return abs(pred_num - exp_num) <= tolerance
            except:
                pass
        
        return False

    def load_gsm8k(self, max_samples: int = None) -> List[Dict]:
        """Load GSM8K dataset"""
        try:
            logger.info("Loading GSM8K test set...")
            dataset = load_dataset("gsm8k", "main", split="test")
            problems = list(dataset)
            
            if max_samples:
                problems = problems[:max_samples]
                
            logger.info(f"Loaded {len(problems)} problems from GSM8K")
            return problems
        except Exception as e:
            logger.error(f"Failed to load GSM8K: {e}")
            return []

    def load_math500(self, max_samples: int = None) -> List[Dict]:
        """Load MATH-500 dataset"""
        try:
            logger.info("Loading MATH dataset...")
            # Try different dataset names for MATH
            dataset = None
            dataset_names = [
                "lighteval/MATH",
                "TIGER-Lab/MATH",
                "competition_math"
            ]
            
            for name in dataset_names:
                try:
                    dataset = load_dataset(name, split="test")
                    logger.info(f"Successfully loaded MATH dataset from {name}")
                    break
                except:
                    continue
            
            if dataset is None:
                raise Exception("Could not load MATH dataset from any known source")
            
            # MATH-500 is typically a 500-problem subset
            problems = list(dataset)
            
            # Take first 500 or specified max_samples
            if max_samples:
                problems = problems[:min(max_samples, 500)]
            else:
                problems = problems[:500]
            
            logger.info(f"Loaded {len(problems)} problems from MATH dataset")
            return problems
        except Exception as e:
            logger.error(f"Failed to load MATH dataset: {e}")
            return []

    def evaluate_dataset(self, dataset_name: str, problems: List[Dict], 
                        max_questions: int = 100, samples_per_question: int = 16) -> List[Dict]:
        """Evaluate model on specified dataset with multiple samples per question"""
        results = []
        
        problems_subset = problems[:max_questions]
        
        for q_idx, problem in enumerate(tqdm(problems_subset, desc=f"Processing {dataset_name} questions")):
            # Extract question and answer based on dataset format
            if dataset_name == "gsm8k":
                question = problem['question']
                expected_answer = problem['answer'].split('#### ')[-1].strip()
            else:  # MATH dataset
                question = problem['problem']
                expected_answer = problem['solution'].strip()
                # Extract answer from solution if it contains \boxed{}
                boxed_match = re.search(r'\\boxed\{([^}]+)\}', expected_answer)
                if boxed_match:
                    expected_answer = boxed_match.group(1)
            
            # Format prompt
            prompt = self.format_chat_prompt(question, dataset_type="math" if dataset_name == "math" else "gsm8k")
            
            try:
                # Generate multiple samples
                samples = self.generate_multiple_samples_with_entropy(prompt, samples_per_question)
                
                # Process each sample
                sample_results = []
                correct_count = 0
                
                for sample_idx, sample_data in enumerate(samples):
                    generated_text = sample_data['generated_text']
                    
                    # Extract predicted answer based on dataset type
                    if dataset_name == "gsm8k":
                        predicted_answer = self.extract_answer_gsm8k(generated_text, expected_answer)
                    else:
                        predicted_answer = self.extract_answer_math(generated_text, expected_answer)
                    
                    # Check correctness
                    is_correct = self.check_answer_correctness(
                        predicted_answer, expected_answer, 
                        dataset_type="math" if dataset_name == "math" else "gsm8k"
                    )
                    
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
                
                # Aggregate entropy statistics
                question_entropy_stats = self._aggregate_question_entropy_stats(sample_results)
                
                question_result = {
                    'dataset': dataset_name,
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
                logger.info(f"{dataset_name} Q{q_idx+1}: {correct_count}/{samples_per_question} correct ({question_accuracy:.1%}). Overall: {overall_accuracy:.1%}")
                
                # Show examples for debugging
                if q_idx < 3:
                    logger.info(f"Expected: '{expected_answer}', Predicted samples: {[s['predicted_answer'] for s in sample_results[:3]]}")
                
            except Exception as e:
                logger.error(f"Error processing question {q_idx}: {e}")
                continue
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        return results

    def _aggregate_question_entropy_stats(self, sample_results: List[Dict]) -> Dict:
        """Aggregate entropy statistics across all samples for a question"""
        ranges = ['first_50', 'first_100', 'first_200', 'all']
        question_stats = {}
        
        for range_name in ranges:
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

    def analyze_results(self, results: List[Dict], dataset_name: str) -> Dict:
        """Analyze correlation between entropy and accuracy"""
        if not results:
            return {}
        
        question_data = []
        for r in results:
            row = {
                'question_id': r['question_id'],
                'question_accuracy': r['question_accuracy'],
                'correct_count': r['correct_count'],
            }
            for key, value in r.items():
                if key.startswith('question_'):
                    row[key] = value
            question_data.append(row)
        
        df = pd.DataFrame(question_data)
        
        analysis = {
            'dataset': dataset_name,
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

    def create_comparison_visualization(self, gsm8k_analysis: Dict, math_analysis: Dict, output_dir: str):
        """Create comparison visualization between GSM8K and MATH results"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('GSM8K vs MATH-500: Entropy-Accuracy Analysis Comparison', fontsize=16)
        
        # 1. Overall accuracy comparison
        ax = axes[0, 0]
        datasets = ['GSM8K', 'MATH-500']
        accuracies = [gsm8k_analysis['overall_accuracy'], math_analysis['overall_accuracy']]
        stds = [gsm8k_analysis['accuracy_std'], math_analysis['accuracy_std']]
        
        bars = ax.bar(datasets, accuracies, yerr=stds, capsize=10, color=['#3498db', '#e74c3c'])
        ax.set_ylabel('Overall Accuracy')
        ax.set_title('Dataset Performance Comparison')
        ax.set_ylim(0, 1.0)
        
        for i, (acc, std) in enumerate(zip(accuracies, stds)):
            ax.text(i, acc + std + 0.02, f'{acc:.1%}', ha='center')
        
        # 2. Correlation comparison
        ax = axes[0, 1]
        ranges = ['first_50', 'first_100', 'first_200', 'all']
        gsm8k_corrs = []
        math_corrs = []
        
        for range_name in ranges:
            if range_name in gsm8k_analysis.get('correlations', {}):
                gsm8k_corrs.append(gsm8k_analysis['correlations'][range_name]['pearson_avg_entropy'])
            else:
                gsm8k_corrs.append(0)
            
            if range_name in math_analysis.get('correlations', {}):
                math_corrs.append(math_analysis['correlations'][range_name]['pearson_avg_entropy'])
            else:
                math_corrs.append(0)
        
        x = np.arange(len(ranges))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, gsm8k_corrs, width, label='GSM8K', color='#3498db')
        bars2 = ax.bar(x + width/2, math_corrs, width, label='MATH-500', color='#e74c3c')
        
        ax.set_ylabel('Pearson Correlation')
        ax.set_title('Entropy-Accuracy Correlations by Token Range')
        ax.set_xticks(x)
        ax.set_xticklabels(ranges)
        ax.legend()
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        
        # 3. Question performance distribution
        ax = axes[0, 2]
        categories = ['Perfect\n(100%)', 'Partial\n(1-99%)', 'Zero\n(0%)']
        gsm8k_dist = [gsm8k_analysis['perfect_questions'], gsm8k_analysis['partial_questions'], gsm8k_analysis['zero_questions']]
        math_dist = [math_analysis['perfect_questions'], math_analysis['partial_questions'], math_analysis['zero_questions']]
        
        x = np.arange(len(categories))
        bars1 = ax.bar(x - width/2, gsm8k_dist, width, label='GSM8K', color='#3498db')
        bars2 = ax.bar(x + width/2, math_dist, width, label='MATH-500', color='#e74c3c')
        
        ax.set_ylabel('Number of Questions')
        ax.set_title('Question Performance Distribution')
        ax.set_xticks(x)
        ax.set_xticklabels(categories)
        ax.legend()
        
        # 4. Average entropy comparison
        ax = axes[1, 0]
        avg_entropies_gsm8k = []
        avg_entropies_math = []
        
        for range_name in ranges:
            if range_name in gsm8k_analysis.get('correlations', {}):
                avg_entropies_gsm8k.append(gsm8k_analysis['correlations'][range_name]['avg_entropy_mean'])
            else:
                avg_entropies_gsm8k.append(0)
            
            if range_name in math_analysis.get('correlations', {}):
                avg_entropies_math.append(math_analysis['correlations'][range_name]['avg_entropy_mean'])
            else:
                avg_entropies_math.append(0)
        
        x = np.arange(len(ranges))
        bars1 = ax.bar(x - width/2, avg_entropies_gsm8k, width, label='GSM8K', color='#3498db')
        bars2 = ax.bar(x + width/2, avg_entropies_math, width, label='MATH-500', color='#e74c3c')
        
        ax.set_ylabel('Average Entropy')
        ax.set_title('Average Entropy by Token Range')
        ax.set_xticks(x)
        ax.set_xticklabels(ranges)
        ax.legend()
        
        # 5. Summary statistics
        ax = axes[1, 1]
        ax.axis('off')
        summary_text = f"""
        GSM8K Summary:
        - Questions: {gsm8k_analysis['total_questions']}
        - Samples: {gsm8k_analysis['total_samples']}
        - Accuracy: {gsm8k_analysis['overall_accuracy']:.1%} ± {gsm8k_analysis['accuracy_std']:.1%}
        - Strongest correlation: {min(gsm8k_corrs):.3f}
        
        MATH-500 Summary:
        - Questions: {math_analysis['total_questions']}
        - Samples: {math_analysis['total_samples']}
        - Accuracy: {math_analysis['overall_accuracy']:.1%} ± {math_analysis['accuracy_std']:.1%}
        - Strongest correlation: {min(math_corrs):.3f}
        """
        ax.text(0.1, 0.9, summary_text, transform=ax.transAxes, fontsize=12,
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # 6. Key findings
        ax = axes[1, 2]
        ax.axis('off')
        
        # Determine key findings
        gsm8k_sig = any(gsm8k_analysis['correlations'][r]['p_value_avg_entropy'] < 0.05 for r in ranges if r in gsm8k_analysis['correlations'])
        math_sig = any(math_analysis['correlations'][r]['p_value_avg_entropy'] < 0.05 for r in ranges if r in math_analysis['correlations'])
        
        findings = "Key Findings:\n\n"
        if gsm8k_sig:
            findings += "✓ GSM8K: Significant negative correlation\n  between entropy and accuracy\n\n"
        else:
            findings += "✗ GSM8K: No significant correlation found\n\n"
        
        if math_sig:
            findings += "✓ MATH-500: Significant negative correlation\n  between entropy and accuracy\n\n"
        else:
            findings += "✗ MATH-500: No significant correlation found\n\n"
        
        findings += f"Both datasets show {'similar' if (gsm8k_sig == math_sig) else 'different'} patterns"
        
        ax.text(0.1, 0.9, findings, transform=ax.transAxes, fontsize=12,
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/dataset_comparison.png", dpi=300, bbox_inches='tight')
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Multi-dataset entropy evaluation")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct", help="Model name")
    parser.add_argument("--datasets", nargs='+', default=["gsm8k", "math"], choices=["gsm8k", "math", "both"],
                       help="Datasets to evaluate")
    parser.add_argument("--max_questions", type=int, default=100, help="Max questions per dataset")
    parser.add_argument("--samples_per_question", type=int, default=16, help="Samples per question")
    parser.add_argument("--output_dir", default="./multi_dataset_entropy_results", help="Output directory")
    
    args = parser.parse_args()
    
    if not torch.cuda.is_available():
        logger.error("No GPUs available!")
        return
    
    # Create output directory with timestamp
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"{args.output_dir}_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    evaluator = MultiDatasetEntropyEvaluator(args.model)
    
    all_results = {}
    all_analyses = {}
    
    # Evaluate GSM8K if requested
    if "gsm8k" in args.datasets or "both" in args.datasets:
        logger.info("=" * 80)
        logger.info("Evaluating GSM8K dataset")
        logger.info("=" * 80)
        
        gsm8k_problems = evaluator.load_gsm8k(args.max_questions)
        if gsm8k_problems:
            gsm8k_results = evaluator.evaluate_dataset(
                "gsm8k", gsm8k_problems, args.max_questions, args.samples_per_question
            )
            
            if gsm8k_results:
                gsm8k_analysis = evaluator.analyze_results(gsm8k_results, "gsm8k")
                all_results['gsm8k'] = gsm8k_results
                all_analyses['gsm8k'] = gsm8k_analysis
                
                # Save GSM8K results
                with open(f"{output_dir}/gsm8k_results.json", 'w') as f:
                    json.dump(gsm8k_results, f, indent=2, default=str)
                
                with open(f"{output_dir}/gsm8k_analysis.json", 'w') as f:
                    json.dump(gsm8k_analysis, f, indent=2, default=str)
    
    # Evaluate MATH-500 if requested
    if "math" in args.datasets or "both" in args.datasets:
        logger.info("=" * 80)
        logger.info("Evaluating MATH-500 dataset")
        logger.info("=" * 80)
        
        math_problems = evaluator.load_math500(args.max_questions)
        if math_problems:
            math_results = evaluator.evaluate_dataset(
                "math", math_problems, args.max_questions, args.samples_per_question
            )
            
            if math_results:
                math_analysis = evaluator.analyze_results(math_results, "math")
                all_results['math'] = math_results
                all_analyses['math'] = math_analysis
                
                # Save MATH results
                with open(f"{output_dir}/math_results.json", 'w') as f:
                    json.dump(math_results, f, indent=2, default=str)
                
                with open(f"{output_dir}/math_analysis.json", 'w') as f:
                    json.dump(math_analysis, f, indent=2, default=str)
    
    # Print comprehensive results
    print(f"\n{'='*80}")
    print("MULTI-DATASET ENTROPY vs ACCURACY ANALYSIS")
    print(f"{'='*80}")
    print(f"Model: {args.model}")
    
    for dataset_name, analysis in all_analyses.items():
        print(f"\n{dataset_name.upper()} Results:")
        print(f"  Questions evaluated: {analysis['total_questions']}")
        print(f"  Total samples: {analysis['total_samples']}")
        print(f"  Overall accuracy: {analysis['overall_accuracy']:.1%} ± {analysis['accuracy_std']:.1%}")
        print(f"  Perfect questions: {analysis['perfect_questions']}")
        print(f"  Partial questions: {analysis['partial_questions']}")
        print(f"  Zero questions: {analysis['zero_questions']}")
        
        if 'correlations' in analysis:
            print(f"\n  Entropy-Accuracy Correlations:")
            for range_name, corr_data in analysis['correlations'].items():
                avg_corr = corr_data['pearson_avg_entropy']
                avg_p = corr_data['p_value_avg_entropy']
                sig = "***" if avg_p < 0.001 else "**" if avg_p < 0.01 else "*" if avg_p < 0.05 else ""
                print(f"    {range_name}: r = {avg_corr:.3f} (p = {avg_p:.3f}) {sig}")
    
    # Create comparison visualization if both datasets were evaluated
    if len(all_analyses) == 2:
        logger.info("Creating comparison visualizations...")
        evaluator.create_comparison_visualization(
            all_analyses.get('gsm8k', {}), 
            all_analyses.get('math', {}), 
            output_dir
        )
    
    print(f"\n📁 Results saved to: {output_dir}/")


if __name__ == "__main__":
    main()
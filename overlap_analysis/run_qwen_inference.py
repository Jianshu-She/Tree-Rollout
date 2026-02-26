#!/usr/bin/env python3
"""Run flat rollout inference on MATH-500 using Qwen3-30B-A3B-Thinking model via vLLM."""

import argparse
import json
import os
import re
import time

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer


SYSTEM_PROMPT = (
    "You are a helpful math assistant. Solve the given math problem step by step. "
    "Show your reasoning clearly and put your final answer in \\boxed{}."
)


def load_math500(path):
    with open(path) as f:
        return json.load(f)


def format_prompts(problems, tokenizer):
    """Format problems using chat template with thinking enabled."""
    prompts = []
    for p in problems:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": p["problem"]},
        ]
        try:
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=True,
            )
        except TypeError:
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
        prompts.append(prompt)
    return prompts


def extract_thinking(response_text):
    """Extract thinking content from <think>...</think> tags."""
    match = re.search(r"<think>(.*?)</think>", response_text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return response_text


def main():
    parser = argparse.ArgumentParser(
        description="Run flat rollout inference on MATH-500 with Qwen3 thinking model.")
    parser.add_argument("--data", default="data-prepare/data/MATH500_train.json")
    parser.add_argument("--model", default="Qwen/Qwen3-30B-A3B-Thinking-2507")
    parser.add_argument("--num_problems", type=int, default=None)
    parser.add_argument("--num_rollouts", type=int, default=12)
    parser.add_argument("--max_tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--tensor_parallel_size", type=int, default=2)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.90)
    parser.add_argument("--problem_offset", type=int, default=0,
                        help="Skip first N problems (for parallel sharding)")
    parser.add_argument("--output_dir", default="overlap_analysis/qwen_results")
    args = parser.parse_args()

    # Load data
    all_problems = load_math500(args.data)
    problems = all_problems[args.problem_offset:]
    if args.num_problems:
        problems = problems[:args.num_problems]
    print(f"Loaded {len(problems)} problems (offset={args.problem_offset})")

    # Load tokenizer
    print(f"Loading tokenizer for {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    # Format prompts
    prompts = format_prompts(problems, tokenizer)
    print(f"Formatted {len(prompts)} prompts (sample length: {len(prompts[0])} chars)")

    # Initialize vLLM
    print(f"Initializing vLLM with {args.model}, tp={args.tensor_parallel_size}...")
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        trust_remote_code=True,
        max_model_len=8192,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )

    sampling_params = SamplingParams(
        n=args.num_rollouts,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
    )

    # Generate
    print(f"Generating {args.num_rollouts} rollouts for {len(problems)} problems...")
    start_time = time.time()
    outputs = llm.generate(prompts, sampling_params)
    elapsed = time.time() - start_time
    print(f"Generation completed in {elapsed:.1f}s")

    # Format results
    results = []
    total_tokens = 0
    for i, (problem, output) in enumerate(zip(problems, outputs)):
        rollouts = []
        for j, completion in enumerate(output.outputs):
            response = completion.text
            thinking = extract_thinking(response)
            n_tokens = len(completion.token_ids)
            total_tokens += n_tokens
            rollouts.append({
                "rollout_id": j,
                "response": response,
                "thinking": thinking,
                "num_tokens": n_tokens,
            })
        results.append({
            "problem_id": args.problem_offset + i,
            "problem": problem["problem"],
            "answer": problem["answer"],
            "rollouts": rollouts,
        })

    # Save
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, "inference_results.json")
    output_data = {
        "config": {
            "model": args.model,
            "num_rollouts": args.num_rollouts,
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
        },
        "num_problems": len(problems),
        "total_tokens_generated": total_tokens,
        "elapsed_seconds": elapsed,
        "results": results,
    }
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"\nResults saved to {output_path}")
    print(f"Total tokens generated: {total_tokens:,}")
    print(f"Avg tokens per rollout: {total_tokens / (len(problems) * args.num_rollouts):.0f}")


if __name__ == "__main__":
    main()

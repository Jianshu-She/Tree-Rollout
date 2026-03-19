#!/usr/bin/env python3
"""Run flat rollouts for a single model on MATH500 problems."""

import argparse
import json
import os
import sys
import time

import numpy as np
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "mcts_inference"))
from mcts_inference.utils import is_correct


def load_math500_problems(json_path, num_problems=10, seed=42, start_idx=None, end_idx=None):
    with open(json_path) as f:
        data = json.load(f)

    if start_idx is not None and end_idx is not None:
        # Use contiguous range [start_idx, end_idx)
        indices = list(range(start_idx, min(end_idx, len(data))))
    else:
        # Random sampling (legacy behavior)
        rng = np.random.RandomState(seed)
        indices = rng.choice(len(data), size=min(num_problems, len(data)), replace=False)
        indices.sort()

    problems = []
    for idx in indices:
        entry = data[idx]
        # Build chat messages for Qwen2.5-Math
        messages = [
            {"role": "system", "content": "Please reason step by step, and put your final answer within \\boxed{}."},
            {"role": "user", "content": entry["problem"]},
        ]
        problems.append({
            "index": int(idx),
            "messages": messages,
            "ground_truth": str(entry["answer"]),
            "level": entry.get("level", "?"),
            "problem_text": entry["problem"][:300],
        })
    return problems


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--data_path", default="data-prepare/data/MATH500_train.json")
    parser.add_argument("--output_dir", default="faithful_baseline/results/training_stages_math500")
    parser.add_argument("--num_problems", type=int, default=10)
    parser.add_argument("--start_idx", type=int, default=None,
                        help="Start index for contiguous problem range (inclusive)")
    parser.add_argument("--end_idx", type=int, default=None,
                        help="End index for contiguous problem range (exclusive)")
    parser.add_argument("--num_rollouts", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=50,
                        help="Number of problems to process per batch")
    parser.add_argument("--max_tokens", type=int, default=16384)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--tensor_parallel_size", type=int, default=2)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.92)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, f"rollouts_{args.model_name}.json")

    if os.path.exists(output_path):
        print(f"Output already exists: {output_path}, skipping")
        return

    # Load problems
    if args.start_idx is not None and args.end_idx is not None:
        print(f"Loading MATH500 problems [{args.start_idx}, {args.end_idx})...")
        problems = load_math500_problems(args.data_path, start_idx=args.start_idx, end_idx=args.end_idx)
    else:
        print(f"Loading {args.num_problems} MATH500 problems (seed={args.seed})...")
        problems = load_math500_problems(args.data_path, args.num_problems, seed=args.seed)
    print(f"  {len(problems)} problems, indices: {[p['index'] for p in problems[:5]]}...{[p['index'] for p in problems[-3:]]}")

    # Save problems metadata
    problems_meta = [{"index": p["index"], "ground_truth": p["ground_truth"],
                       "level": p["level"]} for p in problems]
    meta_path = os.path.join(args.output_dir, "problems.json")
    if not os.path.exists(meta_path):
        with open(meta_path, "w") as f:
            json.dump(problems_meta, f, indent=2)

    # Load model
    print(f"Loading model from {args.model_path} (tp={args.tensor_parallel_size})...")
    llm = LLM(
        model=args.model_path,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=True,
        max_model_len=args.max_tokens + 2048,
    )

    tokenizer = llm.get_tokenizer()
    sampling_params = SamplingParams(
        n=args.num_rollouts,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
    )

    # Build prompts
    prompts = []
    for prob in problems:
        text = tokenizer.apply_chat_template(
            prob["messages"], tokenize=False, add_generation_prompt=True,
        )
        prompts.append(text)

    # Generate in batches to manage memory
    batch_size = args.batch_size
    n_batches = (len(prompts) + batch_size - 1) // batch_size
    print(f"Generating {args.num_rollouts} rollouts × {len(problems)} problems "
          f"(max_tokens={args.max_tokens}, {n_batches} batches of {batch_size})...")

    results = []
    total_correct = 0
    total_rollouts = 0
    t0 = time.time()

    for batch_i in range(n_batches):
        b_start = batch_i * batch_size
        b_end = min(b_start + batch_size, len(prompts))
        batch_prompts = prompts[b_start:b_end]
        batch_problems = problems[b_start:b_end]

        print(f"\n  Batch {batch_i+1}/{n_batches} (problems {b_start}-{b_end-1})...")
        bt0 = time.time()
        outputs = llm.generate(batch_prompts, sampling_params)
        print(f"    Generated in {time.time()-bt0:.1f}s")

        for prob, output in zip(batch_problems, outputs):
            rollouts = []
            for i, completion in enumerate(output.outputs):
                rollouts.append({
                    "rollout_id": i,
                    "response": completion.text,
                    "full_text": completion.text,
                    "num_tokens": len(completion.token_ids),
                })

            n_correct = sum(1 for r in rollouts if is_correct(r["response"], prob["ground_truth"]))
            total_correct += n_correct
            total_rollouts += len(rollouts)

            results.append({
                "problem_index": prob["index"],
                "ground_truth": prob["ground_truth"],
                "answer": prob["ground_truth"],
                "level": prob["level"],
                "prompt": prob["problem_text"],
                "rollouts": rollouts,
                "num_correct": n_correct,
                "accuracy": n_correct / len(rollouts),
            })

        batch_correct = sum(r["num_correct"] for r in results[b_start:b_end])
        batch_total = sum(len(r["rollouts"]) for r in results[b_start:b_end])
        print(f"    Batch acc: {batch_correct}/{batch_total} = {batch_correct/batch_total:.1%}")

    elapsed = time.time() - t0
    print(f"\nGeneration done in {elapsed:.1f}s")
    print(f"Overall: {total_correct}/{total_rollouts} = {total_correct/total_rollouts:.1%}")

    # Save
    save_data = {
        "model": args.model_name,
        "model_path": args.model_path,
        "num_problems": len(problems),
        "num_rollouts": args.num_rollouts,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "results": results,
    }
    with open(output_path, "w") as f:
        json.dump(save_data, f)
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()

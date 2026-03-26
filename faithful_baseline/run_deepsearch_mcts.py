#!/usr/bin/env python3
"""Run DeepSearch-style MCTS on MATH500 problems.

Uses global frontier selection + entropy guidance + dynamic expansion width.
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "mcts_inference"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import MCTSConfig
from deepsearch_engine import DeepSearchEngine
from mcts_inference.utils import is_correct, extract_answer


MODEL_PATHS = {
    "step_0": "/mnt/weka/home/jianshu.she/copus/verl/models/Qwen2.5-Math-7B",
    "step_40": "/mnt/weka/home/jianshu.she/copus/verl/models/global_step_40_hf",
    "step_80": "/mnt/weka/home/jianshu.she/copus/verl/models/global_step_80_hf",
    "step_120": "/mnt/weka/home/jianshu.she/copus/verl/models/global_step_120_hf",
}

DATA_PATH = "data-prepare/data/MATH500_train.json"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=str, default="step_0",
                        choices=["step_0", "step_40", "step_80", "step_120"])
    parser.add_argument("--num_problems", type=int, default=10)
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--num_rollouts", type=int, default=128)
    parser.add_argument("--num_children", type=int, default=8,
                        help="Base expansion width (decays with depth)")
    parser.add_argument("--max_depth", type=int, default=16)
    parser.add_argument("--ds_lambda_quality", type=float, default=1.0)
    parser.add_argument("--ds_lambda_entropy", type=float, default=0.5)
    parser.add_argument("--ds_lambda_depth", type=float, default=0.1)
    parser.add_argument("--ds_width_decay", type=int, default=1)
    parser.add_argument("--ds_min_width", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--tensor_parallel_size", type=int, default=2)
    parser.add_argument("--gpu_ids", type=str, default="0,1")
    parser.add_argument("--output_dir", type=str,
                        default="faithful_baseline/results/mcts_deepsearch")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids

    model_path = MODEL_PATHS[args.stage]
    print(f"\n{'='*60}")
    print(f"DeepSearch MCTS")
    print(f"Stage: {args.stage}")
    print(f"Model: {model_path}")
    print(f"Rollouts: {args.num_rollouts}, Base width: {args.num_children}")
    print(f"Lambda quality={args.ds_lambda_quality}, entropy={args.ds_lambda_entropy}, depth={args.ds_lambda_depth}")
    print(f"{'='*60}")

    config = MCTSConfig(
        policy_model_name=model_path,
        tensor_parallel_size=args.tensor_parallel_size,
        temperature=args.temperature,
        top_p=0.95,
        max_tokens_per_node=256,
        max_depth=args.max_depth,
        num_rollouts=args.num_rollouts,
        num_children=args.num_children,
        exploration_constant=1.414,  # not used in DeepSearch but kept for compat
        prm_type="logprob",
        system_prompt="Please reason step by step, and put your final answer within \\boxed{}.",
    )

    # DeepSearch-specific params (attached to config)
    config.ds_lambda_quality = args.ds_lambda_quality
    config.ds_lambda_entropy = args.ds_lambda_entropy
    config.ds_lambda_depth = args.ds_lambda_depth
    config.ds_width_decay = args.ds_width_decay
    config.ds_min_width = args.ds_min_width

    # Load data
    with open(DATA_PATH) as f:
        all_problems = json.load(f)

    problems = []
    for i in range(args.start_idx, min(args.start_idx + args.num_problems, len(all_problems))):
        entry = all_problems[i]
        problems.append({
            "problem": entry["problem"],
            "answer": entry.get("answer", entry.get("solution", "")),
            "problem_index": i,
        })

    # Initialize engine
    engine = DeepSearchEngine(config)

    # Run
    os.makedirs(args.output_dir, exist_ok=True)
    all_results = []
    t0 = time.time()

    for idx, problem in enumerate(problems):
        print(f"\n--- Problem {problem['problem_index']} ({idx+1}/{len(problems)}) ---")
        t1 = time.time()

        result = engine.solve(problem["problem"])
        elapsed = time.time() - t1

        result["problem_index"] = problem["problem_index"]
        result["ground_truth"] = problem["answer"]
        result["elapsed_seconds"] = round(elapsed, 2)

        if result.get("best_answer"):
            result["correct"] = is_correct(result["best_solution"], problem["answer"])
        else:
            result["correct"] = False

        print(f"  Correct: {result['correct']}, "
              f"Nodes: {result['tree_stats']['total_nodes']}, "
              f"Terminals: {result['tree_stats']['terminal_nodes']}, "
              f"MaxDepth: {result['tree_stats']['max_depth_reached']}, "
              f"Time: {elapsed:.1f}s")

        all_results.append(result)

    total_time = time.time() - t0
    correct_count = sum(1 for r in all_results if r.get("correct"))
    accuracy = correct_count / len(all_results) if all_results else 0

    print(f"\n{'='*60}")
    print(f"RESULTS: DeepSearch {args.stage}")
    print(f"{'='*60}")
    print(f"Accuracy: {correct_count}/{len(all_results)} = {accuracy:.1%}")
    print(f"Total time: {total_time:.1f}s")
    avg_nodes = sum(r["tree_stats"]["total_nodes"] for r in all_results) / len(all_results)
    print(f"Avg nodes/tree: {avg_nodes:.1f}")

    # Save
    output_path = os.path.join(args.output_dir,
                               f"deepsearch_{args.stage}_p{args.start_idx}-{args.start_idx+args.num_problems-1}.json")
    save_data = {
        "config": {
            "method": "deepsearch",
            "stage": args.stage,
            "model": model_path,
            "num_rollouts": args.num_rollouts,
            "base_expansion_width": args.num_children,
            "max_depth": args.max_depth,
            "lambda_quality": args.ds_lambda_quality,
            "lambda_entropy": args.ds_lambda_entropy,
            "lambda_depth": args.ds_lambda_depth,
            "width_decay": args.ds_width_decay,
            "min_width": args.ds_min_width,
            "temperature": args.temperature,
            "prm_type": "logprob",
        },
        "summary": {
            "accuracy": accuracy,
            "correct": correct_count,
            "total": len(all_results),
            "avg_nodes": avg_nodes,
            "total_time": total_time,
        },
        "results": all_results,
    }
    with open(output_path, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()

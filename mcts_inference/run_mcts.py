#!/usr/bin/env python3
"""CLI entry point for MCTS inference on math reasoning problems."""

import argparse
import json
import os
import time

from config import MCTSConfig
from mcts_engine import MCTSEngine


def parse_args():
    p = argparse.ArgumentParser(description="MCTS Inference for Math Reasoning")

    # Data
    p.add_argument(
        "--data",
        type=str,
        default="../data-prepare/data/MATH500_train.json",
        help="Path to dataset JSON (list of {problem, answer})",
    )
    p.add_argument("--num_problems", type=int, default=None, help="Limit number of problems")
    p.add_argument("--output_dir", type=str, default="mcts_results", help="Output directory")

    # Policy model
    p.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--tensor_parallel_size", type=int, default=4)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top_p", type=float, default=0.95)

    # MCTS
    p.add_argument("--max_tokens_per_node", type=int, default=256, help="Tokens per tree node")
    p.add_argument("--max_depth", type=int, default=16, help="Max tree depth")
    p.add_argument("--num_rollouts", type=int, default=32, help="MCTS iterations per problem")
    p.add_argument("--num_children", type=int, default=2, help="Children per expansion")
    p.add_argument("--exploration_constant", type=float, default=1.414, help="UCB1 C")

    # Reward model
    p.add_argument(
        "--prm_type",
        type=str,
        default="logprob",
        choices=["logprob", "frozen_prm"],
        help="Reward model type",
    )
    p.add_argument("--prm_model_path", type=str, default=None, help="Path to trained PRM checkpoint")

    return p.parse_args()


def main():
    args = parse_args()

    # Build config
    config = MCTSConfig(
        policy_model_name=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens_per_node=args.max_tokens_per_node,
        max_depth=args.max_depth,
        num_rollouts=args.num_rollouts,
        num_children=args.num_children,
        exploration_constant=args.exploration_constant,
        prm_type=args.prm_type,
        prm_model_path=args.prm_model_path,
    )

    # Load data
    print(f"Loading data from {args.data} ...")
    with open(args.data, "r") as f:
        problems = json.load(f)

    if args.num_problems:
        problems = problems[: args.num_problems]
    print(f"Loaded {len(problems)} problems")

    # Initialize engine
    engine = MCTSEngine(config)

    # Run
    t0 = time.time()
    results = engine.solve_batch(problems)
    total_time = time.time() - t0

    # Summary
    evaluated = [r for r in results if r.get("correct") is not None]
    correct = sum(1 for r in evaluated if r["correct"])
    accuracy = correct / len(evaluated) if evaluated else 0

    print("\n" + "=" * 60)
    print("MCTS INFERENCE RESULTS")
    print("=" * 60)
    print(f"Problems:       {len(results)}")
    print(f"Evaluated:      {len(evaluated)}")
    print(f"Correct:        {correct}")
    print(f"Accuracy:       {accuracy:.2%}")
    print(f"Total time:     {total_time:.1f}s")
    print(f"Avg time/prob:  {total_time / len(results):.1f}s")
    avg_nodes = sum(r["tree_stats"]["total_nodes"] for r in results) / len(results)
    print(f"Avg nodes/tree: {avg_nodes:.1f}")
    print("=" * 60)

    # Save
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")

    results_path = os.path.join(args.output_dir, f"mcts_results_{timestamp}.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    summary = {
        "config": {
            "model": config.policy_model_name,
            "max_tokens_per_node": config.max_tokens_per_node,
            "max_depth": config.max_depth,
            "num_rollouts": config.num_rollouts,
            "num_children": config.num_children,
            "exploration_constant": config.exploration_constant,
            "temperature": config.temperature,
            "prm_type": config.prm_type,
        },
        "results": {
            "total_problems": len(results),
            "evaluated": len(evaluated),
            "correct": correct,
            "accuracy": accuracy,
            "total_time_seconds": round(total_time, 2),
            "avg_time_per_problem": round(total_time / len(results), 2),
            "avg_nodes_per_tree": round(avg_nodes, 1),
        },
    }

    summary_path = os.path.join(args.output_dir, f"mcts_summary_{timestamp}.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to {results_path}")
    print(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    main()

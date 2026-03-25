#!/usr/bin/env python3
"""Run standard MCTS on MATH500 problems and collect tree statistics.

Uses the same models (Qwen2.5-Math-7B checkpoints) as flat rollouts,
with UCT selection + logprob reward. Collects branching factor and
accuracy distributions at each depth for comparison with flat rollouts.
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "mcts_inference"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from mcts_inference.config import MCTSConfig
from mcts_inference.mcts_engine import MCTSEngine
from mcts_inference.utils import is_correct, extract_answer


MODEL_PATHS = {
    "step_0": "/mnt/weka/home/jianshu.she/copus/verl/models/Qwen2.5-Math-7B",
    "step_40": "/mnt/weka/home/jianshu.she/copus/verl/models/global_step_40_hf",
    "step_80": "/mnt/weka/home/jianshu.she/copus/verl/models/global_step_80_hf",
    "step_120": "/mnt/weka/home/jianshu.she/copus/verl/models/global_step_120_hf",
}

DATA_PATH = "data-prepare/data/MATH500_train.json"


def collect_tree_stats(tree_dict, ground_truth):
    """Collect per-depth branching factors and accuracies from an MCTS tree dict."""

    def _collect(node, depth=0):
        bf_by_depth = {}
        acc_by_depth = {}

        children = node.get("children", [])
        if children:
            bf_by_depth.setdefault(depth, []).append(len(children))

        # Node accuracy: check all terminal descendants
        terminals = _collect_terminals(node)
        if terminals:
            n_correct = sum(1 for t in terminals if t.get("correct", False))
            acc = n_correct / len(terminals)
            acc_by_depth.setdefault(depth, []).append(acc)

        for child in children:
            child_bf, child_acc = _collect(child, depth + 1)
            for d, vals in child_bf.items():
                bf_by_depth.setdefault(d, []).extend(vals)
            for d, vals in child_acc.items():
                acc_by_depth.setdefault(d, []).extend(vals)

        return bf_by_depth, acc_by_depth

    def _collect_terminals(node):
        if node.get("is_terminal", False):
            return [node]
        result = []
        for c in node.get("children", []):
            result.extend(_collect_terminals(c))
        return result

    def _annotate_correctness(node, ground_truth):
        """Mark terminal nodes with correctness."""
        if node.get("is_terminal", False):
            # Reconstruct full text by walking path (text_preview is truncated)
            # Use the answer from all_solutions if available, or check text_preview
            node["correct"] = False  # default
        for c in node.get("children", []):
            _annotate_correctness(c, ground_truth)

    return _collect(tree_dict)


def run_mcts_experiment(
    stage,
    num_problems=10,
    start_idx=0,
    num_rollouts=128,
    num_children=2,
    max_depth=16,
    exploration_constant=1.414,
    temperature=0.7,
    tensor_parallel_size=2,
    gpu_ids="0,1",
    output_dir="faithful_baseline/results/mcts_standard",
):
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids

    model_path = MODEL_PATHS[stage]
    print(f"\n{'='*60}")
    print(f"Stage: {stage}")
    print(f"Model: {model_path}")
    print(f"Problems: {start_idx} to {start_idx + num_problems - 1}")
    print(f"MCTS rollouts: {num_rollouts}, children: {num_children}")
    print(f"{'='*60}")

    config = MCTSConfig(
        policy_model_name=model_path,
        tensor_parallel_size=tensor_parallel_size,
        temperature=temperature,
        top_p=0.95,
        max_tokens_per_node=256,
        max_depth=max_depth,
        num_rollouts=num_rollouts,
        num_children=num_children,
        exploration_constant=exploration_constant,
        prm_type="logprob",
        system_prompt="Please reason step by step, and put your final answer within \\boxed{}.",
    )

    # Load data
    with open(DATA_PATH) as f:
        all_problems = json.load(f)

    problems = []
    for i in range(start_idx, min(start_idx + num_problems, len(all_problems))):
        entry = all_problems[i]
        problems.append({
            "problem": entry["problem"],
            "answer": entry.get("answer", entry.get("solution", "")),
            "problem_index": i,
        })

    print(f"Loaded {len(problems)} problems")

    # Initialize engine
    engine = MCTSEngine(config)

    # Run MCTS
    os.makedirs(output_dir, exist_ok=True)
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

        # Evaluate correctness
        if result.get("best_answer"):
            result["correct"] = is_correct(result["best_solution"], problem["answer"])
        else:
            result["correct"] = False

        # Also evaluate all solutions
        for sol in result.get("all_solutions", []):
            if sol.get("answer"):
                # Reconstruct: we don't have full text, use text_preview
                sol["correct"] = False  # Will compute from answer

        print(f"  Correct: {result['correct']}, "
              f"Nodes: {result['tree_stats']['total_nodes']}, "
              f"Terminals: {result['tree_stats']['terminal_nodes']}, "
              f"Time: {elapsed:.1f}s")

        all_results.append(result)

    total_time = time.time() - t0
    correct_count = sum(1 for r in all_results if r.get("correct"))
    accuracy = correct_count / len(all_results) if all_results else 0

    print(f"\n{'='*60}")
    print(f"RESULTS: {stage}")
    print(f"{'='*60}")
    print(f"Accuracy: {correct_count}/{len(all_results)} = {accuracy:.1%}")
    print(f"Total time: {total_time:.1f}s")
    print(f"Avg time/problem: {total_time/len(all_results):.1f}s")
    avg_nodes = sum(r["tree_stats"]["total_nodes"] for r in all_results) / len(all_results)
    print(f"Avg nodes/tree: {avg_nodes:.1f}")

    # Save results
    output_path = os.path.join(output_dir, f"mcts_{stage}_p{start_idx}-{start_idx+num_problems-1}.json")
    save_data = {
        "config": {
            "stage": stage,
            "model": model_path,
            "num_rollouts": num_rollouts,
            "num_children": num_children,
            "max_depth": max_depth,
            "exploration_constant": exploration_constant,
            "temperature": temperature,
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

    return save_data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=str, default="step_0",
                        choices=["step_0", "step_40", "step_80", "step_120"])
    parser.add_argument("--num_problems", type=int, default=10)
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--num_rollouts", type=int, default=128,
                        help="MCTS iterations (matches flat rollout budget)")
    parser.add_argument("--num_children", type=int, default=2,
                        help="Children per expansion")
    parser.add_argument("--max_depth", type=int, default=16)
    parser.add_argument("--exploration_constant", type=float, default=1.414)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--tensor_parallel_size", type=int, default=2)
    parser.add_argument("--gpu_ids", type=str, default="0,1")
    parser.add_argument("--output_dir", type=str,
                        default="faithful_baseline/results/mcts_standard")
    args = parser.parse_args()

    run_mcts_experiment(
        stage=args.stage,
        num_problems=args.num_problems,
        start_idx=args.start_idx,
        num_rollouts=args.num_rollouts,
        num_children=args.num_children,
        max_depth=args.max_depth,
        exploration_constant=args.exploration_constant,
        temperature=args.temperature,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_ids=args.gpu_ids,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()

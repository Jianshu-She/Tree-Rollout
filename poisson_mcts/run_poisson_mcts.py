#!/usr/bin/env python3
"""Run Poisson-MCTS on MATH500 problems.

Distribution-guided branching: NegBin at D0, Poisson at D1+.
α controls blend between flat-rollout-matching and UCB1 exploitation.
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "mcts_inference"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import MCTSConfig
from poisson_mcts_engine import PoissonMCTSEngine
from mcts_inference.utils import is_correct


MODEL_PATHS = {
    "step_0": "/mnt/weka/home/jianshu.she/copus/verl/models/Qwen2.5-Math-7B",
    "step_40": "/mnt/weka/home/jianshu.she/copus/verl/models/global_step_40_hf",
    "step_80": "/mnt/weka/home/jianshu.she/copus/verl/models/global_step_80_hf",
    "step_120": "/mnt/weka/home/jianshu.she/copus/verl/models/global_step_120_hf",
}

DATA_PATH = "data-prepare/data/MATH500_train.json"
FITTED_PARAMS = "faithful_baseline/results/math500_full/train/poisson_beta_analysis/fitted_parameters.json"


def main():
    parser = argparse.ArgumentParser(description="Run Poisson-MCTS")
    parser.add_argument("--stage", type=str, default="step_0",
                        choices=list(MODEL_PATHS.keys()))
    parser.add_argument("--num_problems", type=int, default=10)
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--num_rollouts", type=int, default=128)
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="0=pure UCB1, 1=flat-rollout-matching")
    parser.add_argument("--bf_temperature", type=float, default=1.0,
                        help="<1 reduces BF variance toward mean")
    parser.add_argument("--exploration_constant", type=float, default=1.414)
    parser.add_argument("--max_depth", type=int, default=16)
    parser.add_argument("--tensor_parallel_size", type=int, default=2)
    parser.add_argument("--gpu_ids", type=str, default="0,1")
    parser.add_argument("--fitted_params", type=str, default=FITTED_PARAMS)
    parser.add_argument("--output_dir", type=str, default="poisson_mcts/results")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids

    model_path = MODEL_PATHS[args.stage]
    print(f"\n{'='*60}")
    print(f"Poisson-MCTS")
    print(f"Stage: {args.stage}, α={args.alpha}, T={args.bf_temperature}")
    print(f"Model: {model_path}")
    print(f"{'='*60}")

    config = MCTSConfig(
        policy_model_name=model_path,
        tensor_parallel_size=args.tensor_parallel_size,
        temperature=0.7,
        top_p=0.95,
        max_tokens_per_node=256,
        max_depth=args.max_depth,
        num_rollouts=args.num_rollouts,
        num_children=8,  # not used directly, but kept for compat
        exploration_constant=args.exploration_constant,
        prm_type="logprob",
        system_prompt="Please reason step by step, and put your final answer within \\boxed{}.",
    )

    # Poisson-MCTS specific params
    config.pm_alpha = args.alpha
    config.pm_bf_temperature = args.bf_temperature
    config.pm_fitted_params_path = args.fitted_params
    config.pm_training_stage = args.stage
    config.pm_use_negbin_d0 = True

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

    engine = PoissonMCTSEngine(config)

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

        stats = result["tree_stats"]
        print(f"  Correct: {result['correct']}, "
              f"Nodes: {stats['total_nodes']}, "
              f"Terminals: {stats['terminal_nodes']}, "
              f"KL: {stats['structural_kl']:.4f}, "
              f"Time: {elapsed:.1f}s")
        bp = stats.get("branching_profile", {})
        tp = stats.get("target_profile", {})
        for d in sorted(set(list(bp.keys()) + list(tp.keys())))[:5]:
            a = bp.get(d, 0)
            t = tp.get(d, 0)
            print(f"    D{d}: actual={a:.1f}, target={t:.1f}")

        all_results.append(result)

    total_time = time.time() - t0
    correct_count = sum(1 for r in all_results if r.get("correct"))
    accuracy = correct_count / len(all_results) if all_results else 0

    print(f"\n{'='*60}")
    print(f"RESULTS: Poisson-MCTS {args.stage} (α={args.alpha}, T={args.bf_temperature})")
    print(f"{'='*60}")
    print(f"Accuracy: {correct_count}/{len(all_results)} = {accuracy:.1%}")
    print(f"Total time: {total_time:.1f}s")
    avg_nodes = sum(r["tree_stats"]["total_nodes"] for r in all_results) / len(all_results)
    avg_kl = sum(r["tree_stats"]["structural_kl"] for r in all_results) / len(all_results)
    print(f"Avg nodes/tree: {avg_nodes:.1f}")
    print(f"Avg structural KL: {avg_kl:.4f}")

    # Save
    tag = f"a{args.alpha}_t{args.bf_temperature}"
    output_path = os.path.join(
        args.output_dir,
        f"poisson_{args.stage}_{tag}_p{args.start_idx}-{args.start_idx+args.num_problems-1}.json",
    )
    save_data = {
        "config": {
            "method": "poisson_mcts",
            "stage": args.stage,
            "model": model_path,
            "alpha": args.alpha,
            "bf_temperature": args.bf_temperature,
            "exploration_constant": args.exploration_constant,
            "num_rollouts": args.num_rollouts,
            "max_depth": args.max_depth,
        },
        "summary": {
            "accuracy": accuracy,
            "correct": correct_count,
            "total": len(all_results),
            "avg_nodes": avg_nodes,
            "avg_structural_kl": avg_kl,
            "total_time": total_time,
        },
        "results": all_results,
    }
    with open(output_path, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()

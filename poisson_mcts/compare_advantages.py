#!/usr/bin/env python3
"""Compare GRPO advantage distributions: Flat Rollout vs BFS Tree vs Poisson-MCTS.

For each problem, all three methods produce 128 complete trajectories.
We compute GRPO advantages and compare signal strength, accuracy, and token cost.

Note: Flat rollout data exists for problems 0-399 (from rollouts_step_X.json).
      BFS and Poisson-MCTS generate fresh trajectories using the policy model.

Usage:
    python poisson_mcts/compare_advantages.py \
        --stage step_0 \
        --problem_indices 0,1,2,3,4 \
        --gpu_ids 0,1
"""

import argparse
import json
import os
import random
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "mcts_inference"))

from config import MCTSConfig
from bfs_engine import BFSEngine, load_branching_factors
from poisson_mcts_engine import PoissonMCTSEngine
from deepsearch_engine import DeepSearchEngine
from policy_model import PolicyModel
from reward_model import build_reward_model
from utils import is_correct, extract_answer


MODEL_PATHS = {
    "step_0": "/mnt/weka/home/jianshu.she/copus/verl/models/Qwen2.5-Math-7B",
    "step_40": "/mnt/weka/home/jianshu.she/copus/verl/models/global_step_40_hf",
    "step_80": "/mnt/weka/home/jianshu.she/copus/verl/models/global_step_80_hf",
    "step_120": "/mnt/weka/home/jianshu.she/copus/verl/models/global_step_120_hf",
}

ROLLOUT_PATHS = {
    "step_0": "faithful_baseline/results/math500_full/train/rollouts_step_0.json",
    "step_40": "faithful_baseline/results/math500_full/train/rollouts_step_40.json",
    "step_80": "faithful_baseline/results/math500_full/train/rollouts_step_80.json",
    "step_120": "faithful_baseline/results/math500_full/train/rollouts_step_120.json",
}

FITTED_PARAMS = "faithful_baseline/results/math500_full/train/poisson_beta_analysis/fitted_parameters.json"
DATA_PATH = "data-prepare/data/MATH500_train.json"


# ------------------------------------------------------------------
# GRPO advantage computation (matches verl implementation)
# ------------------------------------------------------------------

def grpo_advantage(rewards, epsilon=1e-6):
    """Compute GRPO advantages for a group of rewards.

    Returns: (advantages, mean_reward, std_reward)
    """
    rewards = np.array(rewards, dtype=float)
    mean_r = float(rewards.mean())
    std_r = float(rewards.std())
    advantages = ((rewards - mean_r) / (std_r + epsilon)).tolist()
    return advantages, mean_r, std_r


# ------------------------------------------------------------------
# Method 1: Flat Rollout (from existing data)
# ------------------------------------------------------------------

def load_flat_rollout_data(rollout_path, problem_indices):
    """Load flat rollout data for specific problem indices.

    Returns: {problem_index: {"rollouts": [...], "ground_truth": str, "prompt": str}}
    """
    print(f"Loading flat rollout data from {rollout_path} ...")
    with open(rollout_path) as f:
        data = json.load(f)

    index_set = set(problem_indices)
    result = {}
    for entry in data["results"]:
        idx = entry["problem_index"]
        if idx in index_set:
            result[idx] = entry
    return result


def eval_flat_rollout(flat_entry, target=128):
    """Evaluate flat rollout trajectories, compute rewards and advantages."""
    gt = flat_entry["ground_truth"]
    rollouts = flat_entry["rollouts"][:target]

    rewards = []
    total_tokens = 0
    for r in rollouts:
        text = r.get("response", r.get("full_text", ""))
        correct = is_correct(text, gt)
        rewards.append(1.0 if correct else 0.0)
        total_tokens += r.get("num_tokens", len(text.split()) * 2)  # fallback estimate

    advantages, mean_r, std_r = grpo_advantage(rewards)

    return {
        "rewards": rewards,
        "advantages": advantages,
        "accuracy": float(np.mean(rewards)),
        "adv_std": float(np.std(advantages)),
        "adv_mean": float(np.mean(advantages)),
        "total_tokens": total_tokens,
        "tokens_per_trajectory": total_tokens / len(rollouts),
        "equivalent_flat_cost": total_tokens,  # no sharing
        "token_savings": 0.0,
        "num_trajectories": len(rollouts),
    }


# ------------------------------------------------------------------
# Method 2: BFS Tree
# ------------------------------------------------------------------

def eval_bfs_tree(bfs_engine, question, ground_truth, branching_factors, target=128):
    """Run BFS tree, evaluate trajectories."""
    result = bfs_engine.solve(
        question=question,
        branching_factors=branching_factors,
        target_terminals=target,
        ground_truth=ground_truth,
    )

    rewards = [1.0 if s.get("correct") else 0.0 for s in result["all_solutions"]]
    advantages, mean_r, std_r = grpo_advantage(rewards)

    total_tokens = result["tree_stats"]["total_tokens_generated"]
    n_traj = len(result["all_solutions"])
    # Equivalent flat cost: sum of path tokens for all trajectories
    equiv_flat = sum(s["path_tokens"] for s in result["all_solutions"])

    return {
        "rewards": rewards,
        "advantages": advantages,
        "accuracy": float(np.mean(rewards)) if rewards else 0.0,
        "adv_std": float(np.std(advantages)) if advantages else 0.0,
        "adv_mean": float(np.mean(advantages)) if advantages else 0.0,
        "total_tokens": total_tokens,
        "tokens_per_trajectory": total_tokens / max(n_traj, 1),
        "equivalent_flat_cost": equiv_flat,
        "token_savings": 1.0 - total_tokens / max(equiv_flat, 1),
        "num_trajectories": n_traj,
        "tree_stats": result["tree_stats"],
        "elapsed": result["elapsed_seconds"],
    }


# ------------------------------------------------------------------
# Method 3: Poisson-MCTS
# ------------------------------------------------------------------

def eval_poisson_mcts(poisson_engine, question, ground_truth, target=128, max_rollouts=512):
    """Run Poisson-MCTS to target terminals, evaluate trajectories."""
    t0 = time.time()
    tree = poisson_engine.solve_to_target(
        question=question,
        target_terminals=target,
        max_rollouts=max_rollouts,
    )
    elapsed = time.time() - t0

    terminals = tree.all_terminal_nodes()
    if len(terminals) > target:
        terminals = random.sample(terminals, target)

    rewards = []
    solutions = []
    total_path_tokens = 0
    for t_node in terminals:
        full_text = t_node.get_full_reasoning()
        correct = is_correct(full_text, ground_truth)
        rewards.append(1.0 if correct else 0.0)
        path_tokens = t_node.get_total_tokens()
        total_path_tokens += path_tokens
        solutions.append({
            "answer": extract_answer(full_text),
            "correct": correct,
            "depth": t_node.depth,
            "path_tokens": path_tokens,
        })

    advantages, mean_r, std_r = grpo_advantage(rewards)

    # Total tree tokens (actual GPU cost)
    total_tokens = sum(
        node.token_count
        for node in _all_nodes(tree.root)
        if node.text  # skip root
    )

    n_traj = len(terminals)

    return {
        "rewards": rewards,
        "advantages": advantages,
        "accuracy": float(np.mean(rewards)) if rewards else 0.0,
        "adv_std": float(np.std(advantages)) if advantages else 0.0,
        "adv_mean": float(np.mean(advantages)) if advantages else 0.0,
        "total_tokens": total_tokens,
        "tokens_per_trajectory": total_tokens / max(n_traj, 1),
        "equivalent_flat_cost": total_path_tokens,
        "token_savings": 1.0 - total_tokens / max(total_path_tokens, 1),
        "num_trajectories": n_traj,
        "tree_stats": tree.stats(),
        "rollouts_used": tree.root.visit_count,
        "elapsed": round(elapsed, 2),
    }


def _all_nodes(node):
    """Recursively collect all nodes in tree."""
    yield node
    for child in node.children:
        yield from _all_nodes(child)


# ------------------------------------------------------------------
# Method 4: DeepSearch MCTS (arxiv 2509.25454)
# ------------------------------------------------------------------

def eval_deepsearch(deepsearch_engine, question, ground_truth, target=128, max_rollouts=512):
    """Run DeepSearch MCTS to target terminals, evaluate trajectories."""
    t0 = time.time()
    tree = deepsearch_engine.solve_to_target(
        question=question,
        target_terminals=target,
        max_rollouts=max_rollouts,
    )
    elapsed = time.time() - t0

    terminals = tree.all_terminal_nodes()
    if len(terminals) > target:
        terminals = random.sample(terminals, target)

    rewards = []
    total_path_tokens = 0
    for t_node in terminals:
        full_text = t_node.get_full_reasoning()
        correct = is_correct(full_text, ground_truth)
        rewards.append(1.0 if correct else 0.0)
        total_path_tokens += t_node.get_total_tokens()

    advantages, _, _ = grpo_advantage(rewards)

    total_tokens = sum(
        node.token_count for node in _all_nodes(tree.root) if node.text
    )
    n_traj = len(terminals)

    return {
        "rewards": rewards,
        "advantages": advantages,
        "accuracy": float(np.mean(rewards)) if rewards else 0.0,
        "adv_std": float(np.std(advantages)) if advantages else 0.0,
        "adv_mean": float(np.mean(advantages)) if advantages else 0.0,
        "total_tokens": total_tokens,
        "tokens_per_trajectory": total_tokens / max(n_traj, 1),
        "equivalent_flat_cost": total_path_tokens,
        "token_savings": 1.0 - total_tokens / max(total_path_tokens, 1),
        "num_trajectories": n_traj,
        "tree_stats": tree.stats(),
        "rollouts_used": tree.root.visit_count,
        "elapsed": round(elapsed, 2),
    }


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="GRPO Advantage Comparison")
    parser.add_argument("--stage", type=str, default="step_0",
                        choices=list(MODEL_PATHS.keys()))
    parser.add_argument("--problem_indices", type=str, default="0,1",
                        help="Comma-separated problem indices (must be in 0-399)")
    parser.add_argument("--target_terminals", type=int, default=128)
    parser.add_argument("--max_mcts_rollouts", type=int, default=512)
    parser.add_argument("--tensor_parallel_size", type=int, default=2)
    parser.add_argument("--gpu_ids", type=str, default="0,1")
    parser.add_argument("--max_depth", type=int, default=16)
    parser.add_argument("--output_dir", type=str,
                        default="poisson_mcts/results/advantage_comparison")
    # Poisson-MCTS params (use defaults or BO-optimized values)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--exploration_constant", type=float, default=1.414)
    parser.add_argument("--bf_temperature", type=float, default=1.0)
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
    problem_indices = [int(x) for x in args.problem_indices.split(",")]

    print(f"{'='*60}")
    print(f"GRPO Advantage Comparison")
    print(f"Stage: {args.stage}")
    print(f"Problems: {problem_indices}")
    print(f"Target terminals: {args.target_terminals}")
    print(f"GPUs: {args.gpu_ids}")
    print(f"{'='*60}\n")

    # ---- Load flat rollout data (no GPU needed) ----
    flat_data = load_flat_rollout_data(ROLLOUT_PATHS[args.stage], problem_indices)
    print(f"Loaded flat data for {len(flat_data)} problems.\n")

    # ---- Load problems for question text ----
    with open(DATA_PATH) as f:
        all_problems = json.load(f)

    # ---- Load branching factors ----
    branching_factors = load_branching_factors(FITTED_PARAMS, args.stage)
    print(f"Branching factors: { {d: round(v, 2) for d, v in sorted(branching_factors.items()) if d <= 6} }\n")

    # ---- Initialize model (once) ----
    config = MCTSConfig(
        policy_model_name=MODEL_PATHS[args.stage],
        tensor_parallel_size=args.tensor_parallel_size,
        temperature=0.7,
        top_p=0.95,
        max_tokens_per_node=256,
        max_depth=args.max_depth,
        num_rollouts=args.max_mcts_rollouts,
        num_children=8,
        exploration_constant=args.exploration_constant,
        prm_type="logprob",
        system_prompt="Please reason step by step, and put your final answer within \\boxed{}.",
    )

    print("Loading policy model (one-time) ...")
    policy_model = PolicyModel(config)
    reward_model = build_reward_model(config)
    print("Model ready.\n")

    # ---- Initialize engines ----
    bfs_engine = BFSEngine(
        policy_model=policy_model,
        max_tokens_per_node=256,
        max_depth=args.max_depth,
    )

    # Poisson-MCTS config
    config.pm_alpha = args.alpha
    config.pm_bf_temperature = args.bf_temperature
    config.pm_fitted_params_path = FITTED_PARAMS
    config.pm_training_stage = args.stage
    config.pm_use_negbin_d0 = True
    poisson_engine = PoissonMCTSEngine(
        config, policy_model=policy_model, reward_model=reward_model,
    )

    # DeepSearch MCTS (arxiv 2509.25454) — baseline from literature
    deepsearch_engine = DeepSearchEngine(
        config, policy_model=policy_model, reward_model=reward_model,
    )

    # ---- Run comparison ----
    all_results = []

    for idx in problem_indices:
        print(f"\n{'='*60}")
        print(f"Problem {idx}")
        print(f"{'='*60}")

        if idx not in flat_data:
            print(f"  SKIP: no flat rollout data for problem {idx}")
            continue

        question = flat_data[idx]["prompt"]
        ground_truth = flat_data[idx]["ground_truth"]
        print(f"  GT: {ground_truth}")

        # 1. Flat rollout
        print(f"\n  [Flat Rollout] evaluating existing data ...")
        flat_result = eval_flat_rollout(flat_data[idx], args.target_terminals)
        print(f"    accuracy={flat_result['accuracy']:.0%}  "
              f"adv_std={flat_result['adv_std']:.4f}  "
              f"tokens={flat_result['total_tokens']:,}")

        # 2. BFS Tree
        print(f"\n  [BFS Tree] generating ...")
        bfs_result = eval_bfs_tree(
            bfs_engine, question, ground_truth,
            branching_factors, args.target_terminals,
        )
        print(f"    accuracy={bfs_result['accuracy']:.0%}  "
              f"adv_std={bfs_result['adv_std']:.4f}  "
              f"tokens={bfs_result['total_tokens']:,}  "
              f"savings={bfs_result['token_savings']:.1%}  "
              f"({bfs_result['elapsed']:.1f}s)")

        # 3. Poisson-MCTS
        print(f"\n  [Poisson-MCTS] generating ...")
        mcts_result = eval_poisson_mcts(
            poisson_engine, question, ground_truth,
            args.target_terminals, args.max_mcts_rollouts,
        )
        print(f"    accuracy={mcts_result['accuracy']:.0%}  "
              f"adv_std={mcts_result['adv_std']:.4f}  "
              f"tokens={mcts_result['total_tokens']:,}  "
              f"savings={mcts_result['token_savings']:.1%}  "
              f"rollouts={mcts_result['rollouts_used']}  "
              f"({mcts_result['elapsed']:.1f}s)")

        # 4. DeepSearch MCTS (arxiv 2509.25454)
        print(f"\n  [DeepSearch] generating ...")
        ds_result = eval_deepsearch(
            deepsearch_engine, question, ground_truth,
            args.target_terminals, args.max_mcts_rollouts,
        )
        print(f"    accuracy={ds_result['accuracy']:.0%}  "
              f"adv_std={ds_result['adv_std']:.4f}  "
              f"tokens={ds_result['total_tokens']:,}  "
              f"savings={ds_result['token_savings']:.1%}  "
              f"rollouts={ds_result['rollouts_used']}  "
              f"({ds_result['elapsed']:.1f}s)")

        problem_result = {
            "problem_index": idx,
            "ground_truth": ground_truth,
            "flat": {k: v for k, v in flat_result.items() if k not in ("rewards", "advantages")},
            "bfs_tree": {k: v for k, v in bfs_result.items() if k not in ("rewards", "advantages")},
            "poisson_mcts": {k: v for k, v in mcts_result.items() if k not in ("rewards", "advantages")},
            "deepsearch": {k: v for k, v in ds_result.items() if k not in ("rewards", "advantages")},
            # Store full advantage arrays for distribution analysis
            "flat_advantages": flat_result["advantages"],
            "bfs_advantages": bfs_result["advantages"],
            "mcts_advantages": mcts_result["advantages"],
            "deepsearch_advantages": ds_result["advantages"],
        }
        all_results.append(problem_result)

    # ---- Aggregate and save ----
    os.makedirs(args.output_dir, exist_ok=True)

    # Summary
    print(f"\n\n{'='*60}")
    print(f"SUMMARY — {args.stage}")
    print(f"{'='*60}")
    print(f"{'Method':<16} {'Accuracy':>10} {'Adv.Std':>10} {'Tokens':>12} {'Savings':>10}")
    print(f"{'-'*58}")

    for method in ["flat", "bfs_tree", "poisson_mcts", "deepsearch"]:
        accs = [r[method]["accuracy"] for r in all_results]
        stds = [r[method]["adv_std"] for r in all_results]
        toks = [r[method]["total_tokens"] for r in all_results]
        savs = [r[method].get("token_savings", 0) for r in all_results]
        print(f"{method:<16} {np.mean(accs):>9.1%} {np.mean(stds):>10.4f} "
              f"{int(np.mean(toks)):>12,} {np.mean(savs):>9.1%}")

    # Save results
    output_path = os.path.join(args.output_dir, f"comparison_{args.stage}.json")
    save_data = {
        "config": {
            "stage": args.stage,
            "problem_indices": problem_indices,
            "target_terminals": args.target_terminals,
            "alpha": args.alpha,
            "exploration_constant": args.exploration_constant,
            "bf_temperature": args.bf_temperature,
        },
        "results": all_results,
    }
    with open(output_path, "w") as f:
        json.dump(save_data, f, indent=2, default=str)
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()

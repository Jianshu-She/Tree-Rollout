#!/usr/bin/env python3
"""Compute-matched analysis of the 4-method comparison (ROADMAP item #10).

Reframes the paper's core claim from "same n=128" to "same compute budget T":
we argue that at any fixed compute T, the *faithful* methods (BFS, NegBin)
produce the best GRPO training signal while DeepSearch lags behind.

Method:
  1. Read the 100-problem 4-method comparison JSON (post-hoc, no new GPU).
  2. Recover per-trajectory binary rewards from the stored advantage arrays
     (GRPO advantages have exactly two unique values, one per reward class).
  3. For a sweep of compute budgets T, subsample each method uniformly to
     approximately T tokens (assuming tokens_per_trajectory is flat), then
     recompute GRPO advantages on the subset.
  4. Report per-budget (accuracy, adv_std, no_advantage_rate, n_trajectories)
     for each method and plot three curves.

Rationale for uniform subsampling: we don't have per-trajectory token counts
stored in the JSON, only aggregates. Tokens_per_trajectory is already an
average. Uniform subsampling at a fraction T / total_tokens is a reasonable
first-order approximation; the resulting N is an unbiased estimate of what
the method would produce under a hard compute cap at T.
"""

import json
import os
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


METHODS = [
    ("flat",         "Flat Rollout",  "#4C72B0", "Flat"),
    ("bfs_tree",     "BFS Tree",      "#DD8452", "BFS"),
    ("poisson_mcts", "NegBin MCTS",   "#55A868", "NegBin"),
    ("deepsearch",   "DeepSearch",    "#C44E52", "DeepSearch"),
]

ADV_KEYS = {
    "flat": "flat_advantages",
    "bfs_tree": "bfs_advantages",
    "poisson_mcts": "mcts_advantages",
    "deepsearch": "deepsearch_advantages",
}


def recover_rewards(advantages, accuracy):
    """Recover binary rewards from GRPO advantage array + stored accuracy.

    GRPO advantages are (r - mean_r) / (std_r + eps). With binary rewards:
    - Two unique values => positive entries correspond to r=1, negative to r=0
    - One unique value (zero) => all rewards are equal. The stored accuracy
      tells us which case: acc > 0.5 means all-correct, else all-wrong.
    """
    adv = np.asarray(advantages, dtype=float)
    if len(adv) == 0:
        return np.array([], dtype=float)
    unique = np.unique(adv)
    if len(unique) == 1:
        # degenerate — all rewards equal; infer from stored accuracy
        return np.ones_like(adv) if accuracy > 0.5 else np.zeros_like(adv)
    return (adv > 0).astype(float)


def grpo_advantage(rewards, eps=1e-6):
    r = np.asarray(rewards, dtype=float)
    if len(r) == 0:
        return np.array([]), 0.0, 0.0
    mean = r.mean()
    std = r.std()
    return (r - mean) / (std + eps), mean, std


def subsample_method(rewards, total_tokens_full, n_full, budget, rng):
    """Given a method's full (rewards, total_tokens, n_full) on one problem,
    return a subsampled (rewards, n_sub, tokens_used) within the budget.

    Uniform tokens_per_trajectory assumption: n_sub ≈ n_full * (T / total).
    """
    if len(rewards) == 0:
        return rewards, 0, 0.0
    if total_tokens_full <= budget or n_full == 0:
        return rewards, len(rewards), total_tokens_full

    tokens_per_traj = total_tokens_full / max(n_full, 1)
    n_sub = max(1, int(budget / tokens_per_traj))
    n_sub = min(n_sub, len(rewards))
    idx = rng.choice(len(rewards), size=n_sub, replace=False)
    return rewards[idx], n_sub, n_sub * tokens_per_traj


def per_problem_stats(rewards):
    """Given rewards for one (problem, method) subset, return metrics."""
    advs, mean_r, std_r = grpo_advantage(rewards)
    n = len(rewards)
    if n == 0:
        return dict(accuracy=0.0, adv_std=0.0, n=0,
                    all_correct=False, all_wrong=False, no_advantage=True)
    acc = float(mean_r)
    adv_std = float(advs.std())
    no_adv = adv_std < 0.01
    return dict(
        accuracy=acc,
        adv_std=adv_std,
        n=n,
        all_correct=no_adv and acc > 0.99,
        all_wrong=no_adv and acc < 0.01,
        no_advantage=bool(no_adv),
    )


def analyze_budget(results, budget, n_bootstraps, seed):
    """Compute per-method aggregate stats at a given compute budget."""
    rng = np.random.default_rng(seed)
    boot_stats = {m[0]: defaultdict(list) for m in METHODS}

    for _ in range(n_bootstraps):
        per_method = {m[0]: [] for m in METHODS}
        for r in results:
            for key, *_ in METHODS:
                rewards_full = recover_rewards(r[ADV_KEYS[key]], r[key]["accuracy"])
                total = r[key]["total_tokens"]
                n_full = r[key]["num_trajectories"]
                sub_rew, n_sub, _ = subsample_method(
                    rewards_full, total, n_full, budget, rng)
                per_method[key].append(per_problem_stats(sub_rew))

        for key, stats in per_method.items():
            accs = np.array([s["accuracy"] for s in stats])
            adv_stds = np.array([s["adv_std"] for s in stats])
            ns = np.array([s["n"] for s in stats])
            all_correct = sum(s["all_correct"] for s in stats)
            all_wrong = sum(s["all_wrong"] for s in stats)
            no_adv = sum(s["no_advantage"] for s in stats)
            boot_stats[key]["mean_accuracy"].append(accs.mean())
            boot_stats[key]["mean_adv_std"].append(adv_stds.mean())
            boot_stats[key]["mean_n"].append(ns.mean())
            boot_stats[key]["all_correct"].append(all_correct)
            boot_stats[key]["all_wrong"].append(all_wrong)
            boot_stats[key]["no_advantage"].append(no_adv)

    summary = {}
    for key, d in boot_stats.items():
        summary[key] = {}
        for metric, vals in d.items():
            arr = np.asarray(vals, dtype=float)
            summary[key][metric] = dict(
                mean=float(arr.mean()),
                std=float(arr.std()),
            )
    return summary


def run_sweep(results, budgets, n_bootstraps=10, seed=0):
    return {int(b): analyze_budget(results, b, n_bootstraps, seed + int(b) % 1000)
            for b in budgets}


def plot_sweep(sweep, out_path):
    budgets = sorted(sweep.keys())
    budgets_k = np.array(budgets) / 1000.0

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    def extract(metric, key):
        return np.array([sweep[b][key][metric]["mean"] for b in budgets]), \
               np.array([sweep[b][key][metric]["std"] for b in budgets])

    # Panel 1: mean accuracy vs budget
    ax = axes[0, 0]
    for key, label, color, short in METHODS:
        mean, std = extract("mean_accuracy", key)
        ax.plot(budgets_k, mean, "o-", color=color, label=label, markersize=5, lw=2)
        ax.fill_between(budgets_k, mean - std, mean + std,
                        color=color, alpha=0.12, linewidth=0)
    ax.set_xscale("log")
    ax.set_xlabel("Compute Budget per Problem (K tokens)")
    ax.set_ylabel("Mean Accuracy")
    ax.set_title("Accuracy vs Compute Budget")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, loc="lower right")

    # Panel 2: adv_std vs budget
    ax = axes[0, 1]
    for key, label, color, short in METHODS:
        mean, std = extract("mean_adv_std", key)
        ax.plot(budgets_k, mean, "o-", color=color, label=label, markersize=5, lw=2)
        ax.fill_between(budgets_k, mean - std, mean + std,
                        color=color, alpha=0.12, linewidth=0)
    ax.set_xscale("log")
    ax.set_xlabel("Compute Budget per Problem (K tokens)")
    ax.set_ylabel("Mean Advantage Std (RL signal strength)")
    ax.set_title("GRPO Signal Strength vs Compute Budget")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)

    # Panel 3: no-advantage rate vs budget
    ax = axes[1, 0]
    for key, label, color, short in METHODS:
        mean, std = extract("no_advantage", key)
        ax.plot(budgets_k, mean, "o-", color=color, label=label, markersize=5, lw=2)
        ax.fill_between(budgets_k, mean - std, mean + std,
                        color=color, alpha=0.12, linewidth=0)
    ax.set_xscale("log")
    ax.set_xlabel("Compute Budget per Problem (K tokens)")
    ax.set_ylabel("# No-Advantage Problems (out of 100)")
    ax.set_title("No-Advantage Rate vs Compute Budget")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)

    # Panel 4: n_trajectories vs budget
    ax = axes[1, 1]
    for key, label, color, short in METHODS:
        mean, std = extract("mean_n", key)
        ax.plot(budgets_k, mean, "o-", color=color, label=label, markersize=5, lw=2)
    ax.set_xscale("log")
    ax.set_xlabel("Compute Budget per Problem (K tokens)")
    ax.set_ylabel("Mean #Trajectories")
    ax.set_title("Effective Trajectory Count vs Compute Budget")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)

    fig.suptitle(
        "Compute-Matched Comparison: Flat / BFS / NegBin / DeepSearch\n"
        "Post-hoc subsampling of 100-problem step_0 data to matched budgets",
        fontsize=14, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def print_summary_table(sweep, out_path):
    budgets = sorted(sweep.keys())
    lines = []
    lines.append("=" * 100)
    lines.append("COMPUTE-MATCHED SUMMARY (100 problems, step_0)")
    lines.append("=" * 100)
    header = f"{'Budget (K)':>12} | " + " | ".join(
        f"{m[3]:>21}" for m in METHODS)
    lines.append(header)
    lines.append("-" * len(header))

    for b in budgets:
        row = [f"{b/1000:>12.0f}"]
        for key, *_ in METHODS:
            acc = sweep[b][key]["mean_accuracy"]["mean"]
            adv = sweep[b][key]["mean_adv_std"]["mean"]
            n = sweep[b][key]["mean_n"]["mean"]
            no_adv = sweep[b][key]["no_advantage"]["mean"]
            row.append(f"a={acc:.1%} σ={adv:.2f} n={n:4.0f} nA={no_adv:2.0f}")
        lines.append(" | ".join(row))

    lines.append("-" * len(header))
    lines.append("a=accuracy  σ=adv_std  n=mean trajectories  nA=no-advantage count")
    out = "\n".join(lines)
    print(out)
    with open(out_path, "w") as f:
        f.write(out + "\n")
    print(f"\nSaved {out_path}")


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else \
        "poisson_mcts/results/advantage_comparison/step_0/comparison_step_0.json"
    out_dir = os.path.dirname(path)

    with open(path) as f:
        data = json.load(f)
    results = data["results"]
    print(f"Loaded {len(results)} problems from {path}")

    # Budget sweep: log scale from 5K to 250K per problem
    budgets = [5000, 10000, 15000, 22000, 30000, 45000, 65000,
               100000, 150000, 220000]

    print(f"Running sweep over {len(budgets)} budgets with 10 bootstraps each...")
    sweep = run_sweep(results, budgets, n_bootstraps=10, seed=42)

    summary_path = os.path.join(out_dir, "compute_matched_summary.txt")
    plot_path = os.path.join(out_dir, "compute_matched_analysis.png")
    json_path = os.path.join(out_dir, "compute_matched_sweep.json")

    print_summary_table(sweep, summary_path)
    plot_sweep(sweep, plot_path)

    with open(json_path, "w") as f:
        json.dump({str(b): v for b, v in sweep.items()}, f, indent=2)
    print(f"Saved {json_path}")


if __name__ == "__main__":
    main()

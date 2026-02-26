#!/usr/bin/env python3
"""Analyze pairwise overlap among flat RL rollouts to motivate tree-based (MCTS) rollouts.

Computes common-prefix ratio, LCS ratio, and divergence point for every pair of
rollouts within each problem, then produces summary statistics and five plots.
"""

import argparse
import json
import os
from difflib import SequenceMatcher
from itertools import combinations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# Core metrics
# ---------------------------------------------------------------------------

def tokenize(text: str) -> list[str]:
    """Split text into word-level tokens (whitespace split)."""
    return text.split()


def common_prefix_length(a: list[str], b: list[str]) -> int:
    """Return the length of the longest common prefix between two token lists."""
    length = min(len(a), len(b))
    for i in range(length):
        if a[i] != b[i]:
            return i
    return length


def lcs_ratio(a: list[str], b: list[str]) -> float:
    """Return LCS length / mean length of the two sequences.

    Uses difflib.SequenceMatcher which computes a ratio based on the longest
    common subsequence.  ratio() = 2 * M / T where M is matching chars and T is
    total elements; we convert to LCS_len / mean_len.
    """
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    sm = SequenceMatcher(None, a, b, autojunk=False)
    # SequenceMatcher.ratio() = 2*M / (len(a)+len(b))
    # M = number of matching elements, which equals LCS length for our purposes.
    # LCS_len / mean_len = M / ((len(a)+len(b))/2) = 2*M / (len(a)+len(b)) = ratio()
    return sm.ratio()


def divergence_point(a: list[str], b: list[str]) -> float:
    """Return the fractional position where the common prefix ends.

    Result is in [0, 1]: prefix_length / mean_length.
    """
    mean_len = (len(a) + len(b)) / 2.0
    if mean_len == 0:
        return 1.0
    return common_prefix_length(a, b) / mean_len


# ---------------------------------------------------------------------------
# Per-problem analysis
# ---------------------------------------------------------------------------

def analyze_problem(rollouts: list[str]) -> dict:
    """Compute pairwise overlap metrics for all rollout pairs within one problem.

    Returns a dict with lists of per-pair values and aggregated statistics.
    """
    tokens_list = [tokenize(r) for r in rollouts]
    n = len(tokens_list)

    prefix_ratios = []
    lcs_ratios = []
    divergence_points = []
    total_tokens = sum(len(t) for t in tokens_list)

    for i, j in combinations(range(n), 2):
        a, b = tokens_list[i], tokens_list[j]
        mean_len = (len(a) + len(b)) / 2.0

        # Common prefix ratio
        cpl = common_prefix_length(a, b)
        pr = cpl / mean_len if mean_len > 0 else 1.0
        prefix_ratios.append(pr)

        # LCS ratio
        lr = lcs_ratio(a, b)
        lcs_ratios.append(lr)

        # Divergence point
        dp = cpl / mean_len if mean_len > 0 else 1.0
        divergence_points.append(dp)

    # Shared-prefix token savings estimate:
    # For each pair, the shared prefix is counted once instead of twice.
    # Across all rollouts, the average shared prefix represents tokens
    # that a tree would store once.  Estimate: per rollout, the average
    # prefix overlap with *any* other rollout approximates the reusable fraction.
    avg_prefix_ratio = float(np.mean(prefix_ratios)) if prefix_ratios else 0.0
    mean_rollout_len = float(np.mean([len(t) for t in tokens_list]))
    estimated_shared_tokens = avg_prefix_ratio * mean_rollout_len * n
    estimated_tree_tokens = total_tokens - estimated_shared_tokens

    return {
        "num_rollouts": n,
        "total_tokens": total_tokens,
        "mean_rollout_length": mean_rollout_len,
        "estimated_tree_tokens": max(estimated_tree_tokens, 0),
        "prefix_ratio_mean": float(np.mean(prefix_ratios)) if prefix_ratios else 0.0,
        "prefix_ratio_median": float(np.median(prefix_ratios)) if prefix_ratios else 0.0,
        "lcs_ratio_mean": float(np.mean(lcs_ratios)) if lcs_ratios else 0.0,
        "lcs_ratio_median": float(np.median(lcs_ratios)) if lcs_ratios else 0.0,
        "divergence_mean": float(np.mean(divergence_points)) if divergence_points else 0.0,
        "divergence_median": float(np.median(divergence_points)) if divergence_points else 0.0,
        "all_divergence_points": divergence_points,
    }


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data(path: str, dataset: str, num_problems: int | None = None) -> list[list[str]]:
    """Load rollout texts from a results JSON file.

    Returns a list of problems, where each problem is a list of rollout strings.
    """
    with open(path, "r") as f:
        data = json.load(f)

    if num_problems is not None:
        data = data[:num_problems]

    problems: list[list[str]] = []
    for entry in data:
        if dataset == "math500":
            rollouts = [r["response"] for r in entry["rollout_details"]]
        elif dataset == "gsm8k":
            rollouts = [s["generated_text"] for s in entry["samples"]]
        else:
            raise ValueError(f"Unknown dataset: {dataset}")
        problems.append(rollouts)

    return problems


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_results(stats: list[dict], output_dir: str) -> None:
    """Generate all five plots and save to output_dir."""
    os.makedirs(output_dir, exist_ok=True)

    prefix_means = [s["prefix_ratio_mean"] for s in stats]
    lcs_means = [s["lcs_ratio_mean"] for s in stats]

    # Collect all pairwise divergence points across problems
    all_div_points = []
    for s in stats:
        all_div_points.extend(s["all_divergence_points"])

    # --- Plot 1: Histogram of per-problem mean prefix overlap ratio ---
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(prefix_means, bins=30, edgecolor="black", alpha=0.7, color="#4C72B0")
    ax.set_xlabel("Mean Prefix Overlap Ratio")
    ax.set_ylabel("Number of Problems")
    ax.set_title("Distribution of Per-Problem Mean Prefix Overlap")
    ax.axvline(np.mean(prefix_means), color="red", linestyle="--",
               label=f"Global mean = {np.mean(prefix_means):.3f}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "prefix_overlap_hist.png"), dpi=150)
    plt.close(fig)

    # --- Plot 2: Histogram of per-problem mean LCS ratio ---
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(lcs_means, bins=30, edgecolor="black", alpha=0.7, color="#55A868")
    ax.set_xlabel("Mean LCS Ratio")
    ax.set_ylabel("Number of Problems")
    ax.set_title("Distribution of Per-Problem Mean LCS Ratio")
    ax.axvline(np.mean(lcs_means), color="red", linestyle="--",
               label=f"Global mean = {np.mean(lcs_means):.3f}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "lcs_overlap_hist.png"), dpi=150)
    plt.close(fig)

    # --- Plot 3: Combined violin/box plot ---
    fig, ax = plt.subplots(figsize=(8, 5))
    parts = ax.violinplot([prefix_means, lcs_means], positions=[1, 2],
                          showmeans=True, showmedians=True)
    for pc in parts["bodies"]:
        pc.set_alpha(0.7)
    ax.boxplot([prefix_means, lcs_means], positions=[1, 2], widths=0.15,
               showfliers=False, zorder=3)
    ax.set_xticks([1, 2])
    ax.set_xticklabels(["Prefix Overlap", "LCS Ratio"])
    ax.set_ylabel("Ratio")
    ax.set_title("Prefix Overlap vs LCS Ratio Across Problems")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "overlap_violin.png"), dpi=150)
    plt.close(fig)

    # --- Plot 4: Divergence depth CDF ---
    fig, ax = plt.subplots(figsize=(8, 5))
    sorted_div = np.sort(all_div_points)
    cdf = np.arange(1, len(sorted_div) + 1) / len(sorted_div)
    ax.plot(sorted_div, cdf, linewidth=2, color="#C44E52")
    ax.set_xlabel("Divergence Point (fraction of sequence length)")
    ax.set_ylabel("Cumulative Probability")
    ax.set_title("CDF of Pairwise Divergence Depth")
    ax.axhline(0.5, color="gray", linestyle=":", alpha=0.5)
    median_div = float(np.median(all_div_points))
    ax.axvline(median_div, color="blue", linestyle="--", alpha=0.6,
               label=f"Median = {median_div:.3f}")
    ax.legend()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "divergence_cdf.png"), dpi=150)
    plt.close(fig)

    # --- Plot 5: Token savings estimate ---
    total_flat = sum(s["total_tokens"] for s in stats)
    total_tree = sum(s["estimated_tree_tokens"] for s in stats)
    savings_pct = (1 - total_tree / total_flat) * 100 if total_flat > 0 else 0

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(["Flat Rollouts", "Tree (est.)"], [total_flat, total_tree],
                  color=["#DD8452", "#4C72B0"], edgecolor="black")
    ax.set_ylabel("Total Tokens")
    ax.set_title(f"Token Usage: Flat vs Tree-Based Rollouts\n"
                 f"Estimated savings: {savings_pct:.1f}%")
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height,
                f"{height:,.0f}", ha="center", va="bottom", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "token_savings.png"), dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Analyze pairwise overlap among flat RL rollouts.")
    parser.add_argument("--results", required=True,
                        help="Path to results JSON file")
    parser.add_argument("--dataset", required=True, choices=["math500", "gsm8k"],
                        help="Dataset format (math500 or gsm8k)")
    parser.add_argument("--output_dir", required=True,
                        help="Directory to save plots and summary JSON")
    parser.add_argument("--num_problems", type=int, default=None,
                        help="Limit number of problems to analyze (default: all)")
    args = parser.parse_args()

    print(f"Loading data from {args.results} (dataset={args.dataset})...")
    problems = load_data(args.results, args.dataset, args.num_problems)
    print(f"Loaded {len(problems)} problems")

    print("Analyzing pairwise overlap...")
    stats = []
    for i, rollouts in enumerate(problems):
        if (i + 1) % 50 == 0 or i == 0:
            print(f"  Processing problem {i + 1}/{len(problems)}...")
        s = analyze_problem(rollouts)
        stats.append(s)

    # Global summary
    global_prefix_mean = float(np.mean([s["prefix_ratio_mean"] for s in stats]))
    global_lcs_mean = float(np.mean([s["lcs_ratio_mean"] for s in stats]))
    global_div_mean = float(np.mean([s["divergence_mean"] for s in stats]))
    total_flat_tokens = sum(s["total_tokens"] for s in stats)
    total_tree_tokens = sum(s["estimated_tree_tokens"] for s in stats)
    savings_pct = (1 - total_tree_tokens / total_flat_tokens) * 100 if total_flat_tokens > 0 else 0

    summary = {
        "dataset": args.dataset,
        "num_problems": len(problems),
        "global_prefix_overlap_mean": global_prefix_mean,
        "global_lcs_ratio_mean": global_lcs_mean,
        "global_divergence_mean": global_div_mean,
        "total_flat_tokens": total_flat_tokens,
        "total_estimated_tree_tokens": total_tree_tokens,
        "estimated_savings_pct": savings_pct,
        "per_problem": [
            {k: v for k, v in s.items() if k != "all_divergence_points"}
            for s in stats
        ],
    }

    os.makedirs(args.output_dir, exist_ok=True)
    summary_path = os.path.join(args.output_dir, "overlap_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to {summary_path}")

    print("\nGenerating plots...")
    plot_results(stats, args.output_dir)
    print(f"Plots saved to {args.output_dir}/")

    # Print key results
    print("\n" + "=" * 60)
    print("OVERLAP ANALYSIS RESULTS")
    print("=" * 60)
    print(f"Dataset:                    {args.dataset}")
    print(f"Problems analyzed:          {len(problems)}")
    print(f"Mean prefix overlap ratio:  {global_prefix_mean:.4f}")
    print(f"Mean LCS ratio:             {global_lcs_mean:.4f}")
    print(f"Mean divergence point:      {global_div_mean:.4f}")
    print(f"Total flat tokens:          {total_flat_tokens:,}")
    print(f"Estimated tree tokens:      {total_tree_tokens:,.0f}")
    print(f"Estimated token savings:    {savings_pct:.1f}%")
    print("=" * 60)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Analyze branching factor distributions from syntactic trees.

Computes P(branching_factor | depth, stage) and other statistics
needed for tuning naive MCTS to match post-hoc tree structures.

Outputs:
- Branching factor distributions per depth per stage
- W(d): avg width at depth d
- B(d): avg branching factor at depth d
- S(d): avg number of surviving rollouts at depth d
- P(d): avg purity (fraction of majority class) at depth d
"""

import json
import os
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from mcts_inference.utils import is_correct


def collect_nodes_at_depth(tree, target_depth, current_depth=0):
    """Collect all nodes at a specific depth."""
    if current_depth == target_depth:
        return [tree]
    nodes = []
    for c in tree.get("children", []):
        nodes.extend(collect_nodes_at_depth(c, target_depth, current_depth + 1))
    return nodes


def collect_leaf_rollout_ids(node):
    if not node.get("children"):
        return set(node.get("rollout_ids", []))
    ids = set()
    for c in node["children"]:
        ids |= collect_leaf_rollout_ids(c)
    return ids


def analyze_tree(tree, correctness):
    """Extract per-depth statistics from a single tree."""
    # Find max depth
    def max_depth(node, d=0):
        if not node.get("children"):
            return d
        return max(max_depth(c, d + 1) for c in node["children"])

    md = max_depth(tree)
    depth_stats = {}

    for d in range(md + 1):
        nodes = collect_nodes_at_depth(tree, d)

        # Width = number of nodes at this depth
        width = len(nodes)

        # Branching factors = number of children per node at this depth
        bfs = [len(n.get("children", [])) for n in nodes if n.get("children")]

        # Surviving rollouts = total rollouts across all nodes at this depth
        surviving = sum(n.get("num_rollouts", len(n.get("rollout_ids", []))) for n in nodes)

        # Purity = for each node, fraction of majority class (correct/incorrect)
        purities = []
        for n in nodes:
            rids = n.get("rollout_ids", [])
            if not rids:
                rids = list(collect_leaf_rollout_ids(n))
            if rids:
                n_correct = sum(1 for r in rids if correctness.get(r, False))
                purity = max(n_correct, len(rids) - n_correct) / len(rids)
                purities.append(purity)

        depth_stats[d] = {
            "width": width,
            "branching_factors": bfs,
            "surviving_rollouts": surviving,
            "purities": purities,
        }

    return depth_stats, md


def main():
    tree_base = "faithful_baseline/results/math500_full/train/trees_syntactic"
    rollout_dir = "faithful_baseline/results/math500_full/train"
    output_dir = "faithful_baseline/results/math500_full/train/branching_analysis"
    os.makedirs(output_dir, exist_ok=True)

    stages = ["step_0", "step_40", "step_80", "step_120"]
    max_depth_to_plot = 15  # Focus on first 15 depths

    # Load rollout data for correctness
    stage_correctness = {}
    for stage in stages:
        rollout_path = os.path.join(rollout_dir, f"rollouts_{stage}.json")
        if not os.path.exists(rollout_path):
            continue
        with open(rollout_path) as f:
            data = json.load(f)
        corr = {}
        for entry in data["results"]:
            pid = entry["problem_index"]
            gt = entry.get("ground_truth", entry.get("answer", ""))
            pid_corr = {}
            for r in entry["rollouts"]:
                rid = r["rollout_id"]
                text = r.get("full_text", "") or r.get("response", "")
                pid_corr[rid] = is_correct(text, gt)
            corr[pid] = pid_corr
        stage_correctness[stage] = corr
        print(f"Loaded correctness for {stage}: {len(corr)} problems")

    # Analyze all trees
    all_results = {}
    for stage in stages:
        stage_dir = os.path.join(tree_base, stage)
        if not os.path.exists(stage_dir):
            print(f"Skipping {stage}: no trees")
            continue

        # Aggregate per-depth stats across all problems
        agg = defaultdict(lambda: {
            "widths": [], "branching_factors": [],
            "surviving": [], "purities": [], "max_depths": []
        })

        tree_files = sorted(f for f in os.listdir(stage_dir) if f.startswith("tree_"))
        print(f"\n{stage}: {len(tree_files)} trees")

        for tf in tree_files:
            # Extract problem index
            pid = int(tf.split("_p")[1].split(".")[0])
            with open(os.path.join(stage_dir, tf)) as f:
                tree = json.load(f)

            corr = stage_correctness.get(stage, {}).get(pid, {})
            depth_stats, md = analyze_tree(tree, corr)

            for d, stats in depth_stats.items():
                agg[d]["widths"].append(stats["width"])
                agg[d]["branching_factors"].extend(stats["branching_factors"])
                agg[d]["surviving"].append(stats["surviving_rollouts"])
                agg[d]["purities"].extend(stats["purities"])
            agg[0]["max_depths"].append(md)

        all_results[stage] = dict(agg)

    # =========================================================================
    # Plot 1: W(d), B(d), S(d), P(d) curves for all stages
    # =========================================================================
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    colors = {"step_0": "#e74c3c", "step_40": "#e67e22", "step_80": "#2ecc71", "step_120": "#3498db"}
    labels = {"step_0": "Base (step 0)", "step_40": "Step 40", "step_80": "Step 80", "step_120": "Step 120"}

    for stage in stages:
        if stage not in all_results:
            continue
        agg = all_results[stage]
        depths = sorted(d for d in agg.keys() if d <= max_depth_to_plot)

        def mean_std(vals):
            if not vals:
                return 0.0, 0.0
            return float(np.mean(vals)), float(np.std(vals))

        w_stats = [mean_std(agg[d]["widths"]) for d in depths]
        w_vals = [m for m, _ in w_stats]
        w_err = [s for _, s in w_stats]
        axes[0, 0].plot(depths, w_vals, "o-", color=colors[stage], label=labels[stage], markersize=4)
        axes[0, 0].fill_between(
            depths,
            [max(0, m - s) for m, s in w_stats],
            [m + s for m, s in w_stats],
            color=colors[stage], alpha=0.15, linewidth=0,
        )

        b_stats = [mean_std(agg[d]["branching_factors"]) for d in depths]
        b_vals = [m for m, _ in b_stats]
        axes[0, 1].plot(depths, b_vals, "o-", color=colors[stage], label=labels[stage], markersize=4)
        axes[0, 1].fill_between(
            depths,
            [max(0, m - s) for m, s in b_stats],
            [m + s for m, s in b_stats],
            color=colors[stage], alpha=0.15, linewidth=0,
        )

        s_stats = [mean_std(agg[d]["surviving"]) for d in depths]
        s_vals = [m for m, _ in s_stats]
        axes[1, 0].plot(depths, s_vals, "o-", color=colors[stage], label=labels[stage], markersize=4)
        axes[1, 0].fill_between(
            depths,
            [max(0, m - s) for m, s in s_stats],
            [m + s for m, s in s_stats],
            color=colors[stage], alpha=0.15, linewidth=0,
        )

        p_stats = [mean_std(agg[d]["purities"]) if agg[d]["purities"] else (0.5, 0.0) for d in depths]
        p_vals = [m for m, _ in p_stats]
        axes[1, 1].plot(depths, p_vals, "o-", color=colors[stage], label=labels[stage], markersize=4)
        axes[1, 1].fill_between(
            depths,
            [max(0, m - s) for m, s in p_stats],
            [min(1, m + s) for m, s in p_stats],
            color=colors[stage], alpha=0.15, linewidth=0,
        )

    axes[0, 0].set_title("W(d): Avg Width at Depth d")
    axes[0, 0].set_xlabel("Depth")
    axes[0, 0].set_ylabel("Width (# nodes)")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].set_title("B(d): Avg Branching Factor at Depth d")
    axes[0, 1].set_xlabel("Depth")
    axes[0, 1].set_ylabel("Branching Factor")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].set_title("S(d): Avg Surviving Rollouts at Depth d")
    axes[1, 0].set_xlabel("Depth")
    axes[1, 0].set_ylabel("# Rollouts")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].set_title("P(d): Avg Node Purity at Depth d")
    axes[1, 1].set_xlabel("Depth")
    axes[1, 1].set_ylabel("Purity")
    axes[1, 1].set_ylim(0.4, 1.05)
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    fig.suptitle("Tree Structure Statistics Across Training Stages (400 problems, syntactic clustering)\nLine = mean, shaded band = ±1 std across problems/nodes", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "tree_curves_WBSP.png"), dpi=150, bbox_inches="tight")
    print(f"\nSaved tree_curves_WBSP.png")

    # =========================================================================
    # Plot 2: Branching factor distributions at D1 and D3
    # =========================================================================
    fig, axes = plt.subplots(2, len(stages), figsize=(16, 8))

    for j, stage in enumerate(stages):
        if stage not in all_results:
            continue
        agg = all_results[stage]

        for row, d in enumerate([1, 3]):
            ax = axes[row, j]
            if d in agg and agg[d]["branching_factors"]:
                bfs = agg[d]["branching_factors"]
                max_bf = min(max(bfs), 30)
                bins = np.arange(0.5, max_bf + 1.5, 1)
                ax.hist(bfs, bins=bins, color=colors[stage], alpha=0.7, edgecolor="black", linewidth=0.5)
                ax.set_title(f"{labels[stage]}, D{d}\n(mean={np.mean(bfs):.1f}, median={np.median(bfs):.0f})")
            else:
                ax.set_title(f"{labels[stage]}, D{d}\n(no data)")
            ax.set_xlabel("Branching Factor")
            ax.set_ylabel("Count")
            ax.grid(True, alpha=0.3)

    fig.suptitle("Branching Factor Distributions P(bf | depth, stage)", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "branching_factor_distributions.png"), dpi=150, bbox_inches="tight")
    print(f"Saved branching_factor_distributions.png")

    # =========================================================================
    # Plot 3: Max depth distribution per stage
    # =========================================================================
    fig, axes = plt.subplots(1, len(stages), figsize=(16, 4))

    for j, stage in enumerate(stages):
        if stage not in all_results:
            continue
        agg = all_results[stage]
        max_depths = agg.get(0, {}).get("max_depths", [])
        if max_depths:
            ax = axes[j]
            ax.hist(max_depths, bins=30, color=colors[stage], alpha=0.7, edgecolor="black", linewidth=0.5)
            ax.set_title(f"{labels[stage]}\n(mean={np.mean(max_depths):.1f}, median={np.median(max_depths):.0f})")
            ax.set_xlabel("Max Tree Depth")
            ax.set_ylabel("Count")
            ax.grid(True, alpha=0.3)

    fig.suptitle("Max Tree Depth Distribution Per Stage", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "max_depth_distribution.png"), dpi=150, bbox_inches="tight")
    print(f"Saved max_depth_distribution.png")

    # =========================================================================
    # Save numerical summary
    # =========================================================================
    summary = {}
    for stage in stages:
        if stage not in all_results:
            continue
        agg = all_results[stage]
        stage_summary = {}
        for d in sorted(agg.keys()):
            if d > max_depth_to_plot:
                continue
            entry = {
                "avg_width": float(np.mean(agg[d]["widths"])),
                "avg_surviving": float(np.mean(agg[d]["surviving"])),
            }
            if agg[d]["branching_factors"]:
                bfs = agg[d]["branching_factors"]
                entry["avg_bf"] = float(np.mean(bfs))
                entry["median_bf"] = float(np.median(bfs))
                entry["std_bf"] = float(np.std(bfs))
            if agg[d]["purities"]:
                entry["avg_purity"] = float(np.mean(agg[d]["purities"]))
            stage_summary[str(d)] = entry
        summary[stage] = stage_summary

    with open(os.path.join(output_dir, "branching_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved branching_summary.json")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Compare MCTS trees (k=2,4,8) with flat rollout post-hoc trees.

Collects branching factor distributions at each depth, fits Poisson/NegBin,
and produces comparison plots + summary table.
"""

import json
import os
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats as sp_stats

OUTPUT_DIR = "faithful_baseline/results/mcts_comparison"
FLAT_PARAMS = "faithful_baseline/results/math500_full/train/poisson_beta_analysis/fitted_parameters.json"

STAGES = ["step_0", "step_40", "step_80", "step_120"]
STAGE_LABELS = {"step_0": "Base (step 0)", "step_40": "Step 40",
                "step_80": "Step 80", "step_120": "Step 120"}
K_VALUES = [2, 4, 8]
K_DIRS = {
    2: "faithful_baseline/results/mcts_standard",
    4: "faithful_baseline/results/mcts_standard_k4",
    8: "faithful_baseline/results/mcts_standard_k8",
}
MAX_DEPTH = 10


def collect_tree_bf(tree_dict, depth=0):
    """Collect branching factors at each depth from MCTS tree dict."""
    bf = defaultdict(list)
    children = tree_dict.get("children", [])
    if children:
        bf[depth].append(len(children))
    for c in children:
        child_bf = collect_tree_bf(c, depth + 1)
        for d, vals in child_bf.items():
            bf[d].extend(vals)
    return bf


def collect_tree_q(tree_dict, depth=0):
    """Collect q-values at each depth."""
    q = defaultdict(list)
    qv = tree_dict.get("q_value", None)
    if qv is not None:
        q[depth].append(qv)
    for c in tree_dict.get("children", []):
        child_q = collect_tree_q(c, depth + 1)
        for d, vals in child_q.items():
            q[d].extend(vals)
    return q


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load flat rollout parameters
    with open(FLAT_PARAMS) as f:
        flat_params = json.load(f)

    # Collect MCTS data for all k values
    all_data = {}  # (k, stage) -> {bf_by_depth, q_by_depth, accuracy, avg_nodes}

    for k in K_VALUES:
        kdir = K_DIRS[k]
        for stage in STAGES:
            files = [f for f in os.listdir(kdir) if f.startswith(f"mcts_{stage}_p") and f.endswith(".json")]
            if not files:
                continue

            bf_agg = defaultdict(list)
            q_agg = defaultdict(list)
            total_correct = 0
            total_problems = 0
            total_nodes = 0

            for fname in files:
                with open(os.path.join(kdir, fname)) as f:
                    data = json.load(f)
                for result in data["results"]:
                    tree = result.get("tree", {})
                    bf_d = collect_tree_bf(tree)
                    q_d = collect_tree_q(tree)
                    for d, vals in bf_d.items():
                        if d <= MAX_DEPTH:
                            bf_agg[d].extend(vals)
                    for d, vals in q_d.items():
                        if d <= MAX_DEPTH:
                            q_agg[d].extend(vals)
                    if result.get("correct"):
                        total_correct += 1
                    total_problems += 1
                    total_nodes += result.get("tree_stats", {}).get("total_nodes", 0)

            all_data[(k, stage)] = {
                "bf": dict(bf_agg),
                "q": dict(q_agg),
                "accuracy": total_correct / max(total_problems, 1),
                "n_correct": total_correct,
                "n_problems": total_problems,
                "avg_nodes": total_nodes / max(total_problems, 1),
            }
            print(f"k={k} {stage}: {total_correct}/{total_problems} correct, "
                  f"avg_nodes={total_nodes/max(total_problems,1):.1f}")

    # =========================================================================
    # Summary table
    # =========================================================================
    print("\n" + "=" * 100)
    print("SUMMARY TABLE: MCTS k=2/4/8 vs Flat Rollout")
    print("=" * 100)
    print(f"{'Stage':<12} | {'Metric':<15} | {'Flat Rollout':<14} | {'MCTS k=2':<12} | {'MCTS k=4':<12} | {'MCTS k=8':<12}")
    print("-" * 85)

    for stage in STAGES:
        flat_acc = {"step_0": 0.438, "step_40": 0.733, "step_80": 0.772, "step_120": 0.796}
        flat_d0_bf = flat_params.get(stage, {}).get("0", {}).get("poisson_lambda", 0)

        accs = []
        nodes = []
        d0_bfs = []
        for k in K_VALUES:
            d = all_data.get((k, stage), {})
            accs.append(f"{d.get('accuracy', 0):.0%}" if d else "n/a")
            nodes.append(f"{d.get('avg_nodes', 0):.1f}" if d else "n/a")
            bf_data = d.get("bf", {}).get(0, [])
            d0_bfs.append(f"{np.mean(bf_data):.1f}" if bf_data else "n/a")

        print(f"{STAGE_LABELS[stage]:<12} | {'Accuracy':<15} | {flat_acc[stage]:<14.1%} | {accs[0]:<12} | {accs[1]:<12} | {accs[2]:<12}")
        print(f"{'':12} | {'Avg Nodes':<15} | {'n/a':<14} | {nodes[0]:<12} | {nodes[1]:<12} | {nodes[2]:<12}")
        print(f"{'':12} | {'D0 Avg BF':<15} | {flat_d0_bf:<14.1f} | {d0_bfs[0]:<12} | {d0_bfs[1]:<12} | {d0_bfs[2]:<12}")
        print("-" * 85)

    # =========================================================================
    # Plot 1: Branching factor λ(d) — flat vs k=2/4/8
    # =========================================================================
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    colors_k = {2: "black", 4: "purple", 8: "brown"}
    colors_flat = {"step_0": "#e74c3c", "step_40": "#e67e22", "step_80": "#2ecc71", "step_120": "#3498db"}

    for idx, stage in enumerate(STAGES):
        ax = axes[idx // 2, idx % 2]

        # Flat rollout
        flat_depths = sorted(int(d) for d in flat_params.get(stage, {})
                            if "poisson_lambda" in flat_params[stage][d] and int(d) <= MAX_DEPTH)
        flat_lambdas = [flat_params[stage][str(d)]["poisson_lambda"] for d in flat_depths]
        ax.plot(flat_depths, flat_lambdas, "o-", color=colors_flat[stage],
                label="Flat Rollout", linewidth=2.5, markersize=7)

        # MCTS k=2,4,8
        for k in K_VALUES:
            d = all_data.get((k, stage), {})
            if not d:
                continue
            bf = d["bf"]
            depths = sorted(dd for dd in bf if dd <= MAX_DEPTH)
            lambdas = [np.mean(bf[dd]) for dd in depths]
            ax.plot(depths, lambdas, "s--", color=colors_k[k],
                    label=f"MCTS k={k}", linewidth=1.5, markersize=5)

        ax.set_title(f"{STAGE_LABELS[stage]}", fontsize=12)
        ax.set_xlabel("Depth")
        ax.set_ylabel("Avg Branching Factor")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0)

    fig.suptitle("Branching Factor by Depth: Flat Rollout vs Standard MCTS (k=2,4,8)", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "bf_comparison_all_k.png"), dpi=150, bbox_inches="tight")
    print("\nSaved bf_comparison_all_k.png")

    # =========================================================================
    # Plot 2: Accuracy vs Compute (nodes) trade-off
    # =========================================================================
    fig, ax = plt.subplots(figsize=(10, 6))

    for stage in STAGES:
        xs, ys, labels = [], [], []
        for k in K_VALUES:
            d = all_data.get((k, stage), {})
            if d:
                xs.append(d["avg_nodes"])
                ys.append(d["accuracy"])
                labels.append(f"k={k}")

        ax.plot(xs, ys, "o-", color=colors_flat[stage], label=STAGE_LABELS[stage],
                linewidth=2, markersize=8)
        for x, y, l in zip(xs, ys, labels):
            ax.annotate(l, (x, y), textcoords="offset points", xytext=(5, 5), fontsize=7)

    ax.set_xlabel("Avg Nodes per Tree (compute)")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy vs Compute Trade-off: Standard MCTS k=2,4,8")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.1)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "accuracy_vs_compute.png"), dpi=150, bbox_inches="tight")
    print("Saved accuracy_vs_compute.png")

    # =========================================================================
    # Plot 3: Branching factor distribution histograms at D0
    # =========================================================================
    fig, axes = plt.subplots(len(K_VALUES), len(STAGES), figsize=(16, 3.5 * len(K_VALUES)))

    for row, k in enumerate(K_VALUES):
        for col, stage in enumerate(stages_with_data := [s for s in STAGES]):
            ax = axes[row, col]
            d = all_data.get((k, stage), {})
            bf_data = d.get("bf", {}).get(0, []) if d else []

            if not bf_data:
                ax.set_title(f"k={k}, {STAGE_LABELS[stage]}\n(no data)", fontsize=9)
                continue

            arr = np.array(bf_data)
            max_val = min(int(np.max(arr)) + 2, 20)
            bins = np.arange(-0.5, max_val + 1.5, 1)
            ax.hist(arr, bins=bins, density=True, color=colors_flat[stage],
                    alpha=0.6, edgecolor="black", linewidth=0.5)
            ax.axvline(x=k, color="red", linestyle="--", alpha=0.7, label=f"k={k}")
            ax.set_title(f"k={k}, {STAGE_LABELS[stage]}\nmean={np.mean(arr):.1f}, n={len(arr)}", fontsize=9)
            ax.set_xlabel("Branching Factor")
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.2)

    fig.suptitle("D0 Branching Factor Distribution: MCTS k=2,4,8", fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "d0_bf_histograms.png"), dpi=150, bbox_inches="tight")
    print("Saved d0_bf_histograms.png")

    # =========================================================================
    # Save JSON summary
    # =========================================================================
    summary = {}
    for k in K_VALUES:
        k_summary = {}
        for stage in STAGES:
            d = all_data.get((k, stage), {})
            if not d:
                continue
            k_summary[stage] = {
                "accuracy": d["accuracy"],
                "avg_nodes": d["avg_nodes"],
                "n_problems": d["n_problems"],
            }
            for depth in range(MAX_DEPTH + 1):
                bf_data = d["bf"].get(depth, [])
                if bf_data:
                    k_summary[stage][f"d{depth}_bf_mean"] = float(np.mean(bf_data))
                    k_summary[stage][f"d{depth}_bf_count"] = len(bf_data)
        summary[f"k={k}"] = k_summary
    summary["flat_rollout"] = {stage: {"d0_lambda": flat_params[stage]["0"]["poisson_lambda"]}
                                for stage in STAGES if stage in flat_params}

    with open(os.path.join(OUTPUT_DIR, "mcts_comparison_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\nDone!")


if __name__ == "__main__":
    main()

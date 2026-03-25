#!/usr/bin/env python3
"""Analyze standard MCTS tree distributions and compare with flat rollouts.

Collects branching factors and node accuracies at each depth from MCTS trees,
fits Poisson/NegBin + Beta distributions, and compares with flat rollout parameters.
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

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from mcts_inference.utils import is_correct


def collect_mcts_tree_stats(tree_dict, ground_truth):
    """Extract per-depth branching factors and node accuracies from MCTS tree dict."""

    # First, annotate terminal nodes with correctness
    def annotate(node, gt):
        if node.get("is_terminal"):
            # Check if the answer in all_solutions matches
            # For terminal nodes, use text_preview to attempt answer extraction
            node["_correct"] = False  # will be set from all_solutions
        for c in node.get("children", []):
            annotate(c, gt)

    annotate(tree_dict, ground_truth)

    # Collect stats
    bf_by_depth = defaultdict(list)
    acc_by_depth = defaultdict(list)

    def collect(node, depth=0):
        children = node.get("children", [])

        # Branching factor: only for nodes with children
        if children:
            bf_by_depth[depth].append(len(children))

        # Node "accuracy": use q_value as proxy (visit-weighted reward)
        # Or count terminal correctness under this node
        terminals = collect_terminals(node)
        if terminals:
            # q_value is the mean reward from backpropagation
            acc_by_depth[depth].append(node.get("q_value", 0.5))

        for c in children:
            collect(c, depth + 1)

    def collect_terminals(node):
        if node.get("is_terminal"):
            return [node]
        result = []
        for c in node.get("children", []):
            result.extend(collect_terminals(c))
        return result

    collect(tree_dict)
    return dict(bf_by_depth), dict(acc_by_depth)


def fit_negbin(data):
    if not data or len(data) < 2:
        return None, None
    arr = np.array(data, dtype=float)
    mean = np.mean(arr)
    var = np.var(arr)
    if var <= mean or mean <= 0:
        return None, None
    p = mean / var
    r = mean * p / (1 - p)
    return r, p


def main():
    mcts_dir = "faithful_baseline/results/mcts_standard"
    flat_params_path = "faithful_baseline/results/math500_full/train/poisson_beta_analysis/fitted_parameters.json"
    output_dir = os.path.join(mcts_dir, "distribution_analysis")
    os.makedirs(output_dir, exist_ok=True)

    stages = ["step_0", "step_40", "step_80", "step_120"]
    stage_labels = {
        "step_0": "Base (step 0)",
        "step_40": "Step 40",
        "step_80": "Step 80",
        "step_120": "Step 120",
    }
    colors = {
        "step_0": "#e74c3c",
        "step_40": "#e67e22",
        "step_80": "#2ecc71",
        "step_120": "#3498db",
    }
    MAX_DEPTH = 10

    # Load flat rollout parameters
    with open(flat_params_path) as f:
        flat_params = json.load(f)
    print("Loaded flat rollout parameters")

    # Collect MCTS tree stats
    mcts_bf = {}  # stage -> depth -> list
    mcts_acc = {}

    for stage in stages:
        # Find MCTS result files
        pattern = f"mcts_{stage}_p"
        files = [f for f in os.listdir(mcts_dir) if f.startswith(pattern) and f.endswith(".json")]
        if not files:
            print(f"No MCTS results for {stage}")
            continue

        bf_agg = defaultdict(list)
        acc_agg = defaultdict(list)
        n_problems = 0

        for fname in sorted(files):
            with open(os.path.join(mcts_dir, fname)) as f:
                data = json.load(f)

            for result in data["results"]:
                tree = result.get("tree", {})
                gt = result.get("ground_truth", "")

                bf_d, acc_d = collect_mcts_tree_stats(tree, gt)
                for d, vals in bf_d.items():
                    if d <= MAX_DEPTH:
                        bf_agg[d].extend(vals)
                for d, vals in acc_d.items():
                    if d <= MAX_DEPTH:
                        acc_agg[d].extend(vals)
                n_problems += 1

        mcts_bf[stage] = dict(bf_agg)
        mcts_acc[stage] = dict(acc_agg)
        print(f"{stage}: {n_problems} problems, max depth with data: {max(bf_agg.keys()) if bf_agg else 0}")

    # =========================================================================
    # Fit MCTS distributions
    # =========================================================================
    mcts_params = {}
    for stage in stages:
        if stage not in mcts_bf:
            continue
        stage_fits = {}
        for d in range(MAX_DEPTH + 1):
            entry = {}
            bf_data = mcts_bf[stage].get(d, [])
            if bf_data:
                arr = np.array(bf_data, dtype=float)
                entry["poisson_lambda"] = float(np.mean(arr))
                entry["bf_mean"] = float(np.mean(arr))
                entry["bf_var"] = float(np.var(arr))
                entry["bf_count"] = len(bf_data)
                entry["dispersion"] = float(np.var(arr) / max(np.mean(arr), 1e-9))
                nb_r, nb_p = fit_negbin(bf_data)
                if nb_r is not None:
                    entry["negbin_r"] = float(nb_r)
                    entry["negbin_p"] = float(nb_p)

            acc_data = mcts_acc[stage].get(d, [])
            if acc_data:
                arr = np.array(acc_data, dtype=float)
                arr_clamped = np.clip(arr, 1e-6, 1 - 1e-6)
                try:
                    a, b, _, _ = sp_stats.beta.fit(arr_clamped, floc=0, fscale=1)
                except:
                    mean = np.mean(arr_clamped)
                    var = np.var(arr_clamped)
                    common = mean * (1 - mean) / max(var, 1e-6) - 1
                    a = max(mean * common, 0.01)
                    b = max((1 - mean) * common, 0.01)
                entry["beta_alpha"] = float(a)
                entry["beta_beta"] = float(b)
                entry["acc_mean"] = float(np.mean(acc_data))
                entry["acc_count"] = len(acc_data)

            if entry:
                stage_fits[str(d)] = entry
        mcts_params[stage] = stage_fits

    with open(os.path.join(output_dir, "mcts_fitted_parameters.json"), "w") as f:
        json.dump(mcts_params, f, indent=2)

    # =========================================================================
    # Print comparison table
    # =========================================================================
    print("\n" + "=" * 100)
    print("MCTS vs FLAT ROLLOUT: Branching Factor (Poisson λ)")
    print("=" * 100)
    for stage in stages:
        if stage not in mcts_params:
            continue
        print(f"\n{stage_labels[stage]}:")
        print(f"{'Depth':>5} | {'MCTS λ':>8} | {'Flat λ':>8} | {'Ratio':>8} | "
              f"{'MCTS disp':>10} | {'Flat disp':>10} | {'MCTS n':>7}")
        print("-" * 80)
        for d in range(MAX_DEPTH + 1):
            me = mcts_params[stage].get(str(d), {})
            fe = flat_params.get(stage, {}).get(str(d), {})
            if not me:
                continue
            ml = me.get("poisson_lambda", 0)
            fl = fe.get("poisson_lambda", 0)
            ratio = ml / fl if fl > 0 else float("inf")
            md = me.get("dispersion", 0)
            fd = fe.get("dispersion", 0)
            mn = me.get("bf_count", 0)
            print(f"{d:>5} | {ml:>8.2f} | {fl:>8.2f} | {ratio:>8.2f} | "
                  f"{md:>10.2f} | {fd:>10.2f} | {mn:>7}")

    # =========================================================================
    # Plot: MCTS vs Flat Rollout λ(d) comparison
    # =========================================================================
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for idx, stage in enumerate(stages):
        if stage not in mcts_params:
            continue
        ax = axes[idx // 2, idx % 2]

        # Flat rollout λ
        flat_depths = sorted(int(d) for d in flat_params.get(stage, {})
                            if "poisson_lambda" in flat_params[stage][d] and int(d) <= MAX_DEPTH)
        flat_lambdas = [flat_params[stage][str(d)]["poisson_lambda"] for d in flat_depths]
        ax.plot(flat_depths, flat_lambdas, "o-", color=colors[stage],
                label="Flat Rollout", linewidth=2, markersize=6)

        # MCTS λ
        mcts_depths = sorted(int(d) for d in mcts_params.get(stage, {})
                            if "poisson_lambda" in mcts_params[stage][d] and int(d) <= MAX_DEPTH)
        mcts_lambdas = [mcts_params[stage][str(d)]["poisson_lambda"] for d in mcts_depths]
        ax.plot(mcts_depths, mcts_lambdas, "s--", color="black",
                label="Standard MCTS", linewidth=2, markersize=6)

        ax.set_title(f"{stage_labels[stage]}", fontsize=12)
        ax.set_xlabel("Depth")
        ax.set_ylabel("Poisson λ (avg branching factor)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0)

    fig.suptitle("Branching Factor: Standard MCTS vs Flat Rollout Post-hoc Trees", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "mcts_vs_flat_lambda.png"), dpi=150, bbox_inches="tight")
    print("\nSaved mcts_vs_flat_lambda.png")

    # =========================================================================
    # Plot: MCTS vs Flat Rollout Beta mean comparison
    # =========================================================================
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for idx, stage in enumerate(stages):
        if stage not in mcts_params:
            continue
        ax = axes[idx // 2, idx % 2]

        # Flat rollout accuracy
        flat_depths = sorted(int(d) for d in flat_params.get(stage, {})
                            if "acc_mean" in flat_params[stage][d] and int(d) <= MAX_DEPTH)
        flat_acc = [flat_params[stage][str(d)].get("acc_mean", 0) for d in flat_depths]
        ax.plot(flat_depths, flat_acc, "o-", color=colors[stage],
                label="Flat Rollout", linewidth=2, markersize=6)

        # MCTS accuracy (q_value)
        mcts_depths = sorted(int(d) for d in mcts_params.get(stage, {})
                            if "acc_mean" in mcts_params[stage][d] and int(d) <= MAX_DEPTH)
        mcts_acc_vals = [mcts_params[stage][str(d)].get("acc_mean", 0) for d in mcts_depths]
        ax.plot(mcts_depths, mcts_acc_vals, "s--", color="black",
                label="Standard MCTS (Q-value)", linewidth=2, markersize=6)

        ax.set_title(f"{stage_labels[stage]}", fontsize=12)
        ax.set_xlabel("Depth")
        ax.set_ylabel("Mean Node Accuracy / Q-value")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.05)

    fig.suptitle("Node Accuracy: Standard MCTS vs Flat Rollout Post-hoc Trees", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "mcts_vs_flat_accuracy.png"), dpi=150, bbox_inches="tight")
    print("Saved mcts_vs_flat_accuracy.png")

    # =========================================================================
    # Compute divergence metrics
    # =========================================================================
    print("\n" + "=" * 80)
    print("DIVERGENCE SUMMARY (MCTS vs Flat Rollout)")
    print("=" * 80)

    divergence_summary = {}
    for stage in stages:
        if stage not in mcts_params:
            continue
        # KL divergence of branching factor distribution (approximated)
        bf_diffs = []
        acc_diffs = []
        for d in range(MAX_DEPTH + 1):
            me = mcts_params[stage].get(str(d), {})
            fe = flat_params.get(stage, {}).get(str(d), {})
            if me.get("poisson_lambda") and fe.get("poisson_lambda"):
                ml = me["poisson_lambda"]
                fl = fe["poisson_lambda"]
                # Relative difference
                bf_diffs.append(abs(ml - fl) / max(fl, 0.01))
            if me.get("acc_mean") is not None and fe.get("acc_mean") is not None:
                acc_diffs.append(abs(me["acc_mean"] - fe["acc_mean"]))

        avg_bf_diff = np.mean(bf_diffs) if bf_diffs else 0
        avg_acc_diff = np.mean(acc_diffs) if acc_diffs else 0
        print(f"\n{stage_labels[stage]}:")
        print(f"  Avg |Δλ|/λ_flat: {avg_bf_diff:.3f} (branching factor relative diff)")
        print(f"  Avg |Δacc|:      {avg_acc_diff:.3f} (accuracy absolute diff)")

        divergence_summary[stage] = {
            "avg_bf_relative_diff": float(avg_bf_diff),
            "avg_acc_absolute_diff": float(avg_acc_diff),
        }

    with open(os.path.join(output_dir, "divergence_summary.json"), "w") as f:
        json.dump(divergence_summary, f, indent=2)
    print(f"\nSaved divergence_summary.json")
    print("\nDone!")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Fit Poisson (branching factor) and Beta (node accuracy) distributions
at each depth of post-hoc syntactic trees from flat rollouts.

For 400 problems × 4 training stages (1600 trees total):
- At each depth d, collect all branching factors → fit Poisson(λ_d)
- At each depth d, collect all node accuracies → fit Beta(α_d, β_d)

Outputs distribution parameter tables and comparison plots.
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


def collect_leaf_rollout_ids(node):
    if not node.get("children"):
        return set(node.get("rollout_ids", []))
    ids = set()
    for c in node["children"]:
        ids |= collect_leaf_rollout_ids(c)
    return ids


def collect_nodes_at_depth(tree, target_depth, current_depth=0):
    if current_depth == target_depth:
        return [tree]
    nodes = []
    for c in tree.get("children", []):
        nodes.extend(collect_nodes_at_depth(c, target_depth, current_depth + 1))
    return nodes


def max_depth(node, d=0):
    if not node.get("children"):
        return d
    return max(max_depth(c, d + 1) for c in node["children"])


def extract_depth_data(tree, correctness, max_d=15):
    """Extract branching factors and node accuracies at each depth."""
    md = max_depth(tree)
    bf_by_depth = defaultdict(list)
    acc_by_depth = defaultdict(list)

    for d in range(min(md, max_d) + 1):
        nodes = collect_nodes_at_depth(tree, d)
        for node in nodes:
            children = node.get("children", [])
            # Branching factor: only for non-leaf nodes
            if children:
                bf_by_depth[d].append(len(children))

            # Node accuracy
            rids = list(collect_leaf_rollout_ids(node))
            if rids:
                n_correct = sum(1 for r in rids if correctness.get(r, False))
                acc = n_correct / len(rids)
                acc_by_depth[d].append(acc)

    return bf_by_depth, acc_by_depth


def fit_poisson(data):
    """Fit Poisson distribution. Returns lambda parameter."""
    if not data:
        return None, None
    data = np.array(data, dtype=float)
    # Poisson MLE: lambda = mean
    lam = np.mean(data)
    # Goodness of fit via chi-squared test
    return lam, data


def fit_beta(data):
    """Fit Beta distribution. Returns (alpha, beta) parameters."""
    if not data or len(data) < 2:
        return None, None, None
    data = np.array(data, dtype=float)
    # Clamp to (0, 1) exclusive for Beta fitting
    data = np.clip(data, 1e-6, 1 - 1e-6)
    try:
        a, b, loc, scale = sp_stats.beta.fit(data, floc=0, fscale=1)
        return a, b, data
    except Exception:
        # Fallback: method of moments
        mean = np.mean(data)
        var = np.var(data)
        if var == 0 or mean == 0 or mean == 1:
            return 1.0, 1.0, data
        common = mean * (1 - mean) / var - 1
        a = mean * common
        b = (1 - mean) * common
        return max(a, 0.01), max(b, 0.01), data


def main():
    tree_base = "faithful_baseline/results/math500_full/train/trees_syntactic"
    rollout_dir = "faithful_baseline/results/math500_full/train"
    output_dir = "faithful_baseline/results/math500_full/train/poisson_beta_analysis"
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
    MAX_DEPTH = 12

    # Load correctness data
    stage_correctness = {}
    for stage in stages:
        rollout_path = os.path.join(rollout_dir, f"rollouts_{stage}.json")
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
        print(f"Loaded correctness for {stage}")

    # Collect branching factors and accuracies across all trees
    all_bf = {}  # stage -> depth -> list of branching factors
    all_acc = {}  # stage -> depth -> list of accuracies

    for stage in stages:
        stage_dir = os.path.join(tree_base, stage)
        tree_files = sorted(f for f in os.listdir(stage_dir) if f.startswith("tree_"))
        print(f"\n{stage}: processing {len(tree_files)} trees...")

        bf_agg = defaultdict(list)
        acc_agg = defaultdict(list)

        for tf in tree_files:
            pid = int(tf.split("_p")[1].split(".")[0])
            with open(os.path.join(stage_dir, tf)) as f:
                tree = json.load(f)
            corr = stage_correctness[stage].get(pid, {})
            bf_by_d, acc_by_d = extract_depth_data(tree, corr, max_d=MAX_DEPTH)
            for d in bf_by_d:
                bf_agg[d].extend(bf_by_d[d])
            for d in acc_by_d:
                acc_agg[d].extend(acc_by_d[d])

        all_bf[stage] = dict(bf_agg)
        all_acc[stage] = dict(acc_agg)

    # =========================================================================
    # Fit distributions and save parameters
    # =========================================================================
    fit_results = {}
    for stage in stages:
        stage_fits = {}
        for d in range(MAX_DEPTH + 1):
            entry = {}
            # Poisson for branching factor
            bf_data = all_bf[stage].get(d, [])
            if bf_data:
                lam, _ = fit_poisson(bf_data)
                entry["poisson_lambda"] = float(lam)
                entry["bf_mean"] = float(np.mean(bf_data))
                entry["bf_std"] = float(np.std(bf_data))
                entry["bf_count"] = len(bf_data)

            # Beta for accuracy
            acc_data = all_acc[stage].get(d, [])
            if acc_data:
                a, b, _ = fit_beta(acc_data)
                if a is not None:
                    entry["beta_alpha"] = float(a)
                    entry["beta_beta"] = float(b)
                entry["acc_mean"] = float(np.mean(acc_data))
                entry["acc_std"] = float(np.std(acc_data))
                entry["acc_count"] = len(acc_data)

            if entry:
                stage_fits[str(d)] = entry
        fit_results[stage] = stage_fits

    with open(os.path.join(output_dir, "fitted_parameters.json"), "w") as f:
        json.dump(fit_results, f, indent=2)
    print("\nSaved fitted_parameters.json")

    # Print summary table
    print("\n" + "=" * 80)
    print("FITTED DISTRIBUTION PARAMETERS")
    print("=" * 80)
    for stage in stages:
        print(f"\n{stage_labels[stage]}:")
        print(f"{'Depth':>5} | {'Poisson λ':>10} | {'BF mean':>8} | {'BF std':>8} | "
              f"{'Beta α':>8} | {'Beta β':>8} | {'Acc mean':>8} | {'#nodes':>7}")
        print("-" * 80)
        for d in range(MAX_DEPTH + 1):
            e = fit_results[stage].get(str(d), {})
            if not e:
                continue
            lam = e.get("poisson_lambda", 0)
            bf_m = e.get("bf_mean", 0)
            bf_s = e.get("bf_std", 0)
            ba = e.get("beta_alpha", 0)
            bb = e.get("beta_beta", 0)
            am = e.get("acc_mean", 0)
            n = e.get("bf_count", e.get("acc_count", 0))
            print(f"{d:>5} | {lam:>10.2f} | {bf_m:>8.2f} | {bf_s:>8.2f} | "
                  f"{ba:>8.2f} | {bb:>8.2f} | {am:>8.3f} | {n:>7}")

    # =========================================================================
    # Plot 1: Poisson λ(d) curves across stages
    # =========================================================================
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for stage in stages:
        depths = sorted(int(d) for d in fit_results[stage] if "poisson_lambda" in fit_results[stage][d])
        depths = [d for d in depths if d <= MAX_DEPTH]
        lambdas = [fit_results[stage][str(d)]["poisson_lambda"] for d in depths]
        axes[0].plot(depths, lambdas, "o-", color=colors[stage],
                     label=stage_labels[stage], markersize=5)

    axes[0].set_title("Poisson λ(d): Branching Factor Parameter by Depth", fontsize=12)
    axes[0].set_xlabel("Depth")
    axes[0].set_ylabel("λ (Poisson parameter)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_yscale("log")

    # Beta parameters
    for stage in stages:
        depths = sorted(int(d) for d in fit_results[stage] if "beta_alpha" in fit_results[stage][d])
        depths = [d for d in depths if d <= MAX_DEPTH]
        alphas = [fit_results[stage][str(d)]["beta_alpha"] for d in depths]
        betas = [fit_results[stage][str(d)]["beta_beta"] for d in depths]
        axes[1].plot(depths, alphas, "o-", color=colors[stage],
                     label=f"{stage_labels[stage]} α", markersize=5)
        axes[1].plot(depths, betas, "s--", color=colors[stage],
                     label=f"{stage_labels[stage]} β", markersize=4, alpha=0.6)

    axes[1].set_title("Beta(α,β) at Depth d: Node Accuracy Distribution", fontsize=12)
    axes[1].set_xlabel("Depth")
    axes[1].set_ylabel("Parameter value")
    axes[1].legend(fontsize=7, ncol=2)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_yscale("log")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "parameter_curves.png"), dpi=150, bbox_inches="tight")
    print("Saved parameter_curves.png")

    # =========================================================================
    # Plot 2: Poisson fit quality — histograms + fitted PDF at selected depths
    # =========================================================================
    selected_depths = [0, 1, 2, 4]
    fig, axes = plt.subplots(len(selected_depths), len(stages), figsize=(16, 3.5 * len(selected_depths)))

    for row, d in enumerate(selected_depths):
        for col, stage in enumerate(stages):
            ax = axes[row, col]
            bf_data = all_bf[stage].get(d, [])
            if not bf_data:
                ax.set_title(f"{stage_labels[stage]}, D{d}\n(no data)", fontsize=9)
                continue

            bf_arr = np.array(bf_data)
            lam = np.mean(bf_arr)

            # Histogram
            max_val = min(int(np.percentile(bf_arr, 99)) + 2, 50)
            bins = np.arange(-0.5, max_val + 1.5, 1)
            ax.hist(bf_arr, bins=bins, density=True, color=colors[stage],
                    alpha=0.6, edgecolor="black", linewidth=0.5, label="Data")

            # Poisson PMF overlay
            x_range = np.arange(0, max_val + 1)
            poisson_pmf = sp_stats.poisson.pmf(x_range, lam)
            ax.plot(x_range, poisson_pmf, "k.-", linewidth=1.5, markersize=4,
                    label=f"Poisson(λ={lam:.2f})")

            ax.set_title(f"{stage_labels[stage]}, D{d}\nλ={lam:.2f}, n={len(bf_data)}", fontsize=9)
            ax.set_xlabel("Branching Factor")
            ax.set_ylabel("Density")
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.2)

    fig.suptitle("Poisson Fit to Branching Factor Distribution P(bf | depth, stage)",
                 fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "poisson_fit_histograms.png"), dpi=150, bbox_inches="tight")
    print("Saved poisson_fit_histograms.png")

    # =========================================================================
    # Plot 3: Beta fit quality — histograms + fitted PDF at selected depths
    # =========================================================================
    fig, axes = plt.subplots(len(selected_depths), len(stages), figsize=(16, 3.5 * len(selected_depths)))

    for row, d in enumerate(selected_depths):
        for col, stage in enumerate(stages):
            ax = axes[row, col]
            acc_data = all_acc[stage].get(d, [])
            if not acc_data:
                ax.set_title(f"{stage_labels[stage]}, D{d}\n(no data)", fontsize=9)
                continue

            acc_arr = np.array(acc_data)
            acc_clamped = np.clip(acc_arr, 1e-6, 1 - 1e-6)

            try:
                a, b, _, _ = sp_stats.beta.fit(acc_clamped, floc=0, fscale=1)
            except:
                mean = np.mean(acc_clamped)
                var = np.var(acc_clamped)
                common = mean * (1 - mean) / max(var, 1e-6) - 1
                a = max(mean * common, 0.01)
                b = max((1 - mean) * common, 0.01)

            # Histogram
            bins = np.linspace(0, 1, 30)
            ax.hist(acc_arr, bins=bins, density=True, color=colors[stage],
                    alpha=0.6, edgecolor="black", linewidth=0.5, label="Data")

            # Beta PDF overlay
            x_range = np.linspace(0.001, 0.999, 200)
            beta_pdf = sp_stats.beta.pdf(x_range, a, b)
            # Clip extreme values for display
            beta_pdf = np.clip(beta_pdf, 0, np.percentile(beta_pdf, 99) * 2)
            ax.plot(x_range, beta_pdf, "k-", linewidth=1.5,
                    label=f"Beta(α={a:.2f},β={b:.2f})")

            ax.set_title(f"{stage_labels[stage]}, D{d}\nα={a:.2f}, β={b:.2f}, n={len(acc_data)}",
                         fontsize=9)
            ax.set_xlabel("Accuracy")
            ax.set_ylabel("Density")
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.2)

    fig.suptitle("Beta Fit to Node Accuracy Distribution P(acc | depth, stage)",
                 fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "beta_fit_histograms.png"), dpi=150, bbox_inches="tight")
    print("Saved beta_fit_histograms.png")

    # =========================================================================
    # Plot 4: Combined parameter summary — λ(d) and E[acc](d)
    # =========================================================================
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Panel 1: Poisson λ(d)
    for stage in stages:
        depths = sorted(int(d) for d in fit_results[stage] if "poisson_lambda" in fit_results[stage][d])
        depths = [d for d in depths if d <= MAX_DEPTH]
        lambdas = [fit_results[stage][str(d)]["poisson_lambda"] for d in depths]
        axes[0].plot(depths, lambdas, "o-", color=colors[stage],
                     label=stage_labels[stage], markersize=5, linewidth=2)
    axes[0].set_title("Poisson λ(d)")
    axes[0].set_xlabel("Depth d")
    axes[0].set_ylabel("λ")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Panel 2: Beta mean = α/(α+β)
    for stage in stages:
        depths = sorted(int(d) for d in fit_results[stage] if "beta_alpha" in fit_results[stage][d])
        depths = [d for d in depths if d <= MAX_DEPTH]
        means = [fit_results[stage][str(d)]["beta_alpha"] /
                 (fit_results[stage][str(d)]["beta_alpha"] + fit_results[stage][str(d)]["beta_beta"])
                 for d in depths]
        axes[1].plot(depths, means, "o-", color=colors[stage],
                     label=stage_labels[stage], markersize=5, linewidth=2)
    axes[1].set_title("Beta Mean = α/(α+β)\n(Expected node accuracy)")
    axes[1].set_xlabel("Depth d")
    axes[1].set_ylabel("E[accuracy]")
    axes[1].set_ylim(0, 1.05)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Panel 3: Beta concentration = α+β (higher = more peaked)
    for stage in stages:
        depths = sorted(int(d) for d in fit_results[stage] if "beta_alpha" in fit_results[stage][d])
        depths = [d for d in depths if d <= MAX_DEPTH]
        conc = [fit_results[stage][str(d)]["beta_alpha"] + fit_results[stage][str(d)]["beta_beta"]
                for d in depths]
        axes[2].plot(depths, conc, "o-", color=colors[stage],
                     label=stage_labels[stage], markersize=5, linewidth=2)
    axes[2].set_title("Beta Concentration α+β\n(higher = more certain)")
    axes[2].set_xlabel("Depth d")
    axes[2].set_ylabel("α + β")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    axes[2].set_yscale("log")

    fig.suptitle("Flat Rollout Tree Statistics: Poisson & Beta Distribution Parameters (400 problems × 4 stages)",
                 fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "parameter_summary.png"), dpi=150, bbox_inches="tight")
    print("Saved parameter_summary.png")

    print("\nDone!")


if __name__ == "__main__":
    main()

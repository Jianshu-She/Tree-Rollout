#!/usr/bin/env python3
"""Cross-stage evolution of 4-method comparison metrics.

Reads comparison_step_{0,40,80,120}.json files and plots how accuracy,
accuracy gap, Pearson correlation (drift), no-advantage rate, and
compute ratio evolve over training stages.
"""

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr

STAGES = ["step_0", "step_40", "step_80", "step_120"]
STAGE_X = [0, 40, 80, 120]

METHODS = [
    ("flat",         "Flat Rollout", "#4C72B0", "o"),
    ("bfs_tree",     "BFS Tree",    "#DD8452", "s"),
    ("poisson_mcts", "NegBin MCTS", "#55A868", "^"),
    ("deepsearch",   "DeepSearch",  "#C44E52", "D"),
]

plt.rcParams.update({"font.size": 11, "figure.dpi": 150, "savefig.dpi": 150})


def load_stage(base_dir, stage):
    path = os.path.join(base_dir, stage, f"comparison_{stage}.json")
    with open(path) as f:
        return json.load(f)["results"]


def extract_metrics(all_results):
    """Extract per-stage per-method metrics."""
    out = {m[0]: {"acc": [], "tok_ratio": [], "pearson": [], "no_adv": [],
                  "no_adv_correct": [], "no_adv_wrong": []} for m in METHODS}
    for stage in STAGES:
        results = all_results[stage]
        fa = np.array([r["flat"]["accuracy"] for r in results])
        ft = np.array([r["flat"]["total_tokens"] for r in results])
        n = len(results)
        for key, *_ in METHODS:
            accs = np.array([r[key]["accuracy"] for r in results])
            toks = np.array([r[key]["total_tokens"] for r in results])
            pr = pearsonr(fa, accs)[0] if key != "flat" else 1.0
            no_adv_mask = np.array([r[key]["adv_std"] < 0.01 for r in results])
            all_correct = no_adv_mask & (accs > 0.99)
            all_wrong = no_adv_mask & (accs < 0.01)
            out[key]["acc"].append(float(accs.mean()))
            out[key]["tok_ratio"].append(float(toks.sum() / ft.sum()))
            out[key]["pearson"].append(float(pr))
            out[key]["no_adv"].append(100.0 * no_adv_mask.sum() / n)
            out[key]["no_adv_correct"].append(100.0 * all_correct.sum() / n)
            out[key]["no_adv_wrong"].append(100.0 * all_wrong.sum() / n)
    return out


def plot_all(metrics, out_dir):
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # Panel 1: Accuracy vs stage
    ax = axes[0, 0]
    for key, label, color, marker in METHODS:
        ax.plot(STAGE_X, metrics[key]["acc"], f"{marker}-", color=color,
                label=label, markersize=8, lw=2)
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Mean Accuracy")
    ax.set_title("Accuracy vs Training Stage")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.3, 0.95)

    # Panel 2: Accuracy gap vs Flat
    ax = axes[0, 1]
    flat_acc = np.array(metrics["flat"]["acc"])
    for key, label, color, marker in METHODS:
        if key == "flat":
            continue
        gap = np.array(metrics[key]["acc"]) - flat_acc
        ax.plot(STAGE_X, gap * 100, f"{marker}-", color=color,
                label=label, markersize=8, lw=2)
    ax.axhline(0, color="black", ls="--", lw=1, alpha=0.5)
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Accuracy Gap vs Flat (pp)")
    ax.set_title("Accuracy Gap vs Flat Rollout")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel 3: Pearson correlation vs Flat (drift)
    ax = axes[0, 2]
    for key, label, color, marker in METHODS:
        if key == "flat":
            continue
        ax.plot(STAGE_X, metrics[key]["pearson"], f"{marker}-", color=color,
                label=label, markersize=8, lw=2)
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Pearson r (vs Flat)")
    ax.set_title("Per-Problem Accuracy Correlation with Flat\n(lower = more drift)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.7, 1.0)

    # Panel 4: No-advantage rate
    ax = axes[1, 0]
    for key, label, color, marker in METHODS:
        ax.plot(STAGE_X, metrics[key]["no_adv"], f"{marker}-", color=color,
                label=label, markersize=8, lw=2)
    ax.set_xlabel("Training Step")
    ax.set_ylabel("% No-Advantage Problems")
    ax.set_title("No-Advantage Rate vs Training Stage")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel 5: No-advantage split (all-correct vs all-wrong)
    ax = axes[1, 1]
    for key, label, color, marker in METHODS:
        ax.plot(STAGE_X, metrics[key]["no_adv_correct"], f"{marker}-",
                color=color, label=f"{label} (all-correct)", markersize=6, lw=1.5)
        ax.plot(STAGE_X, metrics[key]["no_adv_wrong"], f"{marker}--",
                color=color, label=f"{label} (all-wrong)", markersize=6, lw=1.5,
                alpha=0.6)
    ax.set_xlabel("Training Step")
    ax.set_ylabel("% Problems")
    ax.set_title("No-Advantage Split: All-Correct (good) vs All-Wrong (bad)")
    ax.legend(fontsize=7, ncol=2, loc="upper left")
    ax.grid(True, alpha=0.3)

    # Panel 6: Compute ratio vs stage
    ax = axes[1, 2]
    for key, label, color, marker in METHODS:
        if key == "flat":
            continue
        ax.plot(STAGE_X, [r * 100 for r in metrics[key]["tok_ratio"]],
                f"{marker}-", color=color, label=label, markersize=8, lw=2)
    ax.axhline(100, color="#4C72B0", ls="--", lw=1, alpha=0.5, label="Flat (100%)")
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Compute (% of Flat)")
    ax.set_title("Compute Ratio vs Training Stage")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.suptitle(
        "Cross-Stage Evolution: 4-Method Comparison on 100 MATH500 Problems\n"
        "Models: Qwen2.5-Math-7B checkpoints from flat GRPO training (step 0→120)",
        fontsize=14, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])

    out_path = os.path.join(out_dir, "cross_stage_evolution.png")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")
    return out_path


def main():
    base_dir = sys.argv[1] if len(sys.argv) > 1 else \
        "poisson_mcts/results/advantage_comparison"
    out_dir = base_dir

    all_results = {}
    for stage in STAGES:
        all_results[stage] = load_stage(base_dir, stage)
        print(f"Loaded {stage}: {len(all_results[stage])} problems")

    metrics = extract_metrics(all_results)
    plot_all(metrics, out_dir)

    # Save metrics JSON
    json_path = os.path.join(out_dir, "cross_stage_metrics.json")
    with open(json_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved {json_path}")


if __name__ == "__main__":
    main()

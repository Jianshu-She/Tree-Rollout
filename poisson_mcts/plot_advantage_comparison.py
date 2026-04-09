#!/usr/bin/env python3
"""Visualize GRPO advantage comparison: Flat Rollout vs BFS Tree vs Poisson-MCTS."""

import json
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import pearsonr, spearmanr

# Style
plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "figure.dpi": 150,
    "savefig.dpi": 150,
})

COLORS = {"flat": "#4C72B0", "bfs": "#DD8452", "mcts": "#55A868"}
LABELS = {"flat": "Flat Rollout", "bfs": "BFS Tree", "mcts": "Poisson-MCTS"}


def load_data(path):
    with open(path) as f:
        return json.load(f)


def plot_accuracy_scatter(results, out_dir):
    """Scatter: Flat vs BFS / Flat vs MCTS accuracy per problem."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))

    fa = [r["flat"]["accuracy"] for r in results]
    ba = [r["bfs_tree"]["accuracy"] for r in results]
    ma = [r["poisson_mcts"]["accuracy"] for r in results]
    bn = [r["bfs_tree"]["num_trajectories"] for r in results]
    mn = [r["poisson_mcts"]["num_trajectories"] for r in results]

    # Flat vs BFS
    ax = axes[0]
    sc = ax.scatter(fa, ba, c=bn, cmap="viridis", s=40, alpha=0.8, edgecolors="w", linewidths=0.3)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3, lw=1)
    ax.set_xlabel("Flat Rollout Accuracy")
    ax.set_ylabel("BFS Tree Accuracy")
    pr, _ = pearsonr(fa, ba)
    sr, _ = spearmanr(fa, ba)
    ax.set_title(f"Flat vs BFS  (Pearson={pr:.3f}, Spearman={sr:.3f})")
    cb = fig.colorbar(sc, ax=ax, shrink=0.8)
    cb.set_label("BFS #trajectories")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_aspect("equal")

    # Flat vs MCTS
    ax = axes[1]
    sc = ax.scatter(fa, ma, c=mn, cmap="viridis", s=40, alpha=0.8, edgecolors="w", linewidths=0.3)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3, lw=1)
    ax.set_xlabel("Flat Rollout Accuracy")
    ax.set_ylabel("Poisson-MCTS Accuracy")
    pr, _ = pearsonr(fa, ma)
    sr, _ = spearmanr(fa, ma)
    ax.set_title(f"Flat vs MCTS  (Pearson={pr:.3f}, Spearman={sr:.3f})")
    cb = fig.colorbar(sc, ax=ax, shrink=0.8)
    cb.set_label("MCTS #trajectories")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_aspect("equal")

    fig.suptitle("Per-Problem Accuracy: Tree Methods vs Flat Rollout", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(f"{out_dir}/accuracy_scatter.png", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_dir}/accuracy_scatter.png")


def plot_accuracy_diff_histogram(results, out_dir):
    """Histogram of accuracy differences: BFS-Flat and MCTS-Flat."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    fa = np.array([r["flat"]["accuracy"] for r in results])
    ba = np.array([r["bfs_tree"]["accuracy"] for r in results])
    ma = np.array([r["poisson_mcts"]["accuracy"] for r in results])

    diff_fb = ba - fa
    diff_fm = ma - fa

    bins = np.arange(-0.55, 0.6, 0.05)

    ax = axes[0]
    ax.hist(diff_fb, bins=bins, color=COLORS["bfs"], alpha=0.8, edgecolor="white")
    ax.axvline(0, color="black", ls="--", lw=1, alpha=0.5)
    ax.axvline(np.mean(diff_fb), color="red", ls="-", lw=1.5, label=f"mean={np.mean(diff_fb):+.1%}")
    ax.set_xlabel("BFS Accuracy - Flat Accuracy")
    ax.set_ylabel("Count")
    ax.set_title("BFS vs Flat")
    ax.legend()

    ax = axes[1]
    ax.hist(diff_fm, bins=bins, color=COLORS["mcts"], alpha=0.8, edgecolor="white")
    ax.axvline(0, color="black", ls="--", lw=1, alpha=0.5)
    ax.axvline(np.mean(diff_fm), color="red", ls="-", lw=1.5, label=f"mean={np.mean(diff_fm):+.1%}")
    ax.set_xlabel("MCTS Accuracy - Flat Accuracy")
    ax.set_ylabel("Count")
    ax.set_title("MCTS vs Flat")
    ax.legend()

    fig.suptitle("Per-Problem Accuracy Difference Distribution", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(f"{out_dir}/accuracy_diff_histogram.png", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_dir}/accuracy_diff_histogram.png")


def plot_accuracy_bar(results, out_dir):
    """Grouped bar chart: per-problem accuracy, sorted by flat accuracy."""
    fa = np.array([r["flat"]["accuracy"] for r in results])
    ba = np.array([r["bfs_tree"]["accuracy"] for r in results])
    ma = np.array([r["poisson_mcts"]["accuracy"] for r in results])
    idx = np.array([r["problem_index"] for r in results])

    order = np.argsort(fa)
    fa, ba, ma, idx = fa[order], ba[order], ma[order], idx[order]

    fig, ax = plt.subplots(figsize=(18, 6))
    x = np.arange(len(results))
    w = 0.27
    ax.bar(x - w, fa, w, color=COLORS["flat"], alpha=0.85, label=LABELS["flat"])
    ax.bar(x, ba, w, color=COLORS["bfs"], alpha=0.85, label=LABELS["bfs"])
    ax.bar(x + w, ma, w, color=COLORS["mcts"], alpha=0.85, label=LABELS["mcts"])
    ax.set_xlabel("Problems (sorted by Flat accuracy)")
    ax.set_ylabel("Accuracy")
    ax.set_title("Per-Problem Accuracy Comparison (100 Problems, step_0)", fontsize=15, fontweight="bold")
    ax.legend(loc="upper left")
    ax.set_xticks(x[::5])
    ax.set_xticklabels([f"p{i}" for i in idx[::5]], fontsize=8, rotation=45)
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    fig.savefig(f"{out_dir}/accuracy_bar_sorted.png", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_dir}/accuracy_bar_sorted.png")


def plot_trajectory_counts(results, out_dir):
    """Trajectory count distribution for BFS and MCTS."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    bn = [r["bfs_tree"]["num_trajectories"] for r in results]
    mn = [r["poisson_mcts"]["num_trajectories"] for r in results]

    ax = axes[0]
    ax.hist(bn, bins=20, color=COLORS["bfs"], alpha=0.8, edgecolor="white")
    ax.axvline(128, color="red", ls="--", lw=1.5, label="Target=128")
    ax.axvline(np.mean(bn), color="black", ls="-", lw=1.5, label=f"mean={np.mean(bn):.0f}")
    ax.set_xlabel("#Trajectories")
    ax.set_ylabel("Count")
    ax.set_title(f"BFS Tree (mean={np.mean(bn):.0f}, range={min(bn)}-{max(bn)})")
    ax.legend()

    ax = axes[1]
    ax.hist(mn, bins=30, color=COLORS["mcts"], alpha=0.8, edgecolor="white")
    ax.axvline(128, color="red", ls="--", lw=1.5, label="Target=128")
    ax.axvline(np.mean(mn), color="black", ls="-", lw=1.5, label=f"mean={np.mean(mn):.0f}")
    ax.set_xlabel("#Trajectories")
    ax.set_ylabel("Count")
    ax.set_title(f"Poisson-MCTS (mean={np.mean(mn):.0f}, range={min(mn)}-{max(mn)})")
    ax.legend()

    fig.suptitle("Number of Complete Trajectories (Target: 128)", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(f"{out_dir}/trajectory_counts.png", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_dir}/trajectory_counts.png")


def plot_token_efficiency(results, out_dir):
    """Token cost comparison."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    ft = [r["flat"]["total_tokens"] for r in results]
    bt = [r["bfs_tree"]["total_tokens"] for r in results]
    mt = [r["poisson_mcts"]["total_tokens"] for r in results]

    # Box plot
    ax = axes[0]
    bp = ax.boxplot(
        [np.array(ft) / 1000, np.array(bt) / 1000, np.array(mt) / 1000],
        tick_labels=["Flat", "BFS", "MCTS"],
        patch_artist=True,
        widths=0.5,
    )
    for patch, color in zip(bp["boxes"], [COLORS["flat"], COLORS["bfs"], COLORS["mcts"]]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_ylabel("Total Tokens (K)")
    ax.set_title("Token Cost per Problem")

    # Ratio scatter
    ax = axes[1]
    ratio_b = np.array(bt) / np.array(ft)
    ratio_m = np.array(mt) / np.array(ft)
    x = np.arange(len(results))
    ax.scatter(x, ratio_b, s=20, alpha=0.6, color=COLORS["bfs"], label=f"BFS/Flat (mean={np.mean(ratio_b):.1%})")
    ax.scatter(x, ratio_m, s=20, alpha=0.6, color=COLORS["mcts"], label=f"MCTS/Flat (mean={np.mean(ratio_m):.1%})")
    ax.axhline(1.0, color="black", ls="--", lw=1, alpha=0.3)
    ax.set_xlabel("Problem Index")
    ax.set_ylabel("Token Ratio (Tree / Flat)")
    ax.set_title("Token Savings per Problem")
    ax.legend(fontsize=9)
    ax.set_ylim(0, max(max(ratio_b), max(ratio_m)) * 1.1)

    fig.suptitle("Compute Efficiency: Tree Methods vs Flat Rollout", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(f"{out_dir}/token_efficiency.png", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_dir}/token_efficiency.png")


def plot_purity_analysis(results, out_dir):
    """Purity (adv_std=0) analysis: who loses signal and why."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))

    fa = np.array([r["flat"]["accuracy"] for r in results])
    ba = np.array([r["bfs_tree"]["accuracy"] for r in results])
    ma = np.array([r["poisson_mcts"]["accuracy"] for r in results])
    f_pure = np.array([r["flat"]["adv_std"] < 0.01 for r in results])
    b_pure = np.array([r["bfs_tree"]["adv_std"] < 0.01 for r in results])
    m_pure = np.array([r["poisson_mcts"]["adv_std"] < 0.01 for r in results])
    bn = np.array([r["bfs_tree"]["num_trajectories"] for r in results])
    mn = np.array([r["poisson_mcts"]["num_trajectories"] for r in results])

    # Purity counts
    ax = axes[0]
    counts = [f_pure.sum(), b_pure.sum(), m_pure.sum()]
    bars = ax.bar(["Flat", "BFS", "MCTS"], counts,
                  color=[COLORS["flat"], COLORS["bfs"], COLORS["mcts"]], alpha=0.8)
    for bar, c in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                str(c), ha="center", fontsize=12, fontweight="bold")
    ax.set_ylabel("# Pure Problems (no RL signal)")
    ax.set_title("Pure Groups (adv_std = 0)")
    ax.set_ylim(0, max(counts) * 1.3)

    # MCTS: trajectory count vs purity
    ax = axes[1]
    ax.scatter(mn[~m_pure], ma[~m_pure], s=30, alpha=0.6, color=COLORS["mcts"], label="Has signal")
    ax.scatter(mn[m_pure], ma[m_pure], s=60, alpha=0.9, color="red", marker="x", lw=2, label="Pure (no signal)")
    ax.set_xlabel("MCTS #Trajectories")
    ax.set_ylabel("MCTS Accuracy")
    ax.set_title("MCTS: Trajectory Count vs Purity")
    ax.legend(fontsize=9)

    # Flat accuracy of problems that became pure in tree methods
    ax = axes[2]
    bfs_lost = ~f_pure & b_pure
    mcts_lost = ~f_pure & m_pure
    bins = np.arange(0, 1.05, 0.1)
    if bfs_lost.sum() > 0:
        ax.hist(fa[bfs_lost], bins=bins, alpha=0.6, color=COLORS["bfs"],
                label=f"BFS lost ({bfs_lost.sum()})", edgecolor="white")
    if mcts_lost.sum() > 0:
        ax.hist(fa[mcts_lost], bins=bins, alpha=0.6, color=COLORS["mcts"],
                label=f"MCTS lost ({mcts_lost.sum()})", edgecolor="white")
    ax.set_xlabel("Flat Rollout Accuracy")
    ax.set_ylabel("Count")
    ax.set_title("Signal Lost: Flat Accuracy of Affected Problems")
    ax.legend(fontsize=9)

    fig.suptitle("Purity Analysis: When Does Tree Search Lose RL Signal?", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(f"{out_dir}/purity_analysis.png", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_dir}/purity_analysis.png")


def plot_advantage_distributions(results, out_dir):
    """Example advantage distributions for selected problems."""
    # Pick 6 representative problems
    candidates = []
    for r in results:
        diff_b = abs(r["flat"]["accuracy"] - r["bfs_tree"]["accuracy"])
        diff_m = abs(r["flat"]["accuracy"] - r["poisson_mcts"]["accuracy"])
        candidates.append((r["problem_index"], diff_b, diff_m, r["flat"]["accuracy"]))

    # Select diverse examples
    selected = []
    # Similar accuracy
    similar = sorted(candidates, key=lambda x: x[1] + x[2])
    selected.append(similar[0][0])
    # BFS much better
    bfs_better = sorted(candidates, key=lambda x: -(results[x[0]]["bfs_tree"]["accuracy"] - results[x[0]]["flat"]["accuracy"]))
    for c in bfs_better:
        if c[0] not in selected:
            selected.append(c[0])
            break
    # MCTS much better
    mcts_better = sorted(candidates, key=lambda x: -(results[x[0]]["poisson_mcts"]["accuracy"] - results[x[0]]["flat"]["accuracy"]))
    for c in mcts_better:
        if c[0] not in selected:
            selected.append(c[0])
            break
    # MCTS much worse
    mcts_worse = sorted(candidates, key=lambda x: (results[x[0]]["poisson_mcts"]["accuracy"] - results[x[0]]["flat"]["accuracy"]))
    for c in mcts_worse:
        if c[0] not in selected:
            selected.append(c[0])
            break
    # Mid accuracy
    mid = sorted(candidates, key=lambda x: abs(x[3] - 0.5))
    for c in mid:
        if c[0] not in selected:
            selected.append(c[0])
            break
    # Hard problem
    hard = sorted(candidates, key=lambda x: x[3])
    for c in hard:
        if c[0] not in selected and c[3] > 0:
            selected.append(c[0])
            break

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    for ax_i, pidx in enumerate(selected[:6]):
        ax = axes[ax_i]
        r = results[pidx]
        fa_adv = np.array(r["flat_advantages"])
        ba_adv = np.array(r["bfs_advantages"])
        ma_adv = np.array(r["mcts_advantages"])

        bins = np.linspace(min(fa_adv.min(), ba_adv.min(), ma_adv.min()) - 0.2,
                           max(fa_adv.max(), ba_adv.max(), ma_adv.max()) + 0.2, 25)

        ax.hist(fa_adv, bins=bins, alpha=0.5, color=COLORS["flat"],
                label=f"Flat ({r['flat']['accuracy']:.0%}, n={r['flat']['num_trajectories']})", density=True)
        ax.hist(ba_adv, bins=bins, alpha=0.5, color=COLORS["bfs"],
                label=f"BFS ({r['bfs_tree']['accuracy']:.0%}, n={r['bfs_tree']['num_trajectories']})", density=True)
        ax.hist(ma_adv, bins=bins, alpha=0.5, color=COLORS["mcts"],
                label=f"MCTS ({r['poisson_mcts']['accuracy']:.0%}, n={r['poisson_mcts']['num_trajectories']})", density=True)
        ax.set_title(f"Problem {pidx}", fontsize=12)
        ax.legend(fontsize=7, loc="upper right")
        ax.set_xlabel("GRPO Advantage")

    fig.suptitle("GRPO Advantage Distribution Examples (6 Representative Problems)",
                 fontsize=15, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(f"{out_dir}/advantage_distributions.png", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_dir}/advantage_distributions.png")


def plot_summary_table(results, out_dir):
    """Summary figure with key metrics as a visual table + bar chart."""
    fig = plt.figure(figsize=(15, 7))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1.2])

    # Left: summary table
    ax = fig.add_subplot(gs[0])
    ax.axis("off")

    fa = [r["flat"]["accuracy"] for r in results]
    ba = [r["bfs_tree"]["accuracy"] for r in results]
    ma = [r["poisson_mcts"]["accuracy"] for r in results]
    ft = [r["flat"]["total_tokens"] for r in results]
    bt = [r["bfs_tree"]["total_tokens"] for r in results]
    mt = [r["poisson_mcts"]["total_tokens"] for r in results]
    fn = [r["flat"]["num_trajectories"] for r in results]
    bn = [r["bfs_tree"]["num_trajectories"] for r in results]
    mn = [r["poisson_mcts"]["num_trajectories"] for r in results]
    f_pure = sum(1 for r in results if r["flat"]["adv_std"] < 0.01)
    b_pure = sum(1 for r in results if r["bfs_tree"]["adv_std"] < 0.01)
    m_pure = sum(1 for r in results if r["poisson_mcts"]["adv_std"] < 0.01)

    cell_text = [
        [f"{np.mean(fa):.1%}", f"{np.mean(ba):.1%}", f"{np.mean(ma):.1%}"],
        [f"{np.mean(fn):.0f}", f"{np.mean(bn):.0f}", f"{np.mean(mn):.0f}"],
        [f"{f_pure}", f"{b_pure}", f"{m_pure}"],
        [f"{np.mean(ft)/1000:.0f}K", f"{np.mean(bt)/1000:.0f}K", f"{np.mean(mt)/1000:.0f}K"],
        ["--", f"{np.sum(bt)/np.sum(ft):.1%}", f"{np.sum(mt)/np.sum(ft):.1%}"],
        ["--", f"{pearsonr(fa,ba)[0]:.3f}", f"{pearsonr(fa,ma)[0]:.3f}"],
    ]
    row_labels = ["Mean Accuracy", "Mean #Trajectories", "Pure Problems\n(no signal)",
                  "Mean Tokens", "Token Ratio\n(vs Flat)", "Pearson Corr\n(vs Flat)"]
    col_labels = ["Flat Rollout", "BFS Tree", "Poisson-MCTS"]

    table = ax.table(cellText=cell_text, rowLabels=row_labels, colLabels=col_labels,
                     loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.0, 1.8)

    # Color header
    for j in range(3):
        table[0, j].set_facecolor(list(COLORS.values())[j])
        table[0, j].set_text_props(color="white", fontweight="bold")

    ax.set_title("Key Metrics", fontsize=13, pad=20)

    # Right: accuracy bucket comparison
    ax2 = fig.add_subplot(gs[1])
    bins_edges = [0, 0.01, 0.2, 0.4, 0.6, 0.8, 0.99, 1.01]
    bin_labels = ["0%", "1-19%", "20-39%", "40-59%", "60-79%", "80-99%", "100%"]
    fc = np.histogram(fa, bins=bins_edges)[0]
    bc = np.histogram(ba, bins=bins_edges)[0]
    mc = np.histogram(ma, bins=bins_edges)[0]

    x = np.arange(len(bin_labels))
    w = 0.25
    ax2.bar(x - w, fc, w, color=COLORS["flat"], alpha=0.85, label=LABELS["flat"])
    ax2.bar(x, bc, w, color=COLORS["bfs"], alpha=0.85, label=LABELS["bfs"])
    ax2.bar(x + w, mc, w, color=COLORS["mcts"], alpha=0.85, label=LABELS["mcts"])
    ax2.set_xticks(x)
    ax2.set_xticklabels(bin_labels)
    ax2.set_xlabel("Accuracy Bucket")
    ax2.set_ylabel("# Problems")
    ax2.set_title("Accuracy Distribution by Bucket")
    ax2.legend(fontsize=9)

    fig.suptitle("GRPO Advantage Comparison Summary (100 Problems, step_0)",
                 fontsize=15, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(f"{out_dir}/summary.png", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_dir}/summary.png")


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "poisson_mcts/results/advantage_comparison/step_0/comparison_step_0.json"
    data = load_data(path)
    results = data["results"]
    out_dir = "/".join(path.split("/")[:-1])

    print(f"Loaded {len(results)} problems from {path}")
    print(f"Output dir: {out_dir}\n")

    plot_accuracy_scatter(results, out_dir)
    plot_accuracy_diff_histogram(results, out_dir)
    plot_accuracy_bar(results, out_dir)
    plot_trajectory_counts(results, out_dir)
    plot_token_efficiency(results, out_dir)
    plot_purity_analysis(results, out_dir)
    plot_advantage_distributions(results, out_dir)
    plot_summary_table(results, out_dir)

    print("\nAll plots saved.")


if __name__ == "__main__":
    main()

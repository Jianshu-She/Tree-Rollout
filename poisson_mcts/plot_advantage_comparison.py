#!/usr/bin/env python3
"""Visualize GRPO advantage comparison across 4 methods:
Flat Rollout, BFS Tree, NegBin MCTS, DeepSearch MCTS (arxiv 2509.25454).

Backward-compatible with 3-method JSONs that omit the 'deepsearch' key —
DeepSearch panels are skipped if the data is absent.
"""

import json
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import pearsonr, spearmanr

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "figure.dpi": 150,
    "savefig.dpi": 150,
})

# Method registry: JSON key → display label, color, short name, advantage array key
METHODS = [
    ("flat",         "Flat Rollout", "#4C72B0", "Flat",      "flat_advantages"),
    ("bfs_tree",     "BFS Tree",     "#DD8452", "BFS",       "bfs_advantages"),
    ("poisson_mcts", "NegBin MCTS",  "#55A868", "NegBin",    "mcts_advantages"),
    ("deepsearch",   "DeepSearch",   "#C44E52", "DeepSearch","deepsearch_advantages"),
]


def active_methods(results):
    """Return METHODS list filtered to those actually present in results."""
    present = set()
    for r in results:
        present.update(r.keys())
    return [m for m in METHODS if m[0] in present]


def get_accs(results, key):
    return np.array([r[key]["accuracy"] for r in results])


def get_stds(results, key):
    return np.array([r[key]["adv_std"] for r in results])


def get_tokens(results, key):
    return np.array([r[key]["total_tokens"] for r in results])


def get_n_traj(results, key):
    return np.array([r[key]["num_trajectories"] for r in results])


def split_no_adv(results, key):
    """Return (all_correct_mask, all_wrong_mask) for no-advantage problems."""
    accs = get_accs(results, key)
    no_adv = get_stds(results, key) < 0.01
    return no_adv & (accs > 0.99), no_adv & (accs < 0.01)


def load_data(path):
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------
# Plot 1: Per-problem accuracy scatter vs flat (one subplot per tree method)
# ---------------------------------------------------------------------
def plot_accuracy_scatter(results, out_dir):
    methods = [m for m in active_methods(results) if m[0] != "flat"]
    n = len(methods)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 6))
    if n == 1:
        axes = [axes]

    fa = get_accs(results, "flat")
    for ax, (key, label, color, _, _) in zip(axes, methods):
        ma = get_accs(results, key)
        mn = get_n_traj(results, key)
        sc = ax.scatter(fa, ma, c=mn, cmap="viridis", s=40, alpha=0.8,
                        edgecolors="w", linewidths=0.3)
        ax.plot([0, 1], [0, 1], "k--", alpha=0.3, lw=1)
        pr, _ = pearsonr(fa, ma)
        sr, _ = spearmanr(fa, ma)
        ax.set_xlabel("Flat Rollout Accuracy")
        ax.set_ylabel(f"{label} Accuracy")
        ax.set_title(f"Flat vs {label}\nPearson={pr:.3f}, Spearman={sr:.3f}")
        cb = fig.colorbar(sc, ax=ax, shrink=0.8)
        cb.set_label(f"{label} #trajectories")
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.set_aspect("equal")

    fig.suptitle("Per-Problem Accuracy: Tree Methods vs Flat Rollout",
                 fontsize=15, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(f"{out_dir}/accuracy_scatter.png", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_dir}/accuracy_scatter.png")


# ---------------------------------------------------------------------
# Plot 2: Accuracy diff histograms (method − flat)
# ---------------------------------------------------------------------
def plot_accuracy_diff_histogram(results, out_dir):
    methods = [m for m in active_methods(results) if m[0] != "flat"]
    n = len(methods)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5.5))
    if n == 1:
        axes = [axes]

    fa = get_accs(results, "flat")
    bins = np.arange(-0.55, 0.6, 0.05)

    for ax, (key, label, color, _, _) in zip(axes, methods):
        ma = get_accs(results, key)
        diff = ma - fa
        ax.hist(diff, bins=bins, color=color, alpha=0.8, edgecolor="white")
        ax.axvline(0, color="black", ls="--", lw=1, alpha=0.5)
        ax.axvline(np.mean(diff), color="red", ls="-", lw=1.5,
                   label=f"mean={np.mean(diff):+.1%}")
        ax.set_xlabel(f"{label} Accuracy − Flat Accuracy")
        ax.set_ylabel("Count")
        ax.set_title(f"{label} vs Flat (σ={np.std(diff):.2f})")
        ax.legend()

    fig.suptitle("Per-Problem Accuracy Difference Distribution",
                 fontsize=15, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(f"{out_dir}/accuracy_diff_histogram.png", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_dir}/accuracy_diff_histogram.png")


# ---------------------------------------------------------------------
# Plot 3: Grouped bar chart sorted by flat accuracy
# ---------------------------------------------------------------------
def plot_accuracy_bar(results, out_dir):
    methods = active_methods(results)
    n = len(methods)
    fa = get_accs(results, "flat")
    order = np.argsort(fa)
    idx = np.array([r["problem_index"] for r in results])[order]

    accs_by_method = []
    for key, _, _, _, _ in methods:
        accs_by_method.append(get_accs(results, key)[order])

    fig, ax = plt.subplots(figsize=(18, 6))
    x = np.arange(len(results))
    w = 0.8 / n
    offsets = np.linspace(-(n - 1) / 2, (n - 1) / 2, n) * w
    for i, ((key, label, color, _, _), accs) in enumerate(zip(methods, accs_by_method)):
        ax.bar(x + offsets[i], accs, w, color=color, alpha=0.85, label=label)
    ax.set_xlabel("Problems (sorted by Flat accuracy)")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"Per-Problem Accuracy Comparison ({len(results)} Problems)",
                 fontsize=15, fontweight="bold")
    ax.legend(loc="upper left")
    ax.set_xticks(x[::5])
    ax.set_xticklabels([f"p{i}" for i in idx[::5]], fontsize=8, rotation=45)
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    fig.savefig(f"{out_dir}/accuracy_bar_sorted.png", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_dir}/accuracy_bar_sorted.png")


# ---------------------------------------------------------------------
# Plot 4: Trajectory count distributions (tree methods only)
# ---------------------------------------------------------------------
def plot_trajectory_counts(results, out_dir):
    methods = [m for m in active_methods(results) if m[0] != "flat"]
    n = len(methods)
    fig, axes = plt.subplots(1, n, figsize=(5.5 * n, 5.5))
    if n == 1:
        axes = [axes]

    for ax, (key, label, color, _, _) in zip(axes, methods):
        counts = get_n_traj(results, key)
        bins = np.arange(0, 135, 5)
        ax.hist(counts, bins=bins, color=color, alpha=0.8, edgecolor="white")
        ax.axvline(128, color="red", ls="--", lw=1.5, label="Target=128")
        ax.axvline(counts.mean(), color="black", ls="-", lw=1.5,
                   label=f"mean={counts.mean():.0f}")
        ax.set_xlabel("#Trajectories")
        ax.set_ylabel("Count")
        ax.set_title(f"{label}\n(mean={counts.mean():.0f}, range={counts.min()}-{counts.max()})")
        ax.legend()

    fig.suptitle("Number of Complete Trajectories (Target: 128)",
                 fontsize=15, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(f"{out_dir}/trajectory_counts.png", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_dir}/trajectory_counts.png")


# ---------------------------------------------------------------------
# Plot 5: Token efficiency (boxplot + per-problem ratio)
# ---------------------------------------------------------------------
def plot_token_efficiency(results, out_dir):
    methods = active_methods(results)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    tokens_by_method = [get_tokens(results, m[0]) for m in methods]
    ft = get_tokens(results, "flat")

    # Box plot
    ax = axes[0]
    data = [arr / 1000 for arr in tokens_by_method]
    short_names = [m[3] for m in methods]
    colors = [m[2] for m in methods]
    bp = ax.boxplot(data, tick_labels=short_names, patch_artist=True, widths=0.55)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_ylabel("Total Tokens (K)")
    ax.set_title("Token Cost per Problem")
    ax.set_yscale("log")

    # Ratio vs flat
    ax = axes[1]
    x = np.arange(len(results))
    for (key, label, color, short, _), tok in zip(methods, tokens_by_method):
        if key == "flat":
            continue
        ratio = tok / ft
        ax.scatter(x, ratio, s=18, alpha=0.6, color=color,
                   label=f"{short}/Flat (mean={ratio.mean():.1%})")
    ax.axhline(1.0, color="black", ls="--", lw=1, alpha=0.3)
    ax.set_xlabel("Problem Index")
    ax.set_ylabel("Token Ratio (Tree / Flat)")
    ax.set_title("Token Savings per Problem")
    ax.legend(fontsize=9)

    fig.suptitle("Compute Efficiency: Tree Methods vs Flat Rollout",
                 fontsize=15, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(f"{out_dir}/token_efficiency.png", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_dir}/token_efficiency.png")


# ---------------------------------------------------------------------
# Plot 6: No-advantage split (all-correct vs all-wrong)
# ---------------------------------------------------------------------
def plot_no_advantage_analysis(results, out_dir):
    methods = active_methods(results)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    fa = get_accs(results, "flat")
    splits = {m[0]: split_no_adv(results, m[0]) for m in methods}

    # Panel 1: stacked all-correct (green) + all-wrong (red) per method
    ax = axes[0]
    short_names = [m[3] for m in methods]
    correct_counts = [int(splits[m[0]][0].sum()) for m in methods]
    wrong_counts = [int(splits[m[0]][1].sum()) for m in methods]
    x = np.arange(len(methods))
    ax.bar(x, correct_counts, color="#4CAF50", alpha=0.85,
           label="All correct (model solved ✓)", edgecolor="white")
    ax.bar(x, wrong_counts, bottom=correct_counts, color="#E53935", alpha=0.85,
           label="All wrong (model failed ✗)", edgecolor="white")
    for i, (c, w) in enumerate(zip(correct_counts, wrong_counts)):
        if c > 0:
            ax.text(i, c / 2, str(c), ha="center", va="center",
                    fontsize=11, fontweight="bold", color="white")
        if w > 0:
            ax.text(i, c + w / 2, str(w), ha="center", va="center",
                    fontsize=11, fontweight="bold", color="white")
        ax.text(i, c + w + 0.5, f"total: {c + w}", ha="center", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(short_names)
    ax.set_ylabel("# No-Advantage Problems")
    ax.set_title("No-Advantage: Good vs Bad")
    ax.legend(fontsize=9, loc="upper left")
    max_total = max(c + w for c, w in zip(correct_counts, wrong_counts)) or 5
    ax.set_ylim(0, max_total * 1.4)

    # Panel 2: newly all-correct vs newly all-wrong per tree method
    ax = axes[1]
    f_corr, f_wrong = splits["flat"]
    cats, vals, colors_bar = [], [], []
    for key, label, color, short, _ in methods:
        if key == "flat":
            continue
        corr, wrong = splits[key]
        cats.extend([f"{short}\nnewly\nall-correct", f"{short}\nnewly\nall-wrong"])
        vals.extend([int((corr & ~f_corr).sum()), int((wrong & ~f_wrong).sum())])
        colors_bar.extend(["#4CAF50", "#E53935"])
    bars = ax.bar(cats, vals, color=colors_bar, alpha=0.85, edgecolor="white")
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                str(v), ha="center", fontsize=11, fontweight="bold")
    ax.set_ylabel("# Problems")
    ax.set_title("Tree vs Flat: Outcome Divergence")
    ax.set_ylim(0, (max(vals) if vals else 1) * 1.3)

    # Panel 3: histogram of flat accuracy on truly-bad cases
    ax = axes[2]
    bins = np.arange(0, 1.05, 0.1)
    for key, label, color, short, _ in methods:
        if key == "flat":
            continue
        _, wrong = splits[key]
        newly_wrong = wrong & ~f_wrong
        if newly_wrong.sum() > 0:
            ax.hist(fa[newly_wrong], bins=bins, alpha=0.5, color=color,
                    label=f"{short} now all-wrong ({int(newly_wrong.sum())})",
                    edgecolor="white")
    ax.set_xlabel("Flat Rollout Accuracy")
    ax.set_ylabel("Count")
    ax.set_title("Truly Bad Cases: Flat Accuracy of\nTree's newly-all-wrong problems")
    ax.legend(fontsize=9)

    fig.suptitle("No-Advantage Analysis: All-Correct (Good) vs All-Wrong (Bad)",
                 fontsize=15, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(f"{out_dir}/no_advantage_analysis.png", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_dir}/no_advantage_analysis.png")


# ---------------------------------------------------------------------
# Plot 7: Advantage distributions on 6 representative problems
# ---------------------------------------------------------------------
def plot_advantage_distributions(results, out_dir):
    methods = active_methods(results)
    tree_methods = [m for m in methods if m[0] != "flat"]

    # Pick 6 representative problems
    N = len(results)
    idx_by_problem = {r["problem_index"]: i for i, r in enumerate(results)}
    diffs = [sum(abs(r["flat"]["accuracy"] - r[tk]["accuracy"])
                 for tk, *_ in tree_methods) for r in results]
    similar = np.argsort(diffs)
    selected = [results[similar[0]]["problem_index"]]

    for tk, label, *_ in tree_methods:
        better = np.argsort([-(results[i][tk]["accuracy"] - results[i]["flat"]["accuracy"])
                             for i in range(N)])
        for bi in better:
            pidx = results[bi]["problem_index"]
            if pidx not in selected:
                selected.append(pidx)
                break

    mid = np.argsort([abs(r["flat"]["accuracy"] - 0.5) for r in results])
    for mi in mid:
        pidx = results[mi]["problem_index"]
        if pidx not in selected:
            selected.append(pidx)
            break

    hard = np.argsort([r["flat"]["accuracy"] for r in results])
    for hi in hard:
        pidx = results[hi]["problem_index"]
        if pidx not in selected and results[hi]["flat"]["accuracy"] > 0:
            selected.append(pidx)
            break

    selected = selected[:6]

    fig, axes = plt.subplots(2, 3, figsize=(17, 10))
    axes = axes.flatten()

    for ax_i, pidx in enumerate(selected):
        ax = axes[ax_i]
        r = results[idx_by_problem[pidx]]

        advs_by_method = []
        for key, label, color, short, adv_key in methods:
            if adv_key not in r:
                continue
            arr = np.array(r[adv_key])
            advs_by_method.append((short, color, label, arr, r[key]))

        if not advs_by_method:
            continue

        all_vals = np.concatenate([a for _, _, _, a, _ in advs_by_method])
        bins = np.linspace(all_vals.min() - 0.2, all_vals.max() + 0.2, 25)

        for short, color, label, arr, stats in advs_by_method:
            ax.hist(arr, bins=bins, alpha=0.45, color=color, density=True,
                    label=f"{short} ({stats['accuracy']:.0%}, n={stats['num_trajectories']})")
        ax.set_title(f"Problem {pidx}", fontsize=12)
        ax.legend(fontsize=7, loc="upper right")
        ax.set_xlabel("GRPO Advantage")

    for ax in axes[len(selected):]:
        ax.axis("off")

    fig.suptitle("GRPO Advantage Distribution Examples (Representative Problems)",
                 fontsize=15, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(f"{out_dir}/advantage_distributions.png", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_dir}/advantage_distributions.png")


# ---------------------------------------------------------------------
# Plot 8: Summary table + accuracy bucket bars
# ---------------------------------------------------------------------
def plot_summary_table(results, out_dir):
    methods = active_methods(results)
    n = len(methods)
    fig = plt.figure(figsize=(7 + 3 * n, 7))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1.2])

    ax = fig.add_subplot(gs[0])
    ax.axis("off")

    ft = get_tokens(results, "flat")
    fa_list = get_accs(results, "flat")
    pearsons = {}
    for key, *_ in methods:
        if key == "flat":
            continue
        ma = get_accs(results, key)
        pearsons[key] = pearsonr(fa_list, ma)[0]

    rows = {
        "Mean Accuracy": [],
        "Mean #Trajectories": [],
        "All-Correct ✓\n(model solved)": [],
        "All-Wrong ✗\n(model failed)": [],
        "Mean Tokens": [],
        "Token Ratio\n(vs Flat)": [],
        "Pearson Corr\n(vs Flat)": [],
    }
    for key, label, color, short, _ in methods:
        accs = get_accs(results, key)
        n_traj = get_n_traj(results, key)
        toks = get_tokens(results, key)
        correct, wrong = split_no_adv(results, key)
        rows["Mean Accuracy"].append(f"{accs.mean():.1%}")
        rows["Mean #Trajectories"].append(f"{n_traj.mean():.0f}")
        rows["All-Correct ✓\n(model solved)"].append(f"{int(correct.sum())}")
        rows["All-Wrong ✗\n(model failed)"].append(f"{int(wrong.sum())}")
        rows["Mean Tokens"].append(f"{toks.mean()/1000:.0f}K")
        if key == "flat":
            rows["Token Ratio\n(vs Flat)"].append("--")
            rows["Pearson Corr\n(vs Flat)"].append("--")
        else:
            rows["Token Ratio\n(vs Flat)"].append(f"{toks.sum()/ft.sum():.1%}")
            rows["Pearson Corr\n(vs Flat)"].append(f"{pearsons[key]:.3f}")

    row_labels = list(rows.keys())
    col_labels = [m[1] for m in methods]
    cell_text = [rows[rl] for rl in row_labels]

    table = ax.table(cellText=cell_text, rowLabels=row_labels, colLabels=col_labels,
                     loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.0, 1.8)
    for j, m in enumerate(methods):
        table[0, j].set_facecolor(m[2])
        table[0, j].set_text_props(color="white", fontweight="bold")
    ax.set_title("Key Metrics", fontsize=13, pad=20)

    # Right: accuracy bucket bars
    ax2 = fig.add_subplot(gs[1])
    bins_edges = [0, 0.01, 0.2, 0.4, 0.6, 0.8, 0.99, 1.01]
    bin_labels = ["0%", "1-19%", "20-39%", "40-59%", "60-79%", "80-99%", "100%"]
    x = np.arange(len(bin_labels))
    w = 0.8 / n
    offsets = np.linspace(-(n - 1) / 2, (n - 1) / 2, n) * w
    for i, (key, label, color, _, _) in enumerate(methods):
        accs = get_accs(results, key)
        counts = np.histogram(accs, bins=bins_edges)[0]
        ax2.bar(x + offsets[i], counts, w, color=color, alpha=0.85, label=label)
    ax2.set_xticks(x)
    ax2.set_xticklabels(bin_labels)
    ax2.set_xlabel("Accuracy Bucket")
    ax2.set_ylabel("# Problems")
    ax2.set_title("Accuracy Distribution by Bucket")
    ax2.legend(fontsize=9)

    fig.suptitle(f"GRPO Advantage Comparison Summary ({len(results)} Problems)",
                 fontsize=15, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(f"{out_dir}/summary.png", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_dir}/summary.png")


# ---------------------------------------------------------------------
# NEW Plot 9: Pareto with Flat Rollout as the compute reference
# ---------------------------------------------------------------------
def plot_pareto(results, out_dir):
    """Pareto scatter with Flat Rollout as the compute reference.

    X axis: compute as % of Flat Rollout compute (log scale, Flat = 100%)
    Y axis: mean accuracy across 100 problems
    Each point = one method; vertical/horizontal lines through Flat show
    the "dominance quadrant" — any method upper-left of Flat is strictly
    better (higher acc, lower compute).
    """
    methods = active_methods(results)
    flat_tokens = get_tokens(results, "flat").mean()
    flat_acc = get_accs(results, "flat").mean()

    fig, ax = plt.subplots(figsize=(11, 7))

    # Light per-problem scatter in the background (small alpha)
    for key, label, color, short, _ in methods:
        toks = get_tokens(results, key)
        accs = get_accs(results, key)
        compute_pct = 100.0 * toks / flat_tokens
        ax.scatter(compute_pct, accs, s=18, alpha=0.15, color=color,
                   edgecolors="none")

    # Main: mean point per method + error bars (per-problem std)
    for key, label, color, short, _ in methods:
        toks = get_tokens(results, key)
        accs = get_accs(results, key)
        compute_pct = 100.0 * toks / flat_tokens
        mean_c = compute_pct.mean()
        mean_a = accs.mean()
        std_a = accs.std() / np.sqrt(len(accs))  # SEM

        ax.errorbar(
            mean_c, mean_a, yerr=std_a, fmt="o",
            color=color, markersize=18, mew=2.5, capsize=6,
            markerfacecolor=color, markeredgecolor="white",
            ecolor=color, zorder=5,
        )
        # Annotate next to the point — manual offsets to avoid overlap
        offset_map = {
            "flat":         (14,  -4, "left",  "center"),
            "bfs_tree":     (14,  -20, "left",  "top"),
            "poisson_mcts": (14,  20, "left",  "bottom"),
            "deepsearch":   (14,  4, "left",  "center"),
        }
        offset_x, offset_y, ha, va = offset_map.get(key, (8, 0, "left", "center"))
        ax.annotate(
            f"{label}\nacc={mean_a:.1%}, compute={mean_c:.1f}%",
            xy=(mean_c, mean_a), xytext=(offset_x, offset_y),
            textcoords="offset points", fontsize=10, ha=ha, va=va,
            color=color, fontweight="bold",
        )

    # Reference lines through Flat
    ax.axhline(flat_acc, color="#4C72B0", ls="--", lw=1.2, alpha=0.6)
    ax.axvline(100.0, color="#4C72B0", ls="--", lw=1.2, alpha=0.6)

    # Quadrant labels: upper-left = dominates flat; rest labels explain
    ax.text(5.5, 0.68, "DOMINATES flat\n(better + cheaper)",
            fontsize=10, ha="center", va="center",
            color="#2E7D32", fontweight="bold",
            bbox=dict(facecolor="#E8F5E9", edgecolor="#2E7D32",
                      boxstyle="round,pad=0.3", alpha=0.8))
    ax.text(5.5, 0.3, "same compute,\nlower acc",
            fontsize=9, ha="center", va="center",
            color="#757575", alpha=0.7)
    ax.text(300, 0.68, "worse cost /\nacc trade-off",
            fontsize=9, ha="center", va="center",
            color="#757575", alpha=0.7)
    ax.text(300, 0.3, "strictly worse than flat\n(worse + more expensive)",
            fontsize=9, ha="center", va="center",
            color="#C62828", alpha=0.7)

    ax.set_xscale("log")
    ax.set_xlabel("Compute (% of Flat Rollout)", fontsize=12)
    ax.set_ylabel("Mean Accuracy (100 problems)", fontsize=12)
    ax.set_title(
        "Pareto Frontier: Accuracy vs Compute (Flat Rollout = 100% reference)\n"
        "Error bars = SEM; faded dots = per-problem observations",
        fontsize=13, fontweight="bold",
    )
    ax.grid(True, alpha=0.3, which="both")
    ax.set_xlim(3, 400)
    ax.set_ylim(0.25, 0.75)
    fig.tight_layout()
    fig.savefig(f"{out_dir}/pareto_accuracy_vs_tokens.png", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_dir}/pareto_accuracy_vs_tokens.png")


# ---------------------------------------------------------------------
# NEW Plot 10: Drift correlation matrix (method × method Pearson)
# ---------------------------------------------------------------------
def plot_drift_matrix(results, out_dir):
    """Pairwise per-problem accuracy Pearson correlation across methods.
    Faithful methods should be close to 1.0 with Flat; "drift" methods
    lower correlation = more deviation from natural flat behavior."""
    methods = active_methods(results)
    n = len(methods)
    mat = np.zeros((n, n))
    accs_all = [get_accs(results, m[0]) for m in methods]
    for i in range(n):
        for j in range(n):
            mat[i, j] = pearsonr(accs_all[i], accs_all[j])[0] if i != j else 1.0

    fig, ax = plt.subplots(figsize=(1.6 * n + 2, 1.6 * n + 2))
    im = ax.imshow(mat, cmap="RdYlGn", vmin=0.5, vmax=1.0)
    labels = [m[3] for m in methods]
    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{mat[i, j]:.3f}", ha="center", va="center",
                    color="black" if mat[i, j] > 0.75 else "white",
                    fontsize=12, fontweight="bold")
    fig.colorbar(im, ax=ax, shrink=0.8, label="Pearson r")
    ax.set_title("Per-Problem Accuracy Correlation\n"
                 "(lower = method drifts further from flat rollout behavior)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(f"{out_dir}/drift_correlation_matrix.png", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_dir}/drift_correlation_matrix.png")


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else \
        "poisson_mcts/results/advantage_comparison/step_0/comparison_step_0.json"
    data = load_data(path)
    results = data["results"]
    out_dir = "/".join(path.split("/")[:-1])

    print(f"Loaded {len(results)} problems from {path}")
    print(f"Active methods: {[m[0] for m in active_methods(results)]}")
    print(f"Output dir: {out_dir}\n")

    plot_accuracy_scatter(results, out_dir)
    plot_accuracy_diff_histogram(results, out_dir)
    plot_accuracy_bar(results, out_dir)
    plot_trajectory_counts(results, out_dir)
    plot_token_efficiency(results, out_dir)
    plot_no_advantage_analysis(results, out_dir)
    plot_advantage_distributions(results, out_dir)
    plot_summary_table(results, out_dir)
    plot_pareto(results, out_dir)
    plot_drift_matrix(results, out_dir)

    print("\nAll plots saved.")


if __name__ == "__main__":
    main()

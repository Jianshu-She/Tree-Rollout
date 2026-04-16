#!/usr/bin/env python3
"""Ablation: decompose BFS no-advantage rate into n effect vs tree correlation.

Subsample flat trajectories to BFS's actual n per problem, then compare
no-advantage rates. The difference between flat@same_n and BFS reveals
how much of the no-advantage gap is due to tree correlation (shared
prefixes) vs simply having fewer samples.

Key finding: ~64% of BFS's no-advantage excess comes from reduced n
(a necessary consequence of compute savings) and ~36% from tree
correlation (shared prefixes cause correlated outcomes).
"""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

STAGES = ["step_0", "step_40", "step_80", "step_120"]
STAGE_X = [0, 40, 80, 120]
N_BOOT = 50


def recover_rewards(advantages, accuracy):
    adv = np.asarray(advantages, dtype=float)
    if len(adv) == 0:
        return np.array([])
    if len(np.unique(adv)) == 1:
        return np.ones_like(adv) if accuracy > 0.5 else np.zeros_like(adv)
    return (adv > 0).astype(float)


def run_ablation():
    rng = np.random.default_rng(42)
    rows = []

    for stage in STAGES:
        with open(f"poisson_mcts/results/advantage_comparison/{stage}/comparison_{stage}.json") as f:
            results = json.load(f)["results"]

        bfs_ns = [r["bfs_tree"]["num_trajectories"] for r in results]
        bfs_no_adv = sum(1 for r in results if r["bfs_tree"]["adv_std"] < 0.01)
        flat_no_adv = sum(1 for r in results if r["flat"]["adv_std"] < 0.01)
        mean_n = int(np.mean(bfs_ns))

        flat_sub_counts = []
        for _ in range(N_BOOT):
            count = 0
            for i, r in enumerate(results):
                rewards = recover_rewards(r["flat_advantages"], r["flat"]["accuracy"])
                n_sub = min(bfs_ns[i], len(rewards))
                idx = rng.choice(len(rewards), size=n_sub, replace=False)
                if rewards[idx].std() < 0.01:
                    count += 1
            flat_sub_counts.append(count)

        flat_sub_mean = np.mean(flat_sub_counts)
        flat_sub_std = np.std(flat_sub_counts)

        n_effect = flat_sub_mean - flat_no_adv
        corr_effect = bfs_no_adv - flat_sub_mean
        total = bfs_no_adv - flat_no_adv

        rows.append({
            "stage": stage, "mean_bfs_n": mean_n,
            "flat_128": flat_no_adv, "flat_sub": flat_sub_mean,
            "flat_sub_std": flat_sub_std, "bfs": bfs_no_adv,
            "n_effect": n_effect, "corr_effect": corr_effect, "total": total,
            "n_pct": n_effect / max(total, 1) * 100,
            "corr_pct": corr_effect / max(total, 1) * 100,
        })

    return rows


def plot_decomposition(rows, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    stages = [r["stage"] for r in rows]
    flat_128 = [r["flat_128"] for r in rows]
    flat_sub = [r["flat_sub"] for r in rows]
    bfs = [r["bfs"] for r in rows]

    # Panel 1: stacked bar (n effect + correlation effect)
    ax = axes[0]
    x = np.arange(len(stages))
    ax.bar(x, flat_128, color="#4C72B0", alpha=0.85, label="Flat (n=128) baseline")
    ax.bar(x, [r["n_effect"] for r in rows], bottom=flat_128,
           color="#F39C12", alpha=0.85, label="n effect (subsample gap)")
    ax.bar(x, [r["corr_effect"] for r in rows],
           bottom=[f + n for f, n in zip(flat_128, [r["n_effect"] for r in rows])],
           color="#E74C3C", alpha=0.85, label="Tree correlation effect")
    ax.set_xticks(x)
    ax.set_xticklabels(stages)
    ax.set_ylabel("% No-Advantage Problems")
    ax.set_title("BFS No-Advantage Decomposition")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    # Panel 2: percentage breakdown
    ax = axes[1]
    n_pcts = [r["n_pct"] for r in rows]
    c_pcts = [r["corr_pct"] for r in rows]
    ax.bar(x, n_pcts, color="#F39C12", alpha=0.85, label="n effect")
    ax.bar(x, c_pcts, bottom=n_pcts, color="#E74C3C", alpha=0.85,
           label="Tree correlation effect")
    for i in range(len(stages)):
        ax.text(i, n_pcts[i] / 2, f"{n_pcts[i]:.0f}%", ha="center",
                fontsize=11, fontweight="bold", color="white")
        ax.text(i, n_pcts[i] + c_pcts[i] / 2, f"{c_pcts[i]:.0f}%",
                ha="center", fontsize=11, fontweight="bold", color="white")
    ax.set_xticks(x)
    ax.set_xticklabels(stages)
    ax.set_ylabel("% of Total Gap (BFS − Flat@128)")
    ax.set_title("What Causes BFS No-Advantage?")
    ax.legend(fontsize=9)
    ax.set_ylim(0, 110)
    ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle(
        "Ablation: Decomposing BFS No-Advantage into n Effect vs Tree Correlation\n"
        "n effect = subsampling flat to BFS's n; correlation = shared-prefix structure",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def main():
    rows = run_ablation()

    print("=" * 80)
    print("N-MATCHED ABLATION: BFS No-Advantage Decomposition")
    print("=" * 80)
    for r in rows:
        print(f"{r['stage']} (BFS n≈{r['mean_bfs_n']}):")
        print(f"  Flat (n=128):       {r['flat_128']}%")
        print(f"  Flat (n≈{r['mean_bfs_n']}):       {r['flat_sub']:.1f}% ± {r['flat_sub_std']:.1f}")
        print(f"  BFS  (n≈{r['mean_bfs_n']}):       {r['bfs']}%")
        print(f"  Gap: {r['total']:.0f}pp = {r['n_effect']:.0f}pp n-effect ({r['n_pct']:.0f}%)"
              f" + {r['corr_effect']:.0f}pp correlation ({r['corr_pct']:.0f}%)")
        print()

    out_path = "poisson_mcts/results/advantage_comparison/ablation_n_vs_correlation.png"
    plot_decomposition(rows, out_path)


if __name__ == "__main__":
    main()

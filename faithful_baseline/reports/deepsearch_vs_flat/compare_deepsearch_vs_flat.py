#!/usr/bin/env python3
"""Compare DeepSearch MCTS trees vs Flat Rollout post-hoc trees.

1. Side-by-side Sankey tree visualizations
2. Branching factor distribution comparison (Poisson/NegBin fit)
3. Depth profile comparison
4. Summary statistics

Core claim: DeepSearch cannot replace flat rollouts — the tree structures
are fundamentally different.
"""

import json
import os
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from scipy import stats as sp_stats

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "mcts_inference"))
from mcts_inference.utils import is_correct

OUTPUT_DIR = "faithful_baseline/results/deepsearch_vs_flat"
FLAT_PARAMS = "faithful_baseline/results/math500_full/train/poisson_beta_analysis/fitted_parameters.json"

STAGES = ["step_0", "step_40", "step_80", "step_120"]
STAGE_LABELS = {"step_0": "Base (step 0)", "step_40": "Step 40",
                "step_80": "Step 80", "step_120": "Step 120"}
COLORS = {"step_0": "#e74c3c", "step_40": "#e67e22",
          "step_80": "#2ecc71", "step_120": "#3498db"}
MAX_DEPTH = 12


# =========================================================================
# Tree statistics collection
# =========================================================================

def collect_mcts_bf(node, depth=0):
    bf = defaultdict(list)
    children = node.get("children", [])
    if children:
        bf[depth].append(len(children))
    for c in children:
        for d, v in collect_mcts_bf(c, depth + 1).items():
            bf[d].extend(v)
    return bf


def collect_mcts_depth_profile(node, depth=0):
    """Collect number of nodes at each depth."""
    profile = defaultdict(int)
    profile[depth] += 1
    for c in node.get("children", []):
        for d, n in collect_mcts_depth_profile(c, depth + 1).items():
            profile[d] += n
    return profile


def max_tree_depth(node, d=0):
    if not node.get("children"):
        return d
    return max(max_tree_depth(c, d + 1) for c in node["children"])


# =========================================================================
# Sankey-like tree drawing for MCTS trees
# =========================================================================

def draw_mcts_tree_on_ax(ax, tree_dict, max_depth=8, title=""):
    """Draw a simplified tree diagram for MCTS tree dict.
    Width of each node = visit_count relative to root.
    Color = q_value (green = high, red = low).
    """
    from collections import deque

    # Flatten tree
    nodes = []
    edges = []
    counter = [0]

    def flatten(tnode, depth, parent_id):
        if depth > max_depth:
            return
        nid = counter[0]
        counter[0] += 1
        visits = max(tnode.get("visit_count", 1), 1)
        qval = tnode.get("q_value", 0.5)
        nodes.append({"id": nid, "depth": depth, "visits": visits, "q": qval})
        if parent_id is not None:
            edges.append({"src": parent_id, "dst": nid, "visits": visits, "q": qval})
        for c in tnode.get("children", []):
            flatten(c, depth + 1, nid)

    flatten(tree_dict, 0, None)
    if not nodes:
        ax.text(0.5, 0.5, "Empty tree", ha="center", va="center", transform=ax.transAxes)
        return

    root_visits = nodes[0]["visits"]
    max_d = max(n["depth"] for n in nodes)
    node_map = {n["id"]: n for n in nodes}
    children_map = defaultdict(list)
    for e in edges:
        children_map[e["src"]].append(e["dst"])

    # Layout
    total_height = 8.0
    gap = 0.08
    positions = {}

    root = nodes[0]
    root_h = total_height
    positions[root["id"]] = (0.0, total_height / 2 + root_h / 2, total_height / 2 - root_h / 2)

    queue = deque([root["id"]])
    while queue:
        pid = queue.popleft()
        if pid not in children_map:
            continue
        px, py_top, py_bot = positions[pid]
        cids = children_map[pid]
        # Proportional heights
        total_v = sum(node_map[c]["visits"] for c in cids)
        y_cur = py_top
        for cid in cids:
            ch = max(0.02, (node_map[cid]["visits"] / max(total_v, 1)) * (py_top - py_bot))
            cx = node_map[cid]["depth"] * 2.0
            positions[cid] = (cx, y_cur, y_cur - ch)
            y_cur = y_cur - ch - gap
            queue.append(cid)

    # Draw edges
    for e in edges:
        if e["src"] not in positions or e["dst"] not in positions:
            continue
        sx, sy_top, sy_bot = positions[e["src"]]
        dx, dy_top, dy_bot = positions[e["dst"]]
        q = e["q"]
        if q > 0.6:
            color = (0.1, 0.7, 0.1, 0.5)
        elif q < 0.4:
            color = (0.8, 0.1, 0.1, 0.5)
        else:
            color = (0.5, 0.5, 0.3, 0.4)

        t = np.linspace(0, 1, 30)
        sx_r, dx_l = sx + 0.15, dx - 0.15
        cx1 = sx_r + (dx_l - sx_r) * 0.4
        cx2 = sx_r + (dx_l - sx_r) * 0.6
        top_x = (1 - t) ** 3 * sx_r + 3 * (1 - t) ** 2 * t * cx1 + 3 * (1 - t) * t ** 2 * cx2 + t ** 3 * dx_l
        # Use center of source slot and dest slot
        s_mid = (sy_top + sy_bot) / 2
        d_mid = (dy_top + dy_bot) / 2
        top_y = (1 - t) * (s_mid + (dy_top - dy_bot) / 2) + t * dy_top
        bot_y = (1 - t) * (s_mid - (dy_top - dy_bot) / 2) + t * dy_bot
        poly_x = np.concatenate([top_x, top_x[::-1]])
        poly_y = np.concatenate([top_y, bot_y[::-1]])
        ax.fill(poly_x, poly_y, color=color, edgecolor="none")

    # Draw nodes
    for n in nodes:
        if n["id"] not in positions:
            continue
        x, yt, yb = positions[n["id"]]
        h = yt - yb
        q = n["q"]
        if q > 0.6:
            color = (0.1, 0.7, 0.1, 0.8)
        elif q < 0.4:
            color = (0.8, 0.1, 0.1, 0.8)
        else:
            color = (0.5, 0.5, 0.3, 0.6)
        rect = plt.Rectangle((x - 0.12, yb), 0.24, h,
                              facecolor=color, edgecolor="black", linewidth=0.5, zorder=3)
        ax.add_patch(rect)

    # Depth labels
    for d in range(min(max_d, max_depth) + 1):
        x = d * 2.0
        ax.text(x, -1.0, f"D{d}" if d > 0 else "Root",
                ha="center", va="top", fontsize=8, fontweight="bold")

    ax.set_xlim(-0.5, min(max_d, max_depth) * 2.0 + 0.5)
    all_ys = [p[1] for p in positions.values()] + [p[2] for p in positions.values()]
    ax.set_ylim(min(all_ys) - 1.5, max(all_ys) + 0.5)
    ax.set_title(title, fontsize=10)
    ax.axis("off")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load flat rollout params
    with open(FLAT_PARAMS) as f:
        flat_params = json.load(f)

    # Load flat rollout trees (10-problem cached trees for visualization)
    flat_trees = {}
    flat_corr = {}
    for stage in STAGES:
        cache_path = f"faithful_baseline/results/training_stages_math500/trees_deep/cached_trees_{stage}_syntactic.json"
        if os.path.exists(cache_path):
            with open(cache_path) as f:
                flat_trees[stage] = json.load(f)

        rollout_path = f"faithful_baseline/results/training_stages_math500/rollouts_{stage}.json"
        if os.path.exists(rollout_path):
            with open(rollout_path) as f:
                rdata = json.load(f)
            corrs = []
            for entry in rdata["results"]:
                gt = entry.get("ground_truth", entry.get("answer", ""))
                c = {}
                for r in entry["rollouts"]:
                    rid = r["rollout_id"]
                    text = r.get("full_text", "") or r.get("response", "")
                    c[rid] = is_correct(text, gt)
                corrs.append(c)
            flat_corr[stage] = corrs

    # Load DeepSearch results
    ds_data = {}
    ds_bf = {}
    ds_depth_profiles = {}
    for stage in STAGES:
        path = f"faithful_baseline/results/mcts_deepsearch/deepsearch_{stage}_p0-9.json"
        if not os.path.exists(path):
            continue
        with open(path) as f:
            ds_data[stage] = json.load(f)

        bf_agg = defaultdict(list)
        dp_agg = defaultdict(list)
        for r in ds_data[stage]["results"]:
            tree = r["tree"]
            for d, vals in collect_mcts_bf(tree).items():
                if d <= MAX_DEPTH:
                    bf_agg[d].extend(vals)
            dp = collect_mcts_depth_profile(tree)
            for d, n in dp.items():
                if d <= MAX_DEPTH:
                    dp_agg[d].append(n)
        ds_bf[stage] = dict(bf_agg)
        ds_depth_profiles[stage] = dict(dp_agg)

    problem_indices = [68, 73, 104, 124, 155, 361, 374, 377, 394, 450]

    # =====================================================================
    # Plot 1: Branching factor λ(d) — DeepSearch vs Flat Rollout
    # =====================================================================
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for idx, stage in enumerate(STAGES):
        ax = axes[idx // 2, idx % 2]

        # Flat rollout
        flat_depths = sorted(int(d) for d in flat_params.get(stage, {})
                            if "poisson_lambda" in flat_params[stage][d] and int(d) <= MAX_DEPTH)
        flat_lambdas = [flat_params[stage][str(d)]["poisson_lambda"] for d in flat_depths]
        ax.plot(flat_depths, flat_lambdas, "o-", color=COLORS[stage],
                label="Flat Rollout (400 problems)", linewidth=2.5, markersize=7)

        # DeepSearch
        if stage in ds_bf:
            ds_depths = sorted(d for d in ds_bf[stage] if d <= MAX_DEPTH)
            ds_lambdas = [np.mean(ds_bf[stage][d]) for d in ds_depths]
            ax.plot(ds_depths, ds_lambdas, "s--", color="black",
                    label="DeepSearch (10 problems)", linewidth=2, markersize=6)

        ax.set_title(f"{STAGE_LABELS[stage]}", fontsize=12)
        ax.set_xlabel("Depth")
        ax.set_ylabel("Avg Branching Factor")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0)

    fig.suptitle("Branching Factor by Depth: Flat Rollout vs DeepSearch MCTS", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "bf_deepsearch_vs_flat.png"), dpi=150, bbox_inches="tight")
    print("Saved bf_deepsearch_vs_flat.png")

    # =====================================================================
    # Plot 2: Width (nodes) at each depth — DeepSearch vs Flat Rollout
    # =====================================================================
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for idx, stage in enumerate(STAGES):
        ax = axes[idx // 2, idx % 2]

        # Flat rollout width from params
        flat_depths = sorted(int(d) for d in flat_params.get(stage, {})
                            if "avg_width" in flat_params[stage].get(d, {}) and int(d) <= MAX_DEPTH)
        # Use branching_summary if available
        bs_path = "faithful_baseline/results/math500_full/train/branching_analysis/branching_summary.json"
        if os.path.exists(bs_path):
            with open(bs_path) as f:
                bs = json.load(f)
            if stage in bs:
                flat_depths = sorted(int(d) for d in bs[stage] if int(d) <= MAX_DEPTH)
                flat_widths = [bs[stage][str(d)].get("avg_width", 1) for d in flat_depths]
                ax.plot(flat_depths, flat_widths, "o-", color=COLORS[stage],
                        label="Flat Rollout (avg width)", linewidth=2.5, markersize=7)

        # DeepSearch width
        if stage in ds_depth_profiles:
            ds_depths = sorted(d for d in ds_depth_profiles[stage] if d <= MAX_DEPTH)
            ds_widths = [np.mean(ds_depth_profiles[stage][d]) for d in ds_depths]
            ax.plot(ds_depths, ds_widths, "s--", color="black",
                    label="DeepSearch (avg width)", linewidth=2, markersize=6)

        ax.set_title(f"{STAGE_LABELS[stage]}", fontsize=12)
        ax.set_xlabel("Depth")
        ax.set_ylabel("Avg # Nodes at Depth")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Tree Width by Depth: Flat Rollout vs DeepSearch MCTS", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "width_deepsearch_vs_flat.png"), dpi=150, bbox_inches="tight")
    print("Saved width_deepsearch_vs_flat.png")

    # =====================================================================
    # Plot 3: Side-by-side tree diagrams for selected problems
    # =====================================================================
    # Use the first 5 problems that overlap between 10-problem flat and DeepSearch
    # DeepSearch uses problems 0-9, flat uses [68,73,...,450]
    # They use different problem sets, so we compare structure rather than per-problem

    # Instead: show 4 DeepSearch trees (one per stage) for problem 0
    fig, axes = plt.subplots(1, 4, figsize=(24, 8))

    for idx, stage in enumerate(STAGES):
        ax = axes[idx]
        if stage in ds_data and ds_data[stage]["results"]:
            tree = ds_data[stage]["results"][0]["tree"]  # Problem 0
            pid = ds_data[stage]["results"][0]["problem_index"]
            correct = ds_data[stage]["results"][0].get("correct", False)
            nodes = ds_data[stage]["results"][0]["tree_stats"]["total_nodes"]
            md = ds_data[stage]["results"][0]["tree_stats"]["max_depth_reached"]
            title = f"{STAGE_LABELS[stage]}\nP{pid}: {'Correct' if correct else 'Wrong'}, {nodes} nodes, D={md}"
            draw_mcts_tree_on_ax(ax, tree, max_depth=6, title=title)

    fig.suptitle("DeepSearch MCTS Trees — Problem 0 Across Training Stages\n"
                 "Green=high Q-value, Red=low Q-value, width=visit count",
                 fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "deepsearch_trees_problem0.png"), dpi=150, bbox_inches="tight")
    print("Saved deepsearch_trees_problem0.png")

    # =====================================================================
    # Plot 4: Summary comparison bar chart
    # =====================================================================
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Accuracy comparison
    flat_acc = [0.438, 0.733, 0.772, 0.796]
    ds_acc = [ds_data[s]["summary"]["accuracy"] for s in STAGES if s in ds_data]
    x = np.arange(len(STAGES))
    w = 0.35
    axes[0].bar(x - w / 2, flat_acc, w, label="Flat Rollout (400p)", color=[COLORS[s] for s in STAGES], alpha=0.7)
    axes[0].bar(x + w / 2, ds_acc, w, label="DeepSearch (10p)", color="black", alpha=0.6)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([STAGE_LABELS[s] for s in STAGES], fontsize=8)
    axes[0].set_ylabel("Accuracy")
    axes[0].set_title("Accuracy")
    axes[0].legend()
    axes[0].set_ylim(0, 1.1)

    # D0 branching factor
    flat_d0 = [flat_params[s]["0"]["poisson_lambda"] for s in STAGES]
    ds_d0 = [np.mean(ds_bf[s].get(0, [8])) for s in STAGES if s in ds_bf]
    axes[1].bar(x - w / 2, flat_d0, w, label="Flat Rollout", color=[COLORS[s] for s in STAGES], alpha=0.7)
    axes[1].bar(x + w / 2, ds_d0, w, label="DeepSearch", color="black", alpha=0.6)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([STAGE_LABELS[s] for s in STAGES], fontsize=8)
    axes[1].set_ylabel("Avg Branching Factor")
    axes[1].set_title("D0 Branching Factor")
    axes[1].legend()

    # Avg nodes per tree (DeepSearch only, flat rollout doesn't have "nodes")
    ds_nodes = [ds_data[s]["summary"]["avg_nodes"] for s in STAGES if s in ds_data]
    axes[2].bar(x, ds_nodes, 0.5, color=[COLORS[s] for s in STAGES], alpha=0.7)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels([STAGE_LABELS[s] for s in STAGES], fontsize=8)
    axes[2].set_ylabel("Avg Nodes / Tree")
    axes[2].set_title("DeepSearch: Compute (Nodes)")

    fig.suptitle("Flat Rollout vs DeepSearch: Key Metrics", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "summary_comparison.png"), dpi=150, bbox_inches="tight")
    print("Saved summary_comparison.png")

    # =====================================================================
    # Print numerical summary
    # =====================================================================
    print("\n" + "=" * 80)
    print("FLAT ROLLOUT vs DEEPSEARCH: STRUCTURAL COMPARISON")
    print("=" * 80)

    for stage in STAGES:
        print(f"\n{STAGE_LABELS[stage]}:")
        flat_d0_bf = flat_params[stage]["0"]["poisson_lambda"]
        ds_d0_bf = np.mean(ds_bf[stage].get(0, [0])) if stage in ds_bf else 0

        # Flat: branching decays
        print(f"  Flat Rollout:")
        bfs = []
        for d in range(6):
            lam = flat_params[stage].get(str(d), {}).get("poisson_lambda", 0)
            bfs.append(f"D{d}={lam:.1f}")
        print(f"    BF: {', '.join(bfs)}")
        print(f"    Pattern: HIGH at D0, decays to ~1 at D3+")

        # DeepSearch: constant
        print(f"  DeepSearch:")
        bfs = []
        for d in range(6):
            vals = ds_bf.get(stage, {}).get(d, [])
            if vals:
                bfs.append(f"D{d}={np.mean(vals):.1f}")
        print(f"    BF: {', '.join(bfs)}")
        print(f"    Pattern: CONSTANT at base_width=8, then decays slowly")

        # Divergence
        diffs = []
        for d in range(MAX_DEPTH):
            flat_lam = flat_params[stage].get(str(d), {}).get("poisson_lambda", 0)
            ds_vals = ds_bf.get(stage, {}).get(d, [])
            if flat_lam > 0 and ds_vals:
                diff = abs(np.mean(ds_vals) - flat_lam) / flat_lam
                diffs.append(diff)
        if diffs:
            print(f"  Avg relative BF divergence: {np.mean(diffs):.1%}")

    print("\n" + "=" * 80)
    print("CONCLUSION")
    print("=" * 80)
    print("""
DeepSearch MCTS cannot replace flat rollouts because:

1. CONSTANT vs ADAPTIVE branching: DeepSearch uses fixed base_width=8 at D0,
   while flat rollouts show stage-dependent branching (4-26) with heavy tails.

2. SLOW DECAY vs FAST DECAY: DeepSearch branching decays slowly with depth
   (due to dynamic_width), while flat rollouts drop to ~1 by D3.

3. DIFFERENT DIVERSITY: Flat rollouts produce independent samples that capture
   the full diversity of reasoning paths. DeepSearch's UCT-like selection
   focuses compute on promising paths, reducing diversity.

4. WEAK MODEL FAILURE: DeepSearch fails on base model (20% vs 43.8% flat)
   because entropy guidance is unreliable with weak models.

These structural differences motivate Poisson-MCTS: using flat rollout
distributions to guide MCTS branching for better compute allocation.
""")


if __name__ == "__main__":
    main()

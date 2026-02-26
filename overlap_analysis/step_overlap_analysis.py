#!/usr/bin/env python3
"""Step-level overlap analysis: chunk rollouts into 256-token steps,
use OpenAI LLM to cluster semantically equivalent steps, build trees,
and generate overlap distribution plots + per-problem tree visualizations."""

import argparse
import asyncio
import json
import os
import time
from collections import defaultdict
from itertools import combinations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from openai import AsyncOpenAI
from transformers import AutoTokenizer


# ============================================================================
# Tokenization & chunking
# ============================================================================

def chunk_text_into_steps(text: str, tokenizer, step_size: int = 256) -> list[str]:
    """Chunk text into fixed-size token steps. Returns list of step strings."""
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    steps = []
    for i in range(0, len(token_ids), step_size):
        chunk_ids = token_ids[i : i + step_size]
        step_text = tokenizer.decode(chunk_ids, skip_special_tokens=True)
        if step_text.strip():
            steps.append(step_text.strip())
    return steps


# ============================================================================
# OpenAI semantic clustering
# ============================================================================

async def cluster_steps_llm(
    client: AsyncOpenAI,
    step_texts: dict[int, str],   # rollout_id -> step text
    semaphore: asyncio.Semaphore,
    model: str = "gpt-4o-mini",
) -> dict[int, int]:
    """Use OpenAI LLM to cluster semantically equivalent steps.
    Returns rollout_id -> cluster_id mapping."""
    rids = sorted(step_texts.keys())

    if len(rids) <= 1:
        return {rids[0]: 0} if rids else {}

    # Fast path: check for exact textual matches
    text_groups = defaultdict(list)
    for rid in rids:
        text_groups[step_texts[rid]].append(rid)
    if len(text_groups) == 1:
        return {rid: 0 for rid in rids}

    # Build prompt — truncate each step to ~300 chars to keep prompt short
    step_lines = []
    for idx, rid in enumerate(rids):
        t = step_texts[rid][:400].replace("\n", " ")
        step_lines.append(f'[{idx}]: "{t}"')

    prompt = (
        f"I have {len(rids)} reasoning steps from different rollouts solving the same math problem. "
        "Group them into clusters where steps in the same cluster perform essentially the same "
        "reasoning operation (same mathematical approach, same key calculation, same conclusion — "
        "possibly with different wording or formatting).\n\n"
        "Steps:\n" + "\n".join(step_lines) + "\n\n"
        "Respond with ONLY a JSON object mapping step index (string) to cluster label (int starting from 0). "
        "Example: {\"0\": 0, \"1\": 0, \"2\": 1}"
    )

    async with semaphore:
        for attempt in range(3):
            try:
                resp = await client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0,
                    max_tokens=300,
                    response_format={"type": "json_object"},
                )
                raw = json.loads(resp.choices[0].message.content)
                mapping = {}
                for idx_str, cluster in raw.items():
                    idx = int(idx_str)
                    if 0 <= idx < len(rids):
                        mapping[rids[idx]] = int(cluster)
                # Fill missing rollouts with unique clusters
                next_c = max(mapping.values(), default=-1) + 1
                for rid in rids:
                    if rid not in mapping:
                        mapping[rid] = next_c
                        next_c += 1
                return mapping
            except Exception as e:
                if attempt == 2:
                    print(f"    LLM clustering failed after 3 attempts: {e}")
                    return {rid: i for i, rid in enumerate(rids)}
                await asyncio.sleep(1 * (attempt + 1))

    return {rid: i for i, rid in enumerate(rids)}


def cluster_steps_heuristic(step_texts: dict[int, str]) -> dict[int, int]:
    """Simple heuristic clustering: group by normalized text similarity (no API)."""
    from difflib import SequenceMatcher

    rids = sorted(step_texts.keys())
    if len(rids) <= 1:
        return {rids[0]: 0} if rids else {}

    # Cluster greedily: assign to existing cluster if similarity > threshold
    threshold = 0.6
    clusters: list[tuple[int, str]] = []  # (cluster_id, representative_text)
    mapping = {}

    for rid in rids:
        text = step_texts[rid]
        assigned = False
        for cid, rep_text in clusters:
            ratio = SequenceMatcher(None, text.split()[:80], rep_text.split()[:80]).ratio()
            if ratio >= threshold:
                mapping[rid] = cid
                assigned = True
                break
        if not assigned:
            new_cid = len(clusters)
            clusters.append((new_cid, text))
            mapping[rid] = new_cid

    return mapping


# ============================================================================
# Tree building
# ============================================================================

async def build_tree_for_problem(
    problem_steps: dict[int, list[str]],  # rollout_id -> [step0, step1, ...]
    client: AsyncOpenAI | None,
    semaphore: asyncio.Semaphore | None,
    use_llm: bool = True,
    eval_model: str = "gpt-4o-mini",
) -> dict:
    """Build a semantic tree for one problem by clustering at each step level.
    Returns tree as nested dict."""
    all_rids = set(problem_steps.keys())
    max_steps = max(len(s) for s in problem_steps.values()) if problem_steps else 0

    root = {
        "rollout_ids": sorted(all_rids),
        "step_level": -1,
        "num_rollouts": len(all_rids),
        "children": [],
    }

    # BFS: process level by level
    current_nodes = [root]

    for level in range(max_steps):
        next_nodes = []
        for node in current_nodes:
            active_rids = {
                rid for rid in node["rollout_ids"]
                if rid in problem_steps and level < len(problem_steps[rid])
            }
            if not active_rids:
                continue

            step_texts = {rid: problem_steps[rid][level] for rid in active_rids}

            # Cluster
            if use_llm and client is not None and semaphore is not None:
                clustering = await cluster_steps_llm(
                    client, step_texts, semaphore, model=eval_model
                )
            else:
                clustering = cluster_steps_heuristic(step_texts)

            # Group by cluster
            cluster_groups = defaultdict(set)
            for rid, cid in clustering.items():
                cluster_groups[cid].add(rid)

            for cid, rids in sorted(cluster_groups.items()):
                child = {
                    "rollout_ids": sorted(rids),
                    "step_level": level,
                    "cluster_id": cid,
                    "num_rollouts": len(rids),
                    "children": [],
                }
                node["children"].append(child)
                next_nodes.append(child)

        current_nodes = next_nodes

    return root


# ============================================================================
# Overlap metrics from tree
# ============================================================================

def compute_overlap_from_tree(tree: dict, num_rollouts: int) -> dict:
    """Compute step-level overlap metrics from a semantic tree."""
    level_data = defaultdict(lambda: {"shared_pairs": 0, "total_pairs": 0, "num_clusters": 0})

    def traverse(node):
        if not node["children"]:
            return
        for child in node["children"]:
            lvl = child["step_level"]
            n = child["num_rollouts"]
            level_data[lvl]["shared_pairs"] += n * (n - 1) // 2
            traverse(child)
        # Count clusters and total at this level
        child_level = node["children"][0]["step_level"] if node["children"] else -1
        if child_level >= 0:
            total_at_level = sum(c["num_rollouts"] for c in node["children"])
            level_data[child_level]["total_pairs"] += total_at_level * (total_at_level - 1) // 2
            level_data[child_level]["num_clusters"] += len(node["children"])

    traverse(tree)

    # Overlap ratio per level = shared_pairs / total_pairs
    max_level = max(level_data.keys()) if level_data else -1
    step_overlaps = []
    branching_factors = []
    for lvl in range(max_level + 1):
        d = level_data[lvl]
        ratio = d["shared_pairs"] / d["total_pairs"] if d["total_pairs"] > 0 else 1.0
        step_overlaps.append(ratio)
        branching_factors.append(d["num_clusters"])

    # Token savings: tree stores each unique node once (256 tokens),
    # flat stores each rollout fully
    num_tree_nodes = 0

    def count_nodes(node):
        nonlocal num_tree_nodes
        for c in node["children"]:
            num_tree_nodes += 1
            count_nodes(c)

    count_nodes(tree)

    return {
        "step_overlaps": step_overlaps,
        "branching_factors": branching_factors,
        "num_tree_nodes": num_tree_nodes,
        "max_depth": max_level + 1 if max_level >= 0 else 0,
    }


# ============================================================================
# Tree visualization
# ============================================================================

def compute_subtree_width(node):
    """Compute width of subtree for layout (leaf = 1)."""
    if not node["children"]:
        return max(1, node["num_rollouts"] * 0.3)
    return sum(compute_subtree_width(c) for c in node["children"])


def collect_positions_and_edges(node, depth=0, x_start=0.0):
    """Recursively compute node positions and edges for tree drawing."""
    w = compute_subtree_width(node)
    x_center = x_start + w / 2.0
    y = -depth

    positions = [(x_center, y, node["num_rollouts"], node.get("step_level", -1))]
    edges = []

    if node["children"]:
        child_x = x_start
        for child in node["children"]:
            child_w = compute_subtree_width(child)
            child_positions, child_edges = collect_positions_and_edges(
                child, depth + 1, child_x
            )
            # Edge from this node to child
            child_x_center = child_x + child_w / 2.0
            edges.append((
                (x_center, y),
                (child_x_center, -(depth + 1)),
                child["num_rollouts"],
            ))
            positions.extend(child_positions)
            edges.extend(child_edges)
            child_x += child_w

    return positions, edges


def plot_problem_tree(tree: dict, problem_id: int, problem_text: str,
                      output_path: str, num_rollouts: int):
    """Draw tree structure for one problem and save as PNG."""
    positions, edges = collect_positions_and_edges(tree)

    if not positions:
        return

    max_x = max(p[0] for p in positions)
    min_x = min(p[0] for p in positions)
    max_depth = max(-p[1] for p in positions)

    fig_width = max(8, min(24, (max_x - min_x) * 1.2 + 3))
    fig_height = max(5, min(16, max_depth * 1.0 + 3))

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    # Draw edges
    for (x1, y1), (x2, y2), n in edges:
        alpha = 0.2 + 0.8 * (n / num_rollouts)
        lw = 0.5 + 3.5 * (n / num_rollouts)
        ax.plot([x1, x2], [y1, y2], "-", color="steelblue", alpha=alpha, linewidth=lw)

    # Draw nodes
    cmap = plt.colormaps["Blues"]
    for x, y, n, lvl in positions:
        frac = n / num_rollouts
        size = 150 + 800 * frac
        color = cmap(0.25 + 0.7 * frac)
        ax.scatter(x, y, s=size, c=[color], edgecolors="black",
                   linewidths=0.8, zorder=3)
        ax.text(x, y, str(n), ha="center", va="center",
                fontsize=max(6, min(10, 7 + 3 * frac)), fontweight="bold", zorder=4)

    # Labels — sanitize LaTeX in problem text to avoid matplotlib parse errors
    safe_text = problem_text[:80].replace("$", "").replace("\\", "/")
    safe_text += "..." if len(problem_text) > 80 else ""
    ax.set_title(f"Problem {problem_id}: Rollout Branching Tree\n\"{safe_text}\"",
                 fontsize=9, pad=12)
    ax.set_ylabel("Step Depth")
    yticks = list(range(0, -(int(max_depth) + 1), -1))
    ax.set_yticks(yticks)
    ax.set_yticklabels([str(-y) for y in yticks])
    ax.set_xticks([])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# ============================================================================
# Summary plots
# ============================================================================

def plot_all_summaries(all_stats: list[dict], output_dir: str, num_rollouts: int,
                       all_problem_steps: list[dict] | None = None):
    """Generate summary distribution plots across all problems."""
    os.makedirs(output_dir, exist_ok=True)

    # Collect data
    all_step_overlaps = [s["step_overlaps"] for s in all_stats]
    all_mean_overlaps = [
        float(np.mean(so)) if so else 0.0 for so in all_step_overlaps
    ]
    all_branching = [s["branching_factors"] for s in all_stats]
    all_tree_depths = [s["max_depth"] for s in all_stats]
    all_tree_nodes = [s["num_tree_nodes"] for s in all_stats]

    max_depth = max(len(so) for so in all_step_overlaps) if all_step_overlaps else 0
    depths = list(range(max_depth))

    # Precompute branching stats per depth
    branch_means = []
    branch_stds = []
    branch_medians = []
    for d in range(max_depth):
        vals = [bf[d] for bf in all_branching if d < len(bf)]
        branch_means.append(float(np.mean(vals)) if vals else 1.0)
        branch_stds.append(float(np.std(vals)) if vals else 0.0)
        branch_medians.append(float(np.median(vals)) if vals else 1.0)

    # ---- PLOT 1: Cluster Count by Depth (PRIMARY METRIC) ----
    # This is the most honest metric: how many distinct paths exist at each step?
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(depths, branch_means, "o-", color="#4C72B0", linewidth=2.5,
            markersize=8, label="Mean clusters", zorder=3)
    ax.fill_between(depths,
                    [max(1, m - s) for m, s in zip(branch_means, branch_stds)],
                    [min(num_rollouts, m + s) for m, s in zip(branch_means, branch_stds)],
                    alpha=0.15, color="#4C72B0")
    ax.plot(depths, branch_medians, "s--", color="#55A868", linewidth=1.5,
            markersize=5, alpha=0.8, label="Median clusters")
    ax.axhline(1, color="green", linestyle=":", alpha=0.5,
               label="1 cluster (all rollouts identical)")
    ax.axhline(num_rollouts, color="red", linestyle=":", alpha=0.5,
               label=f"{num_rollouts} clusters (all rollouts unique)")
    # Shade the "sharing zone" (fewer clusters = more sharing)
    ax.fill_between(depths, 1, branch_means, alpha=0.08, color="green",
                    label="Shared computation (saved by tree)")
    ax.set_xlabel("Step Depth (each step = 256 tokens)", fontsize=11)
    ax.set_ylabel("Number of Distinct Reasoning Clusters", fontsize=11)
    ax.set_title("How Many Distinct Reasoning Paths Exist at Each Step?\n"
                 f"({num_rollouts} rollouts per problem, {len(all_stats)} problems)",
                 fontsize=12)
    ax.set_ylim(0, num_rollouts + 1)
    ax.set_yticks(range(0, num_rollouts + 2, 2))
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "cluster_count_by_depth.png"), dpi=150)
    plt.close(fig)

    # ---- PLOT 2: Cluster Count Distribution (violin) ----
    violin_data = []
    violin_labels = []
    for d in range(min(max_depth, 10)):
        vals = [bf[d] for bf in all_branching if d < len(bf)]
        if vals:
            violin_data.append(vals)
            violin_labels.append(f"Step {d}")

    if violin_data:
        fig, ax = plt.subplots(figsize=(12, 6))
        parts = ax.violinplot(violin_data, showmeans=True, showmedians=True)
        for pc in parts["bodies"]:
            pc.set_facecolor("#4C72B0")
            pc.set_alpha(0.5)
        ax.boxplot(violin_data, widths=0.18, showfliers=False, zorder=3)
        ax.set_xticks(range(1, len(violin_labels) + 1))
        ax.set_xticklabels(violin_labels)
        ax.set_ylabel("Number of Clusters (out of 12 rollouts)")
        ax.set_title("Distribution of Cluster Count Across Problems at Each Step")
        ax.set_ylim(0, num_rollouts + 1)
        ax.axhline(num_rollouts, color="red", linestyle=":", alpha=0.4)
        ax.axhline(1, color="green", linestyle=":", alpha=0.4)
        ax.grid(True, alpha=0.2, axis="y")
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, "cluster_count_violin.png"), dpi=150)
        plt.close(fig)

    # ---- PLOT 3: Sharing Ratio by Depth ----
    # sharing_ratio = 1 - clusters/N  (0 = all unique, 1 = all same)
    sharing_means = [1.0 - b / num_rollouts for b in branch_means]
    sharing_stds = [s / num_rollouts for s in branch_stds]

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["#2ca02c" if s > 0.3 else "#d4a017" if s > 0.1 else "#c44e52"
              for s in sharing_means]
    bars = ax.bar(depths, sharing_means, color=colors, edgecolor="black",
                  alpha=0.8, width=0.8)
    # Add value labels on bars
    for d, bar in enumerate(bars):
        h = bar.get_height()
        if h > 0.02:
            avg_cl = branch_means[d]
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01,
                    f"{avg_cl:.1f}",
                    ha="center", va="bottom", fontsize=8, color="gray")
    ax.set_xlabel("Step Depth (each step = 256 tokens)", fontsize=11)
    ax.set_ylabel("Sharing Ratio  (1 - clusters / N)", fontsize=11)
    ax.set_title("Computation Sharing Ratio by Step Depth\n"
                 "Green = high sharing, Yellow = moderate, Red = low",
                 fontsize=12)
    ax.set_ylim(0, 1.0)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "sharing_ratio_by_depth.png"), dpi=150)
    plt.close(fig)

    # ---- PLOT 4: Largest Cluster Size by Depth ----
    # Shows: at each step, how many rollouts share the most popular approach?
    # We don't have per-cluster sizes in stats, but we can estimate from tree JSON
    # For now, compute from branching: if N rollouts split into B clusters,
    # with uneven splits, the largest cluster is typically > N/B.
    # Actually let's compute this from the trees.json if available.
    trees_path = os.path.join(output_dir, "trees.json")
    if os.path.exists(trees_path):
        with open(trees_path) as f:
            trees_raw = json.load(f)
        # Unwrap: each entry may have {"problem_id": ..., "tree": {...}} or be a tree directly
        trees = []
        for entry in trees_raw:
            trees.append(entry.get("tree", entry))
        # For each problem, at each depth, find the largest cluster
        largest_by_depth = {}
        for tree in trees:
            nodes = [tree]
            depth = 0
            while nodes:
                next_nodes = []
                sizes = [n["num_rollouts"] for n in nodes]
                largest_by_depth.setdefault(depth, []).append(max(sizes))
                for n in nodes:
                    next_nodes.extend(n.get("children", []))
                nodes = next_nodes
                depth += 1

        fig, ax = plt.subplots(figsize=(10, 6))
        plot_depths = sorted(d for d in largest_by_depth if d <= max_depth)
        means = [float(np.mean(largest_by_depth[d])) for d in plot_depths]
        medians = [float(np.median(largest_by_depth[d])) for d in plot_depths]
        p75 = [float(np.percentile(largest_by_depth[d], 75)) for d in plot_depths]
        p25 = [float(np.percentile(largest_by_depth[d], 25)) for d in plot_depths]
        ax.plot(plot_depths, means, "o-", color="#C44E52", linewidth=2.5,
                markersize=7, label="Mean", zorder=3)
        ax.plot(plot_depths, medians, "s--", color="#DD8452", linewidth=1.5,
                markersize=5, label="Median", alpha=0.8)
        ax.fill_between(plot_depths, p25, p75, alpha=0.15, color="#C44E52",
                        label="25th-75th percentile")
        ax.axhline(1, color="gray", linestyle=":", alpha=0.5)
        ax.set_xlabel("Step Depth (each step = 256 tokens)", fontsize=11)
        ax.set_ylabel("Largest Cluster Size (# rollouts)", fontsize=11)
        ax.set_title("Largest Cluster Size at Each Step Depth\n"
                     "(How many rollouts share the most popular approach?)",
                     fontsize=12)
        ax.set_ylim(0, num_rollouts + 1)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, "largest_cluster_by_depth.png"), dpi=150)
        plt.close(fig)

    # ---- PLOT 5: Token savings estimate ----
    if all_problem_steps:
        total_flat_steps = sum(
            sum(len(steps) for steps in prob_steps.values())
            for prob_steps in all_problem_steps
        )
    else:
        total_flat_steps = sum(s["max_depth"] * num_rollouts for s in all_stats)
    total_tree_nodes = sum(all_tree_nodes)
    savings_pct = (1 - total_tree_nodes / total_flat_steps) * 100 if total_flat_steps > 0 else 0

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(
        ["Flat Rollouts\n(total steps)", "Tree\n(unique nodes)"],
        [total_flat_steps, total_tree_nodes],
        color=["#DD8452", "#4C72B0"], edgecolor="black",
    )
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h, f"{h:,.0f}",
                ha="center", va="bottom", fontsize=10)
    ax.set_ylabel("Count (each ≈ 256 tokens)")
    ax.set_title(f"Step-Level Token Savings: Flat vs Tree\n"
                 f"Estimated savings: {savings_pct:.1f}%")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "step_token_savings.png"), dpi=150)
    plt.close(fig)

    # Compute honest summary stats
    global_mean = float(np.mean(all_mean_overlaps))
    depth_overlap_means = []
    for d in range(max_depth):
        vals = [so[d] for so in all_step_overlaps if d < len(so)]
        depth_overlap_means.append(float(np.mean(vals)) if vals else 0.0)

    return {
        "global_mean_step_overlap": global_mean,
        "overlap_by_depth": depth_overlap_means,
        "branching_by_depth": branch_means,
        "sharing_ratio_by_depth": sharing_means,
        "total_flat_steps": total_flat_steps,
        "total_tree_nodes": total_tree_nodes,
        "savings_pct": savings_pct,
        "mean_tree_depth": float(np.mean(all_tree_depths)) if all_tree_depths else 0,
    }


# ============================================================================
# Main pipeline
# ============================================================================

async def process_problem(
    problem_idx: int,
    problem_steps: dict[int, list[str]],
    client: AsyncOpenAI | None,
    semaphore: asyncio.Semaphore | None,
    use_llm: bool,
    eval_model: str,
    num_rollouts: int,
    output_dir: str,
    problem_text: str,
    plot_trees: bool = True,
) -> dict:
    """Process one problem: build tree, compute metrics, optionally plot."""
    tree = await build_tree_for_problem(
        problem_steps, client, semaphore, use_llm=use_llm, eval_model=eval_model,
    )
    stats = compute_overlap_from_tree(tree, num_rollouts)

    if plot_trees:
        trees_dir = os.path.join(output_dir, "trees")
        os.makedirs(trees_dir, exist_ok=True)
        plot_problem_tree(
            tree, problem_idx, problem_text,
            os.path.join(trees_dir, f"tree_problem_{problem_idx:04d}.png"),
            num_rollouts,
        )

    return {
        "problem_id": problem_idx,
        "tree": tree,
        **stats,
    }


async def run_analysis(args):
    """Main async analysis pipeline."""
    # Load results
    print(f"Loading inference results from {args.results}...")
    with open(args.results) as f:
        data = json.load(f)

    results = data["results"]
    model_name = data.get("config", {}).get("model", "unknown")
    num_rollouts = data.get("config", {}).get("num_rollouts", 12)

    if args.num_problems:
        results = results[:args.num_problems]
    print(f"Loaded {len(results)} problems, {num_rollouts} rollouts each")

    # Load tokenizer for chunking
    print(f"Loading tokenizer for {model_name}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    except Exception:
        print(f"  Could not load tokenizer for {model_name}, trying Qwen/Qwen2.5-7B-Instruct...")
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct", trust_remote_code=True)

    # Chunk all rollouts into steps
    print(f"Chunking rollouts into {args.step_size}-token steps...")
    all_problem_steps = []
    for entry in results:
        problem_steps = {}
        for rollout in entry["rollouts"]:
            rid = rollout["rollout_id"]
            # Prefer thinking content; fall back to full response
            text = rollout.get("thinking", "") or rollout.get("response", "")
            steps = chunk_text_into_steps(text, tokenizer, step_size=args.step_size)
            if steps:
                problem_steps[rid] = steps
        all_problem_steps.append(problem_steps)

    step_counts = [len(s) for ps in all_problem_steps for s in ps.values()]
    print(f"  Avg steps per rollout: {np.mean(step_counts):.1f}, "
          f"max: {max(step_counts)}, min: {min(step_counts)}")

    # Set up OpenAI client
    client = None
    semaphore = None
    if args.use_llm:
        api_key = args.openai_api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            print("WARNING: No OpenAI API key provided. Falling back to heuristic clustering.")
            args.use_llm = False
        else:
            client = AsyncOpenAI(api_key=api_key)
            semaphore = asyncio.Semaphore(args.max_concurrent)
            print(f"Using LLM clustering with model={args.eval_model}, "
                  f"max_concurrent={args.max_concurrent}")

    # Process all problems
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"\nBuilding semantic trees for {len(results)} problems...")
    start_time = time.time()

    # Process in batches to show progress
    all_stats = []
    batch_size = 20
    for batch_start in range(0, len(results), batch_size):
        batch_end = min(batch_start + batch_size, len(results))
        print(f"  Processing problems {batch_start}-{batch_end - 1}...")

        tasks = []
        for i in range(batch_start, batch_end):
            tasks.append(process_problem(
                problem_idx=i,
                problem_steps=all_problem_steps[i],
                client=client,
                semaphore=semaphore,
                use_llm=args.use_llm,
                eval_model=args.eval_model,
                num_rollouts=num_rollouts,
                output_dir=args.output_dir,
                problem_text=results[i]["problem"],
                plot_trees=args.plot_trees,
            ))

        batch_results = await asyncio.gather(*tasks)
        all_stats.extend(batch_results)

    elapsed = time.time() - start_time
    print(f"Tree building completed in {elapsed:.1f}s")

    # Generate summary plots
    print("\nGenerating summary plots...")
    global_stats = plot_all_summaries(all_stats, args.output_dir, num_rollouts,
                                      all_problem_steps=all_problem_steps)

    # Save summary JSON
    summary = {
        "config": {
            "model": model_name,
            "num_rollouts": num_rollouts,
            "step_size": args.step_size,
            "use_llm": args.use_llm,
            "eval_model": args.eval_model if args.use_llm else "heuristic",
        },
        "num_problems": len(results),
        "global_stats": global_stats,
        "per_problem": [
            {k: v for k, v in s.items() if k != "tree"}
            for s in all_stats
        ],
    }
    summary_path = os.path.join(args.output_dir, "step_overlap_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    # Also save trees separately (they can be large)
    trees_json = [{"problem_id": s["problem_id"], "tree": s["tree"]} for s in all_stats]
    trees_path = os.path.join(args.output_dir, "trees.json")
    with open(trees_path, "w") as f:
        json.dump(trees_json, f, indent=2)

    # Print summary
    print("\n" + "=" * 65)
    print("STEP-LEVEL OVERLAP ANALYSIS RESULTS")
    print("=" * 65)
    print(f"Model:                       {model_name}")
    print(f"Problems analyzed:           {len(results)}")
    print(f"Rollouts per problem:        {num_rollouts}")
    print(f"Step size:                   {args.step_size} tokens")
    print(f"Clustering method:           {'LLM (' + args.eval_model + ')' if args.use_llm else 'Heuristic'}")
    print(f"Global mean step overlap:    {global_stats['global_mean_step_overlap']:.4f}")
    print(f"Mean tree depth:             {global_stats['mean_tree_depth']:.1f} steps")
    print(f"Total flat steps:            {global_stats['total_flat_steps']:,}")
    print(f"Total tree nodes:            {global_stats['total_tree_nodes']:,}")
    print(f"Estimated step savings:      {global_stats['savings_pct']:.1f}%")
    print(f"\nOverlap by depth:  {['%.3f' % v for v in global_stats['overlap_by_depth'][:10]]}")
    print(f"Branching by depth: {['%.1f' % v for v in global_stats['branching_by_depth'][:10]]}")
    print("=" * 65)
    print(f"\nOutputs saved to {args.output_dir}/")
    if args.plot_trees:
        print(f"Tree plots saved to {args.output_dir}/trees/")


def main():
    parser = argparse.ArgumentParser(
        description="Step-level overlap analysis with semantic clustering and tree visualization.")
    parser.add_argument("--results", required=True,
                        help="Path to inference results JSON")
    parser.add_argument("--output_dir", default="overlap_analysis/step_results",
                        help="Output directory for plots and summaries")
    parser.add_argument("--step_size", type=int, default=256,
                        help="Tokens per step (default: 256)")
    parser.add_argument("--num_problems", type=int, default=None,
                        help="Limit number of problems (default: all)")
    parser.add_argument("--use_llm", action="store_true", default=True,
                        help="Use LLM for semantic clustering (default: True)")
    parser.add_argument("--no_llm", dest="use_llm", action="store_false",
                        help="Use heuristic clustering instead of LLM")
    parser.add_argument("--eval_model", default="gpt-4o-mini",
                        help="OpenAI model for clustering (default: gpt-4o-mini)")
    parser.add_argument("--openai_api_key", default=None,
                        help="OpenAI API key (or set OPENAI_API_KEY env var)")
    parser.add_argument("--max_concurrent", type=int, default=30,
                        help="Max concurrent OpenAI API calls (default: 30)")
    parser.add_argument("--plot_trees", action="store_true", default=True,
                        help="Generate per-problem tree plots (default: True)")
    parser.add_argument("--no_plot_trees", dest="plot_trees", action="store_false",
                        help="Skip per-problem tree plots")
    args = parser.parse_args()

    asyncio.run(run_analysis(args))


if __name__ == "__main__":
    main()

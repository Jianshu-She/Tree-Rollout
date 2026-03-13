#!/usr/bin/env python3
"""Sankey-style tree flow diagram: show how 128 rollouts split across depths.

For a single problem, visualize the tree as a left-to-right flow where:
- Width of each flow = number of rollouts
- Color = accuracy (green=correct, red=incorrect)
"""

import asyncio
import json
import os
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from matplotlib.patches import FancyArrowPatch
import numpy as np
from transformers import AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "mcts_inference"))

from overlap_analysis.step_overlap_analysis import (
    chunk_text_into_steps,
    build_tree_for_problem,
)
from mcts_inference.utils import is_correct


def compute_rollout_correctness(rollouts, ground_truth):
    correctness = {}
    for rollout in rollouts:
        rid = rollout["rollout_id"]
        text = rollout.get("full_text", "") or rollout.get("response", "")
        correctness[rid] = is_correct(text, ground_truth)
    return correctness


def collect_leaf_rollout_ids(node):
    if not node["children"]:
        return set(node["rollout_ids"])
    result = set()
    for child in node["children"]:
        result |= collect_leaf_rollout_ids(child)
    return result


def flatten_tree(tree, correctness, max_depth=8):
    """Flatten tree into nodes and edges with flow information.

    Returns:
        nodes: list of dicts {id, depth, num_rollouts, accuracy, y_center, y_height}
        edges: list of dicts {src, dst, num_rollouts, accuracy}
    """
    nodes = []
    edges = []
    node_id_counter = [0]

    def traverse(tnode, depth, parent_id):
        nid = node_id_counter[0]
        node_id_counter[0] += 1

        rids = collect_leaf_rollout_ids(tnode)
        if not rids:
            return None

        num_correct = sum(1 for r in rids if correctness.get(r, False))
        acc = num_correct / len(rids)

        nodes.append({
            "id": nid,
            "depth": depth,
            "num_rollouts": len(rids),
            "num_correct": num_correct,
            "accuracy": acc,
        })

        if parent_id is not None:
            edges.append({
                "src": parent_id,
                "dst": nid,
                "num_rollouts": len(rids),
                "accuracy": acc,
            })

        if depth < max_depth:
            children = tnode.get("children", [])
            # Sort children: biggest first for better layout
            children_info = []
            for child in children:
                child_rids = collect_leaf_rollout_ids(child)
                if child_rids:
                    child_nc = sum(1 for r in child_rids if correctness.get(r, False))
                    children_info.append((child, len(child_rids), child_nc / len(child_rids)))

            # Sort by accuracy descending (green on top, red on bottom)
            children_info.sort(key=lambda x: -x[2])

            for child, _, _ in children_info:
                traverse(child, depth + 1, nid)

        return nid

    traverse(tree, 0, None)
    return nodes, edges


def acc_to_color(acc):
    """Map accuracy to color: green for correct, red for incorrect, gray for mixed."""
    if acc > 0.85:
        # Bright green
        intensity = (acc - 0.85) / 0.15
        return (0.1, 0.6 + 0.3 * intensity, 0.1, 0.85)
    elif acc < 0.15:
        # Deep red
        intensity = (0.15 - acc) / 0.15
        return (0.7 + 0.2 * intensity, 0.1, 0.1, 0.85)
    else:
        # Gray zone with slight tint
        t = (acc - 0.15) / 0.7  # 0 = near red, 1 = near green
        r = 0.6 - 0.3 * t
        g = 0.3 + 0.3 * t
        return (r, g, 0.2, 0.7)


def draw_sankey_tree(tree, correctness, problem_info, output_path, max_depth=6):
    """Draw a Sankey-style tree flow diagram (standalone figure)."""
    from collections import deque

    nodes, edges = flatten_tree(tree, correctness, max_depth=max_depth)

    if not nodes:
        print("  No nodes to draw")
        return

    # Build lookup structures
    node_map = {n["id"]: n for n in nodes}
    children_map = defaultdict(list)
    for edge in edges:
        children_map[edge["src"]].append(edge["dst"])

    max_d = max(n["depth"] for n in nodes)
    total_rollouts = nodes[0]["num_rollouts"]

    gap = 0.12
    total_height = 10.0
    node_positions = {}

    # Layout root centered
    root = nodes[0]
    root_height = (root["num_rollouts"] / total_rollouts) * total_height
    root_y_top = total_height / 2.0 + root_height / 2.0
    node_positions[root["id"]] = (0.0, root_y_top, root_y_top - root_height)

    # BFS: position children starting from parent's y_top
    queue = deque([root["id"]])
    while queue:
        pid = queue.popleft()
        if pid not in children_map:
            continue
        p_x, p_y_top, p_y_bottom = node_positions[pid]
        child_ids = children_map[pid]
        y_current = p_y_top
        for cid in child_ids:
            child = node_map[cid]
            child_height = (child["num_rollouts"] / total_rollouts) * total_height
            child_x = child["depth"] * 2.0
            y_top = y_current
            y_bottom = y_current - child_height
            node_positions[cid] = (child_x, y_top, y_bottom)
            y_current = y_bottom - gap
            queue.append(cid)

    # Compute actual y range
    all_y_tops = [pos[1] for pos in node_positions.values()]
    all_y_bots = [pos[2] for pos in node_positions.values()]
    y_max = max(all_y_tops)
    y_min = min(all_y_bots)
    actual_height = y_max - y_min

    fig_width = max(14, (max_d + 1) * 2.5)
    fig_height = max(8, actual_height * 1.2 + 3)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    # Track output ports
    src_y_tracker = {}
    dst_y_tracker = {}
    for nid in node_positions:
        x, y_top, y_bottom = node_positions[nid]
        src_y_tracker[nid] = y_top
        dst_y_tracker[nid] = y_top

    # Draw edges
    for edge in edges:
        src_id = edge["src"]
        dst_id = edge["dst"]
        if src_id not in node_positions or dst_id not in node_positions:
            continue
        sx, sy_top, sy_bottom = node_positions[src_id]
        dx, dy_top, dy_bottom = node_positions[dst_id]
        flow_height_src = (sy_top - sy_bottom) * (edge["num_rollouts"] /
                          node_map[src_id]["num_rollouts"])
        flow_height_dst = dy_top - dy_bottom
        s_top = src_y_tracker[src_id]
        s_bottom = s_top - flow_height_src
        src_y_tracker[src_id] = s_bottom
        d_top = dst_y_tracker[dst_id]
        d_bottom = d_top - flow_height_dst
        dst_y_tracker[dst_id] = d_bottom

        color = acc_to_color(edge["accuracy"])
        n_points = 50
        t = np.linspace(0, 1, n_points)
        sx_right = sx + 0.15
        dx_left = dx - 0.15
        cx1 = sx_right + (dx_left - sx_right) * 0.4
        cx2 = sx_right + (dx_left - sx_right) * 0.6
        top_x = (1-t)**3 * sx_right + 3*(1-t)**2*t * cx1 + 3*(1-t)*t**2 * cx2 + t**3 * dx_left
        top_y = (1-t) * s_top + t * d_top
        bot_x = top_x
        bot_y = (1-t) * s_bottom + t * d_bottom
        poly_x = np.concatenate([top_x, bot_x[::-1]])
        poly_y = np.concatenate([top_y, bot_y[::-1]])
        ax.fill(poly_x, poly_y, color=color, edgecolor="none")

    # Draw node rectangles
    node_width = 0.3
    for n in nodes:
        nid = n["id"]
        if nid not in node_positions:
            continue
        x, y_top, y_bottom = node_positions[nid]
        height = y_top - y_bottom
        color = acc_to_color(n["accuracy"])
        rect = plt.Rectangle((x - node_width/2, y_bottom), node_width, height,
                              facecolor=color, edgecolor="black", linewidth=0.8, zorder=3)
        ax.add_patch(rect)
        if height > 0.3:
            label = f'{n["num_rollouts"]}\n({n["accuracy"]:.0%})'
            fontsize = min(9, max(6, height * 2))
            text_color = "white" if n["accuracy"] > 0.7 or n["accuracy"] < 0.3 else "black"
            ax.text(x, (y_top + y_bottom) / 2, label,
                    ha="center", va="center", fontsize=fontsize,
                    color=text_color, fontweight="bold", zorder=4)
        elif height > 0.15:
            ax.text(x, (y_top + y_bottom) / 2, f'{n["num_rollouts"]}',
                    ha="center", va="center", fontsize=5.5,
                    color="white", fontweight="bold", zorder=4)

    # Depth labels
    margin = 0.8
    for d in range(max_d + 1):
        x = d * 2.0
        label = "Root" if d == 0 else f"Depth {d}"
        ax.text(x, y_min - margin, label, ha="center", va="top",
                fontsize=10, fontweight="bold")

    pid = problem_info["pid"]
    level = problem_info["level"]
    overall_acc = problem_info["overall_acc"]
    n_branches = problem_info["n_branches"]
    method_label = problem_info.get("method", "Syntactic t=0.3")
    ax.set_title(
        f"Problem {pid} (Level {level}) — {method_label}\n"
        f"Overall accuracy: {overall_acc:.0%} | First split: {n_branches} branches\n"
        f"Width = rollout count | Green (>85% correct), Red (<15% correct), Gray (mixed)",
        fontsize=12, pad=15,
    )

    ax.set_xlim(-0.5, max_d * 2.0 + 0.5)
    ax.set_ylim(y_min - margin - 0.5, y_max + 0.5)
    ax.axis("off")

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved {output_path}")


def draw_sankey_on_ax(ax, tree, correctness, problem_info, max_depth=6):
    """Draw a Sankey tree on a given axes (for side-by-side comparison).

    Node height is proportional to num_rollouts relative to root total.
    Children are positioned starting from their parent's y position (not centered).
    """
    from collections import deque

    nodes, edges = flatten_tree(tree, correctness, max_depth=max_depth)

    if not nodes:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        return

    # Build lookup structures
    node_map = {n["id"]: n for n in nodes}
    children_map = defaultdict(list)
    for edge in edges:
        children_map[edge["src"]].append(edge["dst"])

    max_d = max(n["depth"] for n in nodes)
    total_rollouts = nodes[0]["num_rollouts"]

    gap = 0.12  # fixed gap between sibling nodes
    total_height = 10.0
    node_positions = {}

    # Layout root centered
    root = nodes[0]
    root_height = (root["num_rollouts"] / total_rollouts) * total_height
    root_y_top = total_height / 2.0 + root_height / 2.0
    node_positions[root["id"]] = (0.0, root_y_top, root_y_top - root_height)

    # BFS: position children starting from parent's y_top (parent-aligned layout)
    queue = deque([root["id"]])
    while queue:
        pid = queue.popleft()
        if pid not in children_map:
            continue

        p_x, p_y_top, p_y_bottom = node_positions[pid]
        child_ids = children_map[pid]

        y_current = p_y_top  # start from parent's top edge

        for cid in child_ids:
            child = node_map[cid]
            child_height = (child["num_rollouts"] / total_rollouts) * total_height
            child_x = child["depth"] * 2.0

            y_top = y_current
            y_bottom = y_current - child_height
            node_positions[cid] = (child_x, y_top, y_bottom)

            y_current = y_bottom - gap
            queue.append(cid)

    # Compute actual y range from all positions
    all_y_tops = [pos[1] for pos in node_positions.values()]
    all_y_bots = [pos[2] for pos in node_positions.values()]
    y_max = max(all_y_tops)
    y_min = min(all_y_bots)

    # Track output ports for flow drawing
    src_y_tracker = {}
    dst_y_tracker = {}
    for nid in node_positions:
        x, y_top, y_bottom = node_positions[nid]
        src_y_tracker[nid] = y_top
        dst_y_tracker[nid] = y_top

    # Draw edges
    for edge in edges:
        src_id = edge["src"]
        dst_id = edge["dst"]
        if src_id not in node_positions or dst_id not in node_positions:
            continue
        sx, sy_top, sy_bottom = node_positions[src_id]
        dx, dy_top, dy_bottom = node_positions[dst_id]
        flow_height_src = (sy_top - sy_bottom) * (edge["num_rollouts"] /
                          node_map[src_id]["num_rollouts"])
        flow_height_dst = dy_top - dy_bottom
        s_top = src_y_tracker[src_id]
        s_bottom = s_top - flow_height_src
        src_y_tracker[src_id] = s_bottom
        d_top = dst_y_tracker[dst_id]
        d_bottom = d_top - flow_height_dst
        dst_y_tracker[dst_id] = d_bottom

        color = acc_to_color(edge["accuracy"])
        n_points = 50
        t = np.linspace(0, 1, n_points)
        sx_right = sx + 0.15
        dx_left = dx - 0.15
        cx1 = sx_right + (dx_left - sx_right) * 0.4
        cx2 = sx_right + (dx_left - sx_right) * 0.6
        top_x = (1-t)**3 * sx_right + 3*(1-t)**2*t * cx1 + 3*(1-t)*t**2 * cx2 + t**3 * dx_left
        top_y = (1-t) * s_top + t * d_top
        bot_x = top_x
        bot_y = (1-t) * s_bottom + t * d_bottom
        poly_x = np.concatenate([top_x, bot_x[::-1]])
        poly_y = np.concatenate([top_y, bot_y[::-1]])
        ax.fill(poly_x, poly_y, color=color, edgecolor="none")

    # Draw node rectangles
    node_width = 0.3
    for n in nodes:
        nid = n["id"]
        if nid not in node_positions:
            continue
        x, y_top, y_bottom = node_positions[nid]
        height = y_top - y_bottom
        color = acc_to_color(n["accuracy"])
        rect = plt.Rectangle((x - node_width/2, y_bottom), node_width, height,
                              facecolor=color, edgecolor="black", linewidth=0.8, zorder=3)
        ax.add_patch(rect)
        if height > 0.3:
            label = f'{n["num_rollouts"]}\n({n["accuracy"]:.0%})'
            fontsize = min(9, max(6, height * 2))
            text_color = "white" if n["accuracy"] > 0.7 or n["accuracy"] < 0.3 else "black"
            ax.text(x, (y_top + y_bottom) / 2, label,
                    ha="center", va="center", fontsize=fontsize,
                    color=text_color, fontweight="bold", zorder=4)
        elif height > 0.15:
            ax.text(x, (y_top + y_bottom) / 2, f'{n["num_rollouts"]}',
                    ha="center", va="center", fontsize=5.5,
                    color="white", fontweight="bold", zorder=4)

    # Depth labels at bottom
    margin = 0.8
    for d in range(max_d + 1):
        x = d * 2.0
        label = "Root" if d == 0 else f"D{d}"
        ax.text(x, y_min - margin, label, ha="center", va="top",
                fontsize=9, fontweight="bold")

    method_label = problem_info.get("method", "")
    n_branches = problem_info["n_branches"]
    ax.set_title(f"{method_label}\n{n_branches} branches at depth 1", fontsize=11, pad=10)
    ax.set_xlim(-0.5, max_d * 2.0 + 0.5)
    ax.set_ylim(y_min - margin - 0.5, y_max + 0.5)
    ax.axis("off")


def main():
    flat_path = "faithful_baseline/results/flat_rollouts/inference_results.json"
    print(f"Loading {flat_path}...")
    with open(flat_path) as f:
        flat_data = json.load(f)

    results = flat_data["results"]
    model_name = flat_data.get("config", {}).get("model", "unknown")

    print("Loading tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct", trust_remote_code=True)

    # Import semantic clustering
    sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
    from node_accuracy_analysis import build_tree_semantic

    output_dir = "faithful_baseline/results/node_accuracy_v2"
    os.makedirs(output_dir, exist_ok=True)

    target_problems = [7, 18, 8, 9]
    threshold = 0.3
    step_size = 256

    # Set up OpenAI client for semantic clustering
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY not set")
        return

    from openai import AsyncOpenAI
    client = AsyncOpenAI(api_key=api_key)
    semaphore = asyncio.Semaphore(30)

    for pid in target_problems:
        entry = results[pid]
        level = entry.get("level", "?")
        print(f"\nProblem {pid} (Level {level})...")

        # Chunk rollouts
        problem_steps = {}
        for rollout in entry["rollouts"]:
            rid = rollout["rollout_id"]
            text = rollout.get("thinking", "") or rollout.get("full_text", "") or rollout.get("response", "")
            steps = chunk_text_into_steps(text, tokenizer, step_size=step_size)
            if steps:
                problem_steps[rid] = steps

        correctness = compute_rollout_correctness(entry["rollouts"], entry.get("answer", ""))
        overall_acc = sum(correctness.values()) / max(len(correctness), 1)

        # Build syntactic tree
        print(f"  Building syntactic tree (t={threshold})...")
        syn_tree = asyncio.run(build_tree_for_problem(
            problem_steps, None, None, use_llm=False, similarity_threshold=threshold,
        ))
        syn_branches = len(syn_tree.get("children", []))

        # Build semantic tree
        print(f"  Building semantic tree (embedding + GPT-4o)...")
        sem_tree = asyncio.run(build_tree_semantic(
            problem_steps, client, semaphore,
            embedding_model="text-embedding-3-small",
            judge_model="gpt-4o",
        ))
        sem_branches = len(sem_tree.get("children", []))

        print(f"  Syntactic: {syn_branches} branches | Semantic: {sem_branches} branches")

        # Draw side-by-side
        max_depth = 6
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 8))

        syn_info = {
            "pid": pid, "level": level, "overall_acc": overall_acc,
            "n_branches": syn_branches, "method": f"Syntactic (t={threshold})",
        }
        sem_info = {
            "pid": pid, "level": level, "overall_acc": overall_acc,
            "n_branches": sem_branches, "method": "Semantic (Embedding + GPT-4o)",
        }

        draw_sankey_on_ax(ax1, syn_tree, correctness, syn_info, max_depth=max_depth)
        draw_sankey_on_ax(ax2, sem_tree, correctness, sem_info, max_depth=max_depth)

        fig.suptitle(
            f"Problem {pid} (Level {level}) — How 128 Rollouts Diverge: Syntactic vs Semantic\n"
            f"Overall accuracy: {overall_acc:.0%} | "
            f"Width = rollout count | Green (>85% correct), Red (<15%), Gray (mixed)",
            fontsize=14, y=1.02,
        )

        # Colorbar
        sm = plt.cm.ScalarMappable(
            cmap=mcolors.LinearSegmentedColormap.from_list("acc", [
                (0.9, 0.1, 0.1), (0.6, 0.5, 0.2), (0.1, 0.9, 0.1)
            ]),
            norm=plt.Normalize(0, 1),
        )
        sm.set_array([])

        fig.tight_layout()
        output_path = os.path.join(output_dir, f"sankey_compare_problem_{pid}.png")
        fig.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"  Saved {output_path}")

    print(f"\nAll comparison Sankey diagrams saved to {output_dir}/")


if __name__ == "__main__":
    main()

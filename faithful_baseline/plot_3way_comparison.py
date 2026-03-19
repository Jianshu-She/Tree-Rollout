#!/usr/bin/env python3
"""3-way comparison: Syntactic vs Old-Semantic vs Pure-LLM Sankey trees.

For each of 10 problems (step_0 only), generate a figure with 3 rows
(syntactic / old-semantic / pure-LLM) showing how 128 rollouts diverge.
"""

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from faithful_baseline.plot_sankey_tree import (
    draw_sankey_on_ax,
    flatten_tree,
    acc_to_color,
    collect_leaf_rollout_ids,
)
from mcts_inference.utils import is_correct


PROBLEM_INDICES = [68, 73, 104, 124, 155, 361, 374, 377, 394, 450]
MAX_DEPTH = 10
RESULTS_DIR = os.path.join(os.path.dirname(__file__),
                           "results", "training_stages_math500")


def compute_correctness(rollouts, ground_truth):
    """Compute correctness dict for all rollouts."""
    correctness = {}
    for rollout in rollouts:
        rid = rollout["rollout_id"]
        text = rollout.get("full_text", "") or rollout.get("response", "")
        correctness[rid] = is_correct(text, ground_truth)
    return correctness


def get_tree_depth(tree, correctness, max_depth):
    """Get the actual max depth of nodes in a flattened tree."""
    nodes, _ = flatten_tree(tree, correctness, max_depth=max_depth)
    if not nodes:
        return 0
    return max(n["depth"] for n in nodes)


def main():
    # --- Load rollout data ---
    rollouts_path = os.path.join(RESULTS_DIR, "rollouts_step_0.json")
    print(f"Loading rollouts from {rollouts_path}...")
    with open(rollouts_path) as f:
        rollout_data = json.load(f)
    entries = rollout_data["results"]

    # Build lookup: problem_index -> position in list
    idx_to_pos = {}
    for pos, entry in enumerate(entries):
        idx_to_pos[entry["problem_index"]] = pos
    print(f"  {len(entries)} problems loaded, indices: "
          f"{[e['problem_index'] for e in entries]}")

    # --- Load cached syntactic trees (list, one per problem in order) ---
    syn_path = os.path.join(RESULTS_DIR, "trees_deep",
                            "cached_trees_step_0_syntactic.json")
    print(f"Loading syntactic trees from {syn_path}...")
    with open(syn_path) as f:
        syn_trees_list = json.load(f)

    # --- Load cached old-semantic trees ---
    sem_path = os.path.join(RESULTS_DIR, "trees_deep",
                            "cached_trees_step_0_semantic.json")
    print(f"Loading old-semantic trees from {sem_path}...")
    with open(sem_path) as f:
        sem_trees_list = json.load(f)

    # --- Load LLM-semantic trees (individual files) ---
    llm_dir = os.path.join(RESULTS_DIR, "trees_llm_semantic")
    print(f"Loading LLM-semantic trees from {llm_dir}...")
    llm_trees = {}
    for prob_idx in PROBLEM_INDICES:
        fpath = os.path.join(llm_dir, f"tree_step_0_problem_{prob_idx}.json")
        if os.path.exists(fpath):
            with open(fpath) as f:
                llm_trees[prob_idx] = json.load(f)
            print(f"  Loaded problem {prob_idx}")
        else:
            print(f"  WARNING: missing {fpath}")

    # --- Output directory ---
    output_dir = llm_dir  # save figures alongside LLM tree JSONs
    os.makedirs(output_dir, exist_ok=True)

    # --- Generate one figure per problem ---
    method_labels = ["Syntactic (t=0.3)", "Old Semantic (Emb+GPT-4o)", "Pure LLM Semantic"]
    method_keys = ["syntactic", "old_semantic", "llm_semantic"]

    for prob_idx in PROBLEM_INDICES:
        if prob_idx not in idx_to_pos:
            print(f"Problem {prob_idx}: not found in rollout data, skipping")
            continue

        pos = idx_to_pos[prob_idx]
        entry = entries[pos]
        gt = entry.get("ground_truth", entry.get("answer", ""))
        level = entry.get("level", "?")

        # Compute correctness
        correctness = compute_correctness(entry["rollouts"], gt)
        total_rids = set(range(len(entry["rollouts"])))
        n_correct = sum(1 for r in total_rids if correctness.get(r, False))
        overall_acc = n_correct / len(total_rids) if total_rids else 0

        # Gather trees
        trees = {
            "syntactic": syn_trees_list[pos],
            "old_semantic": sem_trees_list[pos],
            "llm_semantic": llm_trees.get(prob_idx),
        }

        # Skip if any tree is missing
        missing = [k for k, v in trees.items() if v is None]
        if missing:
            print(f"Problem {prob_idx}: missing trees for {missing}, skipping")
            continue

        # Compute actual depths for width proportions (capped at MAX_DEPTH)
        depths = {}
        for key in method_keys:
            depths[key] = get_tree_depth(trees[key], correctness, MAX_DEPTH)
            depths[key] = min(depths[key], MAX_DEPTH)
            depths[key] = max(depths[key], 1)

        # Use max depth across methods so all 3 rows share the same width
        max_actual_depth = max(depths.values())
        cell_width = max(4, max_actual_depth * 0.8 + 2.0)
        cell_height = 6

        fig, axes = plt.subplots(3, 1, figsize=(cell_width, cell_height * 3))

        for row, (key, label) in enumerate(zip(method_keys, method_labels)):
            tree = trees[key]
            n_branches = len(tree.get("children", []))
            depth = depths[key]

            info = {
                "pid": prob_idx,
                "level": level,
                "overall_acc": overall_acc,
                "n_branches": n_branches,
                "method": f"{label}\nAcc:{overall_acc:.0%}, {n_branches} branches",
            }
            draw_sankey_on_ax(axes[row], tree, correctness, info, max_depth=MAX_DEPTH)

        fig.suptitle(
            f"Problem {prob_idx} (Level {level}, answer={gt}) -- 3-Way Comparison\n"
            f"128 rollouts | Overall accuracy: {overall_acc:.0%} | "
            f"Green=correct, Red=incorrect, Gray=mixed",
            fontsize=13, y=1.02,
        )
        fig.tight_layout()

        out_path = os.path.join(output_dir, f"compare_3way_problem_{prob_idx}.png")
        fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"  Saved {out_path}")

    print(f"\nAll 3-way comparison figures saved to {output_dir}/")


if __name__ == "__main__":
    main()

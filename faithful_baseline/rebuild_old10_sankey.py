#!/usr/bin/env python3
"""Rebuild syntactic trees and Sankey plots for the original 10 problems.

Uses the fixed build_tree_for_problem that preserves terminated rollouts.
"""

import asyncio
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from transformers import AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "mcts_inference"))

from overlap_analysis.step_overlap_analysis import chunk_text_into_steps, build_tree_for_problem
from faithful_baseline.plot_sankey_tree import draw_sankey_on_ax, collect_leaf_rollout_ids, acc_to_color
from mcts_inference.utils import is_correct


RESULTS_DIR = "faithful_baseline/results/training_stages_math500"
OUTPUT_DIR = os.path.join(RESULTS_DIR, "trees_deep")
STAGES = [
    {"name": "step_0", "label": "Step 0"},
    {"name": "step_40", "label": "Step 40"},
    {"name": "step_80", "label": "Step 80"},
    {"name": "step_120", "label": "Step 120"},
]


def _get_actual_max_depth(tree, d=0):
    if not tree.get("children"):
        return d
    return max(_get_actual_max_depth(c, d + 1) for c in tree["children"])


async def main():
    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen2.5-Math-7B-Instruct", trust_remote_code=True
    )

    # Load rollouts for each stage
    all_data = {}
    for stage in STAGES:
        path = os.path.join(RESULTS_DIR, f"rollouts_{stage['name']}.json")
        if not os.path.exists(path):
            print(f"Skipping {stage['name']}: not found")
            continue
        with open(path) as f:
            all_data[stage["name"]] = json.load(f)

    # Get problem indices
    first_stage = list(all_data.values())[0]
    problem_indices = [e["problem_index"] for e in first_stage["results"]]
    print(f"Problems: {problem_indices}")

    # Build trees and plot for each problem
    for pi, prob_idx in enumerate(problem_indices):
        print(f"\nProblem {prob_idx} ({pi+1}/{len(problem_indices)})...")

        stage_trees = {}
        stage_correctness = {}
        stage_accs = {}

        for stage in STAGES:
            sname = stage["name"]
            if sname not in all_data:
                continue

            entry = all_data[sname]["results"][pi]
            gt = entry.get("ground_truth", entry.get("answer", ""))

            # Chunk rollouts
            problem_steps = {}
            for r in entry["rollouts"]:
                rid = r["rollout_id"]
                text = r.get("full_text", "") or r.get("response", "")
                steps = chunk_text_into_steps(text, tokenizer, step_size=256)
                if steps:
                    problem_steps[rid] = steps

            # Build tree
            tree = await build_tree_for_problem(
                problem_steps, None, None, use_llm=False, similarity_threshold=0.3
            )

            # Correctness
            corr = {}
            for r in entry["rollouts"]:
                rid = r["rollout_id"]
                text = r.get("full_text", "") or r.get("response", "")
                corr[rid] = is_correct(text, gt)

            leaf_rids = collect_leaf_rollout_ids(tree)
            acc = sum(1 for r in leaf_rids if corr.get(r, False)) / max(len(leaf_rids), 1)

            stage_trees[sname] = tree
            stage_correctness[sname] = corr
            stage_accs[sname] = acc

            root_n = len(leaf_rids)
            d1_br = len(tree.get("children", []))
            print(f"  {sname}: root={root_n}, branches={d1_br}, acc={acc:.0%}")

        # Save cached trees
        for sname, tree in stage_trees.items():
            cache_path = os.path.join(OUTPUT_DIR, f"cached_trees_{sname}_syntactic_fixed.json")
            # Append to list or create
            if os.path.exists(cache_path):
                with open(cache_path) as f:
                    cache = json.load(f)
            else:
                cache = []
            cache.append(tree)
            with open(cache_path, "w") as f:
                json.dump(cache, f)

        # Plot Sankey - one row with 4 stages
        stages_to_plot = [s for s in STAGES if s["name"] in stage_trees]

        # Compute independent widths
        stage_depths = []
        for s in stages_to_plot:
            d = _get_actual_max_depth(stage_trees[s["name"]])
            stage_depths.append(d)

        cell_widths = [max(3, d * 0.8 + 1.5) for d in stage_depths]
        total_width = sum(cell_widths) + 1
        cell_height = 10

        fig, axes = plt.subplots(
            1, len(stages_to_plot),
            figsize=(total_width, cell_height),
            gridspec_kw={"width_ratios": cell_widths},
        )
        if len(stages_to_plot) == 1:
            axes = [axes]

        for ax, stage, depth in zip(axes, stages_to_plot, stage_depths):
            sname = stage["name"]
            tree = stage_trees[sname]
            corr = stage_correctness[sname]
            acc = stage_accs[sname]
            n_br = len(tree.get("children", []))

            info = {
                "pid": prob_idx,
                "level": "?",
                "overall_acc": acc,
                "n_branches": n_br,
                "method": f"{stage['label']}\nAcc:{acc:.0%}, {n_br}br",
            }
            draw_sankey_on_ax(ax, tree, corr, info, max_depth=depth)

        level = "?"
        fig.suptitle(
            f"Problem {prob_idx} (Level {level})\n"
            f"128 rollouts | Syntactic (t=0.3) | Green=correct, Red=incorrect, Gray=mixed",
            fontsize=12,
        )
        plt.tight_layout()
        out_path = os.path.join(OUTPUT_DIR, f"sankey_problem_{prob_idx}.png")
        fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"  Saved {out_path}")

    print("\nDone!")


if __name__ == "__main__":
    asyncio.run(main())

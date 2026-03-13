#!/usr/bin/env python3
"""Generate comprehensive tree visualizations for training stage analysis.

1. Sankey evolution: all 10 problems × 4 stages (syntactic)
2. Sankey side-by-side: syntactic vs semantic for selected problems
3. Per-problem tree structure plots
"""

import asyncio
import json
import os
import sys
import time
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "mcts_inference"))

from overlap_analysis.step_overlap_analysis import (
    chunk_text_into_steps,
    build_tree_for_problem,
)
from faithful_baseline.node_accuracy_analysis import (
    build_tree_semantic,
    collect_leaf_rollout_ids,
)
from faithful_baseline.plot_sankey_tree import (
    draw_sankey_on_ax,
    flatten_tree,
    acc_to_color,
)
from mcts_inference.utils import is_correct


TRAINING_STAGES = [
    {"name": "step_0", "label": "Step 0\n(Base)", "short": "Step 0", "step": 0},
    {"name": "step_40", "label": "Step 40\n(Early)", "short": "Step 40", "step": 40},
    {"name": "step_80", "label": "Step 80\n(Mid)", "short": "Step 80", "step": 80},
    {"name": "step_120", "label": "Step 120\n(Late)", "short": "Step 120", "step": 120},
]


def load_and_prepare(results_dir, step_size=256):
    """Load all rollout data and prepare trees."""
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen2.5-Math-7B-Instruct", trust_remote_code=True)

    all_stage_data = {}
    for stage in TRAINING_STAGES:
        path = os.path.join(results_dir, f"rollouts_{stage['name']}.json")
        if not os.path.exists(path):
            continue
        with open(path) as f:
            data = json.load(f)

        entries = data["results"]
        stage_info = {"entries": entries, "problem_steps": [], "correctness": []}

        for entry in entries:
            problem_steps = {}
            for rollout in entry["rollouts"]:
                rid = rollout["rollout_id"]
                text = rollout.get("full_text", "") or rollout.get("response", "")
                steps = chunk_text_into_steps(text, tokenizer, step_size=step_size)
                if steps:
                    problem_steps[rid] = steps
            stage_info["problem_steps"].append(problem_steps)

            correctness = {}
            for rollout in entry["rollouts"]:
                rid = rollout["rollout_id"]
                text = rollout.get("full_text", "") or rollout.get("response", "")
                gt = entry.get("ground_truth", entry.get("answer", ""))
                correctness[rid] = is_correct(text, gt)
            stage_info["correctness"].append(correctness)

        all_stage_data[stage["name"]] = stage_info
        print(f"  Loaded {stage['short']}: {len(entries)} problems")

    return all_stage_data


def _trees_cache_path(output_dir, stage_name, method):
    return os.path.join(output_dir, f"cached_trees_{stage_name}_{method}.json")


def _save_trees_cache(trees, output_dir, stage_name, method):
    path = _trees_cache_path(output_dir, stage_name, method)
    with open(path, "w") as f:
        json.dump(trees, f)
    print(f"    Cached {method} trees -> {os.path.basename(path)}")


def _load_trees_cache(output_dir, stage_name, method):
    path = _trees_cache_path(output_dir, stage_name, method)
    if os.path.exists(path):
        with open(path) as f:
            trees = json.load(f)
        print(f"    Loaded cached {method} trees <- {os.path.basename(path)}")
        return trees
    return None


async def build_all_trees(all_stage_data, threshold=0.3, client=None, semaphore=None,
                           cache_dir=None):
    """Build syntactic and semantic trees for all stages and problems.
    Caches results to JSON so API calls aren't repeated."""
    all_trees = {}

    for stage in TRAINING_STAGES:
        if stage["name"] not in all_stage_data:
            continue
        sdata = all_stage_data[stage["name"]]

        print(f"\n  Building trees for {stage['short']}...")
        t0 = time.time()

        # Try loading from cache first
        syn_trees = _load_trees_cache(cache_dir, stage["name"], "syntactic") if cache_dir else None
        if syn_trees is None:
            syn_trees = []
            for i, ps in enumerate(sdata["problem_steps"]):
                syn_tree = await build_tree_for_problem(
                    ps, None, None, use_llm=False, similarity_threshold=threshold)
                syn_trees.append(syn_tree)
            if cache_dir:
                _save_trees_cache(syn_trees, cache_dir, stage["name"], "syntactic")

        sem_trees = _load_trees_cache(cache_dir, stage["name"], "semantic") if cache_dir else None
        if sem_trees is None:
            if client and semaphore:
                sem_trees = []
                for i, ps in enumerate(sdata["problem_steps"]):
                    sem_tree = await build_tree_semantic(
                        ps, client, semaphore,
                        embedding_model="text-embedding-3-small",
                        judge_model="gpt-4o")
                    sem_trees.append(sem_tree)
                if cache_dir:
                    _save_trees_cache(sem_trees, cache_dir, stage["name"], "semantic")
            else:
                sem_trees = [None] * len(sdata["problem_steps"])

        all_trees[stage["name"]] = {"syntactic": syn_trees, "semantic": sem_trees}
        print(f"    Done in {time.time()-t0:.1f}s")

    return all_trees


def _estimate_tree_height(tree, max_depth=6):
    """Estimate needed figure height based on max branches at any depth."""
    from faithful_baseline.plot_sankey_tree import flatten_tree
    # Use a dummy correctness (we only care about node count)
    from faithful_baseline.plot_sankey_tree import collect_leaf_rollout_ids
    rids = collect_leaf_rollout_ids(tree)
    dummy_corr = {r: True for r in rids}
    nodes, _ = flatten_tree(tree, dummy_corr, max_depth=max_depth)
    max_at_depth = 0
    from collections import Counter
    depth_counts = Counter(n["depth"] for n in nodes)
    max_at_depth = max(depth_counts.values()) if depth_counts else 1
    return max_at_depth


def plot_sankey_all_problems(all_stage_data, all_trees, problem_indices, output_dir, max_depth=6):
    """Generate Sankey evolution for ALL 10 problems (compact grid-overview style)."""
    os.makedirs(output_dir, exist_ok=True)
    stages = [s for s in TRAINING_STAGES if s["name"] in all_trees]

    for pid in range(len(problem_indices)):
        prob_idx = problem_indices[pid]
        n_stages = len(stages)

        # Compact fixed size like grid_overview cells
        fig, axes = plt.subplots(1, n_stages, figsize=(5 * n_stages, 4))
        if n_stages == 1:
            axes = [axes]

        for ax, stage in zip(axes, stages):
            tree = all_trees[stage["name"]]["syntactic"][pid]
            correctness = all_stage_data[stage["name"]]["correctness"][pid]

            total_rids = set(tree.get("rollout_ids", []))
            n_correct = sum(1 for r in total_rids if correctness.get(r, False))
            overall_acc = n_correct / len(total_rids) if total_rids else 0
            n_branches = len(tree.get("children", []))

            problem_info = {
                "pid": prob_idx,
                "level": all_stage_data[stage["name"]]["entries"][pid].get("level", "?"),
                "overall_acc": overall_acc,
                "n_branches": n_branches,
                "method": f"{stage['short']}\nAcc:{overall_acc:.0%}, {n_branches}br",
            }
            draw_sankey_on_ax(ax, tree, correctness, problem_info, max_depth=max_depth)

        level = all_stage_data[stages[0]["name"]]["entries"][pid].get("level", "?")
        gt = all_stage_data[stages[0]["name"]]["entries"][pid].get("ground_truth", "?")
        fig.suptitle(
            f"Problem {prob_idx} (Level {level}, answer={gt})\n"
            f"128 rollouts | Syntactic (t=0.3) | Green=correct, Red=incorrect, Gray=mixed",
            fontsize=12, y=1.02,
        )
        fig.tight_layout()
        path = os.path.join(output_dir, f"sankey_problem_{prob_idx}.png")
        fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"  Saved sankey_problem_{prob_idx}.png")


def plot_syn_vs_sem_sankey(all_stage_data, all_trees, problem_indices, output_dir,
                            selected_problems=None, max_depth=6):
    """For selected problems, show syntactic vs semantic side-by-side at each stage."""
    os.makedirs(output_dir, exist_ok=True)
    stages = [s for s in TRAINING_STAGES if s["name"] in all_trees]

    if selected_problems is None:
        selected_problems = list(range(min(5, len(problem_indices))))

    for pid in selected_problems:
        prob_idx = problem_indices[pid]
        has_sem = any(all_trees[s["name"]]["semantic"][pid] is not None for s in stages)
        if not has_sem:
            continue

        n_stages = len(stages)

        # Compact fixed size like grid_overview cells
        fig, axes = plt.subplots(2, n_stages, figsize=(5 * n_stages, 8))

        for col, stage in enumerate(stages):
            syn_tree = all_trees[stage["name"]]["syntactic"][pid]
            sem_tree = all_trees[stage["name"]]["semantic"][pid]
            correctness = all_stage_data[stage["name"]]["correctness"][pid]

            total_rids = set(syn_tree.get("rollout_ids", []))
            n_correct = sum(1 for r in total_rids if correctness.get(r, False))
            overall_acc = n_correct / len(total_rids) if total_rids else 0

            # Syntactic (top row)
            syn_branches = len(syn_tree.get("children", []))
            syn_info = {
                "pid": prob_idx, "level": "?", "overall_acc": overall_acc,
                "n_branches": syn_branches,
                "method": f"Syntactic\n{stage['short']}, Acc:{overall_acc:.0%}, {syn_branches}br",
            }
            draw_sankey_on_ax(axes[0, col], syn_tree, correctness, syn_info, max_depth=max_depth)

            # Semantic (bottom row)
            if sem_tree is not None:
                sem_branches = len(sem_tree.get("children", []))
                sem_info = {
                    "pid": prob_idx, "level": "?", "overall_acc": overall_acc,
                    "n_branches": sem_branches,
                    "method": f"Semantic\n{stage['short']}, Acc:{overall_acc:.0%}, {sem_branches}br",
                }
                draw_sankey_on_ax(axes[1, col], sem_tree, correctness, sem_info, max_depth=max_depth)
            else:
                axes[1, col].text(0.5, 0.5, "No semantic tree",
                                   ha="center", va="center", transform=axes[1, col].transAxes)
                axes[1, col].axis("off")

        level = all_stage_data[stages[0]["name"]]["entries"][pid].get("level", "?")
        gt = all_stage_data[stages[0]["name"]]["entries"][pid].get("ground_truth", "?")
        fig.suptitle(
            f"Problem {prob_idx} (Level {level}, answer={gt})\n"
            f"Top: Syntactic (t=0.3) | Bottom: Semantic (Emb+GPT-4o) | Green=correct, Red=incorrect",
            fontsize=12, y=1.02,
        )
        fig.tight_layout()
        path = os.path.join(output_dir, f"syn_vs_sem_problem_{prob_idx}.png")
        fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"  Saved syn_vs_sem_problem_{prob_idx}.png")


def plot_grid_overview(all_stage_data, all_trees, problem_indices, output_dir, max_depth=5):
    """10×4 grid overview: one small Sankey per (problem, stage) cell."""
    os.makedirs(output_dir, exist_ok=True)
    stages = [s for s in TRAINING_STAGES if s["name"] in all_trees]
    n_problems = len(problem_indices)
    n_stages = len(stages)

    fig, axes = plt.subplots(n_problems, n_stages, figsize=(5 * n_stages, 4 * n_problems))

    for row in range(n_problems):
        prob_idx = problem_indices[row]
        for col, stage in enumerate(stages):
            ax = axes[row, col]
            tree = all_trees[stage["name"]]["syntactic"][row]
            correctness = all_stage_data[stage["name"]]["correctness"][row]

            total_rids = set(tree.get("rollout_ids", []))
            n_correct = sum(1 for r in total_rids if correctness.get(r, False))
            overall_acc = n_correct / len(total_rids) if total_rids else 0
            n_branches = len(tree.get("children", []))

            info = {
                "pid": prob_idx, "level": "?", "overall_acc": overall_acc,
                "n_branches": n_branches,
                "method": f"Acc:{overall_acc:.0%}, {n_branches}br",
            }
            draw_sankey_on_ax(ax, tree, correctness, info, max_depth=max_depth)

            if row == 0:
                ax.set_title(f"{stage['short']}\n{info['method']}", fontsize=10, pad=8)
            else:
                ax.set_title(info["method"], fontsize=9, pad=4)

        # Row label
        level = all_stage_data[stages[0]["name"]]["entries"][row].get("level", "?")
        axes[row, 0].set_ylabel(f"P{prob_idx}\n(L{level})", fontsize=11,
                                  rotation=0, labelpad=40, va="center")

    fig.suptitle(
        "Tree Structure Overview: 10 MATH500 Problems × 4 Training Stages\n"
        "Syntactic clustering (t=0.3) | 128 rollouts each | "
        "Green=correct, Red=incorrect, Gray=mixed",
        fontsize=15, y=1.01,
    )
    fig.tight_layout()
    path = os.path.join(output_dir, "grid_overview.png")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved grid_overview.png")


async def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="faithful_baseline/results/training_stages_math500")
    parser.add_argument("--output_dir", default="faithful_baseline/results/training_stages_math500/trees")
    parser.add_argument("--syntactic_threshold", type=float, default=0.3)
    parser.add_argument("--max_depth", type=int, default=6)
    parser.add_argument("--use_semantic", action="store_true", default=False)
    parser.add_argument("--openai_api_key", default=None)
    parser.add_argument("--max_concurrent", type=int, default=30)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load data
    print("Loading data...")
    all_stage_data = load_and_prepare(args.results_dir)

    # Get problem indices
    first_stage = list(all_stage_data.keys())[0]
    problem_indices = [e["problem_index"] for e in all_stage_data[first_stage]["entries"]]
    print(f"Problem indices: {problem_indices}")

    # OpenAI client
    client = None
    semaphore = None
    if args.use_semantic:
        api_key = args.openai_api_key or os.environ.get("OPENAI_API_KEY")
        if api_key:
            from openai import AsyncOpenAI
            client = AsyncOpenAI(api_key=api_key)
            semaphore = asyncio.Semaphore(args.max_concurrent)

    # Build trees (with caching to avoid redundant API calls)
    cache_dir = args.output_dir
    print("\nBuilding trees (cached results will be reused)...")
    all_trees = await build_all_trees(
        all_stage_data, threshold=args.syntactic_threshold,
        client=client, semaphore=semaphore,
        cache_dir=cache_dir)

    # 1. Sankey evolution for ALL 10 problems
    print("\n=== Sankey Evolution (all problems) ===")
    plot_sankey_all_problems(
        all_stage_data, all_trees, problem_indices, args.output_dir,
        max_depth=args.max_depth)

    # 2. Syntactic vs Semantic side-by-side (works if trees exist from cache or API)
    has_sem = any(
        all_trees[s]["semantic"][0] is not None
        for s in all_trees
    )
    if has_sem:
        print("\n=== Syntactic vs Semantic Comparison ===")
        plot_syn_vs_sem_sankey(
            all_stage_data, all_trees, problem_indices, args.output_dir,
            selected_problems=list(range(len(problem_indices))),
            max_depth=args.max_depth)

    # 3. Grid overview
    print("\n=== Grid Overview ===")
    plot_grid_overview(
        all_stage_data, all_trees, problem_indices, args.output_dir,
        max_depth=min(5, args.max_depth))

    print(f"\nAll tree visualizations saved to {args.output_dir}/")


if __name__ == "__main__":
    asyncio.run(main())

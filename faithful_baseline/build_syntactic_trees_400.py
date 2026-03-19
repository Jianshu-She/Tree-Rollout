#!/usr/bin/env python3
"""Build syntactic trees for 400 MATH500 problems × 4 training stages.

Uses difflib heuristic clustering (no LLM calls needed).
Saves individual tree JSONs and aggregate statistics.
"""

import json
import os
import sys
import time
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "mcts_inference"))

import asyncio
from transformers import AutoTokenizer
from overlap_analysis.step_overlap_analysis import chunk_text_into_steps, build_tree_for_problem
from mcts_inference.utils import is_correct


def collect_leaf_rollout_ids(node):
    """Collect rollout IDs from leaf nodes."""
    if not node.get("children"):
        return set(node.get("rollout_ids", []))
    ids = set()
    for c in node["children"]:
        ids |= collect_leaf_rollout_ids(c)
    return ids


def tree_stats(tree):
    """Compute basic tree statistics."""
    def count_nodes(node):
        c = 1
        for child in node.get("children", []):
            c += count_nodes(child)
        return c

    def max_depth(node, d=0):
        if not node.get("children"):
            return d
        return max(max_depth(c, d + 1) for c in node["children"])

    def branching_at_depth(node, target_d, current_d=0):
        """Collect branching factors at a specific depth."""
        if current_d == target_d:
            n_children = len(node.get("children", []))
            return [n_children] if n_children > 0 else []
        bfs = []
        for c in node.get("children", []):
            bfs.extend(branching_at_depth(c, target_d, current_d + 1))
        return bfs

    md = max_depth(tree)
    nn = count_nodes(tree)
    d1_branches = len(tree.get("children", []))

    # Branching factor per depth
    bf_per_depth = {}
    for d in range(md + 1):
        bfs = branching_at_depth(tree, d)
        if bfs:
            bf_per_depth[d] = {
                "mean": sum(bfs) / len(bfs),
                "values": bfs,
            }

    return {
        "max_depth": md,
        "num_nodes": nn,
        "d1_branches": d1_branches,
        "bf_per_depth": bf_per_depth,
    }


async def build_trees_for_stage(stage_name, rollout_path, tokenizer, output_dir):
    """Build syntactic trees for all problems in a rollout file."""
    print(f"\n{'='*60}")
    print(f"Stage: {stage_name}")
    print(f"{'='*60}")

    with open(rollout_path) as f:
        data = json.load(f)

    entries = data["results"]
    print(f"  {len(entries)} problems, {data.get('num_rollouts', '?')} rollouts each")

    os.makedirs(output_dir, exist_ok=True)
    all_stats = []
    t0 = time.time()

    for i, entry in enumerate(entries):
        pid = entry["problem_index"]
        gt = entry.get("ground_truth", entry.get("answer", ""))

        # Check if tree already exists (resume support)
        tree_path = os.path.join(output_dir, f"tree_{stage_name}_p{pid}.json")
        if os.path.exists(tree_path):
            with open(tree_path) as f:
                tree = json.load(f)
            stats = tree_stats(tree)
        else:
            # Chunk rollouts into steps
            problem_steps = {}
            for rollout in entry["rollouts"]:
                rid = rollout["rollout_id"]
                text = rollout.get("full_text", "") or rollout.get("response", "")
                steps = chunk_text_into_steps(text, tokenizer, step_size=256)
                if steps:
                    problem_steps[rid] = steps

            if not problem_steps:
                print(f"  [{i+1}/{len(entries)}] Problem {pid}: no valid rollouts, skipping")
                continue

            # Build syntactic tree (no LLM)
            tree = await build_tree_for_problem(
                problem_steps,
                client=None,
                semaphore=None,
                use_llm=False,
                similarity_threshold=0.3,
            )

            # Save tree
            with open(tree_path, "w") as f:
                json.dump(tree, f)

            stats = tree_stats(tree)

        # Compute accuracy
        correctness = {}
        for rollout in entry["rollouts"]:
            rid = rollout["rollout_id"]
            text = rollout.get("full_text", "") or rollout.get("response", "")
            correctness[rid] = is_correct(text, gt)

        leaf_rids = collect_leaf_rollout_ids(tree)
        n_correct = sum(1 for r in leaf_rids if correctness.get(r, False))
        acc = n_correct / max(len(leaf_rids), 1)

        stats["problem_index"] = pid
        stats["accuracy"] = acc
        stats["num_rollouts"] = len(leaf_rids)
        stats["num_correct"] = n_correct
        all_stats.append(stats)

        if (i + 1) % 50 == 0 or i == 0:
            elapsed = time.time() - t0
            print(f"  [{i+1}/{len(entries)}] Problem {pid}: depth={stats['max_depth']}, "
                  f"d1_br={stats['d1_branches']}, nodes={stats['num_nodes']}, "
                  f"acc={acc:.0%} ({elapsed:.0f}s)")

    elapsed = time.time() - t0
    print(f"  Done: {len(all_stats)} trees in {elapsed:.1f}s")

    # Save aggregate stats
    # Remove bf_per_depth.values (too large) for the summary
    summary_stats = []
    for s in all_stats:
        s_copy = dict(s)
        if "bf_per_depth" in s_copy:
            s_copy["bf_per_depth"] = {
                d: {"mean": v["mean"], "count": len(v["values"])}
                for d, v in s_copy["bf_per_depth"].items()
            }
        summary_stats.append(s_copy)

    stats_path = os.path.join(output_dir, f"stats_{stage_name}.json")
    with open(stats_path, "w") as f:
        json.dump(summary_stats, f, indent=2)

    return all_stats


async def main():
    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen2.5-Math-7B-Instruct", trust_remote_code=True
    )

    rollout_dir = "faithful_baseline/results/math500_full/train"
    tree_base = os.path.join(rollout_dir, "trees_syntactic")

    stages = ["step_0", "step_40", "step_80", "step_120"]
    all_stage_stats = {}

    for stage in stages:
        rollout_path = os.path.join(rollout_dir, f"rollouts_{stage}.json")
        if not os.path.exists(rollout_path):
            print(f"Skipping {stage}: {rollout_path} not found")
            continue
        output_dir = os.path.join(tree_base, stage)
        stats = await build_trees_for_stage(stage, rollout_path, tokenizer, output_dir)
        all_stage_stats[stage] = stats

    # Print summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for stage, stats in all_stage_stats.items():
        depths = [s["max_depth"] for s in stats]
        d1_brs = [s["d1_branches"] for s in stats]
        accs = [s["accuracy"] for s in stats]
        print(f"\n{stage}:")
        print(f"  Avg depth: {sum(depths)/len(depths):.1f} (range {min(depths)}-{max(depths)})")
        print(f"  Avg D1 branches: {sum(d1_brs)/len(d1_brs):.1f} (range {min(d1_brs)}-{max(d1_brs)})")
        print(f"  Avg accuracy: {sum(accs)/len(accs):.1%}")


if __name__ == "__main__":
    asyncio.run(main())

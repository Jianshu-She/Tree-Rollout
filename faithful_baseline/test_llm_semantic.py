#!/usr/bin/env python3
"""Test pure LLM semantic clustering on the 10-problem dataset."""

import asyncio
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "mcts_inference"))

from transformers import AutoTokenizer
from overlap_analysis.step_overlap_analysis import chunk_text_into_steps
from faithful_baseline.node_accuracy_analysis import build_tree_llm_semantic, collect_leaf_rollout_ids
from mcts_inference.utils import is_correct


async def main():
    results_dir = "faithful_baseline/results/training_stages_math500"
    stage = "step_0"  # test on base model first

    # Load rollouts
    path = os.path.join(results_dir, f"rollouts_{stage}.json")
    with open(path) as f:
        data = json.load(f)

    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen2.5-Math-7B-Instruct", trust_remote_code=True)

    # OpenAI client
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY not set")
        return

    from openai import AsyncOpenAI
    client = AsyncOpenAI(api_key=api_key)
    semaphore = asyncio.Semaphore(30)

    entries = data["results"]
    print(f"Testing pure LLM semantic clustering on {len(entries)} problems ({stage})")
    print(f"Model: gpt-4o-mini\n")

    total_api_calls = 0
    total_input_tokens = 0

    for i, entry in enumerate(entries):
        pid = entry["problem_index"]
        gt = entry.get("ground_truth", entry.get("answer", ""))

        # Chunk rollouts into steps
        problem_steps = {}
        for rollout in entry["rollouts"]:
            rid = rollout["rollout_id"]
            text = rollout.get("full_text", "") or rollout.get("response", "")
            steps = chunk_text_into_steps(text, tokenizer, step_size=256)
            if steps:
                problem_steps[rid] = steps

        print(f"Problem {pid} ({len(problem_steps)} rollouts, "
              f"max {max(len(s) for s in problem_steps.values())} steps)...")

        t0 = time.time()
        tree = await build_tree_llm_semantic(
            problem_steps, client, semaphore, model="gpt-4o-mini")
        elapsed = time.time() - t0

        # Count tree stats
        def count_nodes(node):
            c = 1
            for child in node.get("children", []):
                c += count_nodes(child)
            return c

        def max_depth(node, d=0):
            if not node.get("children"):
                return d
            return max(max_depth(c, d+1) for c in node["children"])

        n_nodes = count_nodes(tree)
        md = max_depth(tree)
        n_branches = len(tree.get("children", []))

        # Accuracy
        correctness = {}
        for rollout in entry["rollouts"]:
            rid = rollout["rollout_id"]
            text = rollout.get("full_text", "") or rollout.get("response", "")
            correctness[rid] = is_correct(text, gt)
        rids = collect_leaf_rollout_ids(tree)
        acc = sum(1 for r in rids if correctness.get(r, False)) / max(len(rids), 1)

        print(f"  -> {n_branches} branches at D1, depth={md}, {n_nodes} nodes, "
              f"acc={acc:.0%}, {elapsed:.1f}s")

        # Save tree
        out_dir = os.path.join(results_dir, "trees_llm_semantic")
        os.makedirs(out_dir, exist_ok=True)
        tree_path = os.path.join(out_dir, f"tree_{stage}_problem_{pid}.json")
        with open(tree_path, "w") as f:
            json.dump(tree, f)

    print(f"\nDone! Trees saved to {out_dir}")


if __name__ == "__main__":
    asyncio.run(main())

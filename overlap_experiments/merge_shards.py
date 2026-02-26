#!/usr/bin/env python3
"""Merge inference result shards from parallel GPU runs into a single file."""

import argparse
import json
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard_dirs", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    all_results = []
    config = None
    total_tokens = 0
    total_elapsed = 0.0

    for shard_dir in args.shard_dirs:
        path = os.path.join(shard_dir, "inference_results.json")
        if not os.path.exists(path):
            print(f"WARNING: Missing shard {path}, skipping")
            continue
        with open(path) as f:
            data = json.load(f)
        if config is None:
            config = data.get("config", {})
        all_results.extend(data["results"])
        total_tokens += data.get("total_tokens_generated", 0)
        total_elapsed = max(total_elapsed, data.get("elapsed_seconds", 0))

    # Sort by problem_id
    all_results.sort(key=lambda x: x["problem_id"])

    merged = {
        "config": config,
        "num_problems": len(all_results),
        "total_tokens_generated": total_tokens,
        "elapsed_seconds": total_elapsed,
        "results": all_results,
    }

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(merged, f, indent=2)

    print(f"Merged {len(all_results)} problems from {len(args.shard_dirs)} shards")
    print(f"Total tokens: {total_tokens:,}")
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()

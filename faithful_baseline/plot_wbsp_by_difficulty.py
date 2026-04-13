#!/usr/bin/env python3
"""WBSP curves split by problem difficulty (easy / medium / hard).

Difficulty buckets are defined by each problem's flat-rollout accuracy at
step_120. The hypothesis: tree structure differs systematically between
problems the model has mastered and problems it still fails on — shallow
branching collapses for easy problems, while hard problems retain width.
"""

import json
import os
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from mcts_inference.utils import is_correct
from faithful_baseline.analyze_branching_factors import analyze_tree


STAGES = ["step_0", "step_40", "step_80", "step_120"]
STAGE_COLORS = {"step_0": "#e74c3c", "step_40": "#e67e22",
                "step_80": "#2ecc71", "step_120": "#3498db"}
STAGE_LABELS = {"step_0": "Base (step 0)", "step_40": "Step 40",
                "step_80": "Step 80", "step_120": "Step 120"}

BUCKETS = [
    ("Easy (acc ≥ 0.8 at step 120)", lambda a: a >= 0.8),
    ("Medium (0.5 ≤ acc < 0.8)", lambda a: 0.5 <= a < 0.8),
    ("Hard (acc < 0.5 at step 120)", lambda a: a < 0.5),
]

MAX_DEPTH = 15


def load_step_correctness(rollout_dir, stage):
    path = os.path.join(rollout_dir, f"rollouts_{stage}.json")
    with open(path) as f:
        data = json.load(f)
    per_problem = {}
    acc_by_problem = {}
    for entry in data["results"]:
        pid = entry["problem_index"]
        gt = entry.get("ground_truth", entry.get("answer", ""))
        pid_corr = {}
        n_correct = 0
        for r in entry["rollouts"]:
            rid = r["rollout_id"]
            text = r.get("full_text", "") or r.get("response", "")
            ok = is_correct(text, gt)
            pid_corr[rid] = ok
            if ok:
                n_correct += 1
        per_problem[pid] = pid_corr
        acc_by_problem[pid] = n_correct / max(1, len(entry["rollouts"]))
    return per_problem, acc_by_problem


def bucket_problem(acc):
    for name, fn in BUCKETS:
        if fn(acc):
            return name
    return None


def collect_per_stage_per_bucket(tree_base, rollout_dir):
    _, step120_acc = load_step_correctness(rollout_dir, "step_120")

    bucket_to_pids = defaultdict(set)
    for pid, acc in step120_acc.items():
        b = bucket_problem(acc)
        if b is not None:
            bucket_to_pids[b].add(pid)

    print("Bucket sizes (by step_120 accuracy):")
    for name, _ in BUCKETS:
        print(f"  {name}: {len(bucket_to_pids[name])}")

    # data[bucket][stage][depth] = {"widths": [...], "branching_factors": [...], ...}
    data = {b[0]: {s: defaultdict(lambda: defaultdict(list)) for s in STAGES} for b in BUCKETS}

    for stage in STAGES:
        stage_dir = os.path.join(tree_base, stage)
        if not os.path.exists(stage_dir):
            continue
        per_problem_corr, _ = load_step_correctness(rollout_dir, stage)

        tree_files = sorted(f for f in os.listdir(stage_dir) if f.startswith("tree_"))
        for tf in tree_files:
            pid = int(tf.split("_p")[1].split(".")[0])
            bucket = None
            for name, _ in BUCKETS:
                if pid in bucket_to_pids[name]:
                    bucket = name
                    break
            if bucket is None:
                continue

            with open(os.path.join(stage_dir, tf)) as f:
                tree = json.load(f)
            corr = per_problem_corr.get(pid, {})
            depth_stats, _ = analyze_tree(tree, corr)
            bucket_data = data[bucket][stage]
            for d, stats in depth_stats.items():
                if d > MAX_DEPTH:
                    continue
                bucket_data[d]["widths"].append(stats["width"])
                bucket_data[d]["branching_factors"].extend(stats["branching_factors"])
                bucket_data[d]["surviving"].append(stats["surviving_rollouts"])
                bucket_data[d]["purities"].extend(stats["purities"])
        print(f"  processed {stage}")
    return data, bucket_to_pids


def mean_std(vals):
    if not vals:
        return 0.0, 0.0
    return float(np.mean(vals)), float(np.std(vals))


def plot_wbsp_by_difficulty(data, bucket_to_pids, out_path):
    # 4 rows (W,B,S,P) x 3 cols (Easy/Med/Hard)
    fig, axes = plt.subplots(4, 3, figsize=(16, 14), sharex=True)
    metric_rows = [
        ("W(d)", "widths", "Width (# nodes)", None),
        ("B(d)", "branching_factors", "Branching Factor", None),
        ("S(d)", "surviving", "# Rollouts", None),
        ("P(d)", "purities", "Purity", (0.4, 1.05)),
    ]

    for col, (bucket_name, _) in enumerate(BUCKETS):
        n_pid = len(bucket_to_pids[bucket_name])
        for row, (title, key, ylabel, ylim) in enumerate(metric_rows):
            ax = axes[row, col]
            for stage in STAGES:
                bucket_data = data[bucket_name][stage]
                depths = sorted(d for d in bucket_data.keys() if d <= MAX_DEPTH)
                if not depths:
                    continue
                stats = [mean_std(bucket_data[d][key]) for d in depths]
                means = [m for m, _ in stats]
                ax.plot(depths, means, "o-",
                        color=STAGE_COLORS[stage],
                        label=STAGE_LABELS[stage], markersize=3)
                lower = [max(0, m - s) for m, s in stats]
                upper = [m + s for m, s in stats]
                if ylim:
                    upper = [min(ylim[1], u) for u in upper]
                ax.fill_between(depths, lower, upper,
                                color=STAGE_COLORS[stage], alpha=0.12, linewidth=0)
            ax.grid(True, alpha=0.3)
            if ylim:
                ax.set_ylim(*ylim)
            if row == 0:
                ax.set_title(f"{bucket_name}\nn={n_pid} problems", fontsize=11)
            if col == 0:
                ax.set_ylabel(f"{title}\n{ylabel}", fontsize=10)
            if row == 3:
                ax.set_xlabel("Depth")
            if row == 0 and col == 2:
                ax.legend(fontsize=8, loc="upper right")

    fig.suptitle(
        "WBSP curves split by problem difficulty (step_120 flat-rollout accuracy)\n"
        "Line = mean, shaded band = ±1 std across problems/nodes",
        fontsize=13,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved {out_path}")


def main():
    tree_base = "faithful_baseline/results/math500_full/train/trees_syntactic"
    rollout_dir = "faithful_baseline/results/math500_full/train"
    output_dir = "faithful_baseline/results/math500_full/train/branching_analysis"
    os.makedirs(output_dir, exist_ok=True)

    data, bucket_to_pids = collect_per_stage_per_bucket(tree_base, rollout_dir)
    out_path = os.path.join(output_dir, "tree_curves_WBSP_by_difficulty.png")
    plot_wbsp_by_difficulty(data, bucket_to_pids, out_path)

    # Also copy to the figures/ tree
    for d in ("figures/training_evolution", "figures/branching_analysis"):
        os.makedirs(d, exist_ok=True)
        dst = os.path.join(d, "tree_curves_WBSP_by_difficulty.png")
        with open(out_path, "rb") as fi, open(dst, "wb") as fo:
            fo.write(fi.read())
        print(f"Copied to {dst}")


if __name__ == "__main__":
    main()

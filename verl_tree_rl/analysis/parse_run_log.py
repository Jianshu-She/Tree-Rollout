#!/usr/bin/env python3
"""Parse a verl training log and plot training/eval curves for a single run.

Usage:
    python parse_run_log.py <log_path> <method_name> [out_dir]

Extracts step-level metrics from stdout `step:N - key:val - ...` lines and
validation metrics from the same lines (they appear on eval steps).
"""

import os
import re
import sys
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


METRIC_RE = re.compile(r"([A-Za-z][A-Za-z0-9_/@\-.]*?):([-0-9eE.+]+)")
STEP_LINE_RE = re.compile(r"step:(\d+)\s+-\s+(.*)")


def parse_log(path):
    """Return list of {step: int, ...metrics} dicts."""
    records = []
    with open(path, errors="ignore") as f:
        for line in f:
            m = STEP_LINE_RE.search(line)
            if not m:
                continue
            step = int(m.group(1))
            rest = m.group(2)
            rec = {"step": step}
            for key, val in METRIC_RE.findall(rest):
                try:
                    rec[key] = float(val)
                except ValueError:
                    continue
            records.append(rec)
    return records


def plot_curves(records, method, out_dir):
    steps = [r["step"] for r in records]

    val_accs = [(r["step"], r["val-core/math_dapo/acc/mean@4"])
                for r in records if "val-core/math_dapo/acc/mean@4" in r]

    panels = [
        ("critic/score/mean",          "Training Reward (score mean)"),
        ("critic/rewards/ratio_correct","Training Accuracy (ratio_correct)"),
        ("actor/entropy",              "Actor Entropy"),
        ("actor/pg_loss",              "PG Loss"),
        ("actor/grad_norm",            "Grad Norm"),
        ("actor/ppo_kl",               "PPO KL"),
        ("response_length/mean",       "Response Length (mean tokens)"),
        ("timing_s/step",              "Step Time (s)"),
    ]

    fig, axes = plt.subplots(3, 3, figsize=(16, 11))
    axes = axes.flatten()

    for i, (key, title) in enumerate(panels):
        ax = axes[i]
        xs = [r["step"] for r in records if key in r]
        ys = [r[key] for r in records if key in r]
        if not xs:
            ax.set_title(f"{title}\n(no data)", fontsize=10)
            continue
        ax.plot(xs, ys, lw=1.2, color="#2c7fb8")
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("step")
        ax.grid(alpha=0.3)

    # Val accuracy curve (big, separate)
    ax = axes[-1]
    if val_accs:
        vx, vy = zip(*val_accs)
        ax.plot(vx, [v * 100 for v in vy], "o-", color="#d73027",
                markersize=8, lw=2)
        for x, y in zip(vx, vy):
            ax.annotate(f"{y*100:.1f}%", (x, y*100),
                        textcoords="offset points", xytext=(0, 8),
                        fontsize=9, ha="center")
        ax.set_title(f"Val Accuracy on MATH-500 (n=4)", fontsize=12,
                     fontweight="bold")
        ax.set_xlabel("training step")
        ax.set_ylabel("% accuracy")
        ax.grid(alpha=0.3)
        ax.set_ylim(0, max(vy) * 120 if vy else 50)
    else:
        ax.set_title("Val Accuracy (no data)", fontsize=10)

    fig.suptitle(f"Training Curves: {method.upper()}",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    out_png = os.path.join(out_dir, f"curves_{method}.png")
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_png}")

    # Also save raw metrics JSON
    out_json = os.path.join(out_dir, f"metrics_{method}.json")
    with open(out_json, "w") as f:
        json.dump(records, f, indent=2)
    print(f"Saved {out_json}")


def main():
    if len(sys.argv) < 3:
        print("Usage: parse_run_log.py <log_path> <method_name> [out_dir]")
        sys.exit(1)
    path = sys.argv[1]
    method = sys.argv[2]
    out_dir = sys.argv[3] if len(sys.argv) > 3 else "verl_tree_rl/analysis"
    os.makedirs(out_dir, exist_ok=True)

    records = parse_log(path)
    print(f"Parsed {len(records)} step records from {path}")

    plot_curves(records, method, out_dir)


if __name__ == "__main__":
    main()

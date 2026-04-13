#!/usr/bin/env python3
"""Three-way distribution comparison for branching factors (ROADMAP item #5).

Qirong: don't claim NegBin is "the" distribution — compare candidates.
We fit three discrete distributions at each (stage, depth):

  1. Poisson(λ)             — 1 param, light tail, variance = mean
  2. Geometric(p)           — 1 param, heavy tail (memoryless), support {0,1,...}
  3. Negative Binomial(r,p) — 2 params, flexible heavy tail

Scoring: log-likelihood, AIC (2k − 2·LL), and a KS-like discrete
goodness-of-fit statistic (max |F_emp − F_fit|). Lower AIC is better;
AIC penalises NB's extra parameter, so it answers "does the extra
flexibility earn its keep."

Writes:
  - bf_distribution_comparison.json  (per stage/depth: LL, AIC, KS for all 3)
  - bf_distribution_comparison.png   (param curves, AIC gap, example fits)
"""

import json
import os
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats as sp_stats

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from faithful_baseline.fit_poisson_beta import extract_depth_data
from mcts_inference.utils import is_correct


STAGES = ["step_0", "step_40", "step_80", "step_120"]
STAGE_LABELS = {"step_0": "Base (step 0)", "step_40": "Step 40",
                "step_80": "Step 80", "step_120": "Step 120"}
STAGE_COLORS = {"step_0": "#e74c3c", "step_40": "#e67e22",
                "step_80": "#2ecc71", "step_120": "#3498db"}
DIST_COLORS = {"poisson": "#2c3e50", "geometric": "#16a085", "negbin": "#c0392b"}
MAX_DEPTH = 12


def fit_poisson(data):
    lam = float(np.mean(data))
    return {"lambda": lam} if lam > 0 else None


def fit_geometric(data):
    """Geometric on {0,1,2,...}, PMF = p(1-p)^k.
    MLE: p_hat = 1 / (1 + mean)."""
    mean = float(np.mean(data))
    if mean < 0:
        return None
    p = 1.0 / (1.0 + mean)
    p = min(max(p, 1e-6), 1 - 1e-6)
    return {"p": p}


def fit_negbin(data):
    """Negative binomial via method of moments."""
    mean = float(np.mean(data))
    var = float(np.var(data))
    if var <= mean or mean <= 0:
        return None
    p = mean / var
    r = mean * p / (1 - p)
    if r <= 0 or p <= 0 or p >= 1:
        return None
    return {"r": float(r), "p": float(p)}


def loglik_poisson(data, params):
    if params is None:
        return -np.inf
    return float(np.sum(sp_stats.poisson.logpmf(data, mu=params["lambda"])))


def loglik_geom(data, params):
    if params is None:
        return -np.inf
    # scipy geom PMF is on {1,2,...}: p(1-p)^(k-1). We want {0,1,...}.
    # Shift: logpmf(k+1, p) on our k gives PMF of k on {0,1,...}.
    return float(np.sum(sp_stats.geom.logpmf(data + 1, p=params["p"])))


def loglik_negbin(data, params):
    if params is None:
        return -np.inf
    return float(np.sum(sp_stats.nbinom.logpmf(data, n=params["r"], p=params["p"])))


def cdf_poisson(x, params):
    return sp_stats.poisson.cdf(x, mu=params["lambda"])


def cdf_geom(x, params):
    return sp_stats.geom.cdf(x + 1, p=params["p"])


def cdf_negbin(x, params):
    return sp_stats.nbinom.cdf(x, n=params["r"], p=params["p"])


def discrete_ks(data, cdf_fn, params):
    """Discrete KS-like statistic: max |F_emp(x) − F_fit(x)| over integer support."""
    if params is None or len(data) == 0:
        return float("inf")
    data_sorted = np.sort(data)
    n = len(data_sorted)
    support = np.arange(0, int(data_sorted.max()) + 2)
    f_emp = np.searchsorted(data_sorted, support, side="right") / n
    f_fit = cdf_fn(support, params)
    return float(np.max(np.abs(f_emp - f_fit)))


def aic(loglik, k):
    return 2 * k - 2 * loglik


def compare_one(data):
    """Return dict of results for all 3 distributions on one data array."""
    data = np.asarray(data, dtype=int)
    n = len(data)
    if n < 2:
        return None

    out = {"n": n, "mean": float(np.mean(data)), "var": float(np.var(data))}

    for name, fit_fn, ll_fn, cdf_fn, k in [
        ("poisson", fit_poisson, loglik_poisson, cdf_poisson, 1),
        ("geometric", fit_geometric, loglik_geom, cdf_geom, 1),
        ("negbin", fit_negbin, loglik_negbin, cdf_negbin, 2),
    ]:
        params = fit_fn(data)
        if params is None:
            out[name] = {"fit": None, "loglik": None, "aic": None, "ks": None}
            continue
        ll = ll_fn(data, params)
        out[name] = {
            "fit": params,
            "loglik": ll,
            "aic": aic(ll, k),
            "ks": discrete_ks(data, cdf_fn, params),
        }
    return out


def collect_bf_data(tree_base, rollout_dir):
    """Aggregate branching factors per stage/depth across 400 problems."""
    stage_correctness = {}
    for stage in STAGES:
        path = os.path.join(rollout_dir, f"rollouts_{stage}.json")
        with open(path) as f:
            raw = json.load(f)
        corr = {}
        for entry in raw["results"]:
            pid = entry["problem_index"]
            gt = entry.get("ground_truth", entry.get("answer", ""))
            pid_corr = {}
            for r in entry["rollouts"]:
                rid = r["rollout_id"]
                text = r.get("full_text", "") or r.get("response", "")
                pid_corr[rid] = is_correct(text, gt)
            corr[pid] = pid_corr
        stage_correctness[stage] = corr
        print(f"  loaded correctness {stage}")

    all_bf = {}
    for stage in STAGES:
        stage_dir = os.path.join(tree_base, stage)
        if not os.path.exists(stage_dir):
            continue
        bf_agg = defaultdict(list)
        tree_files = sorted(f for f in os.listdir(stage_dir) if f.startswith("tree_"))
        for tf in tree_files:
            pid = int(tf.split("_p")[1].split(".")[0])
            with open(os.path.join(stage_dir, tf)) as f:
                tree = json.load(f)
            corr = stage_correctness[stage].get(pid, {})
            bf_by_d, _ = extract_depth_data(tree, corr, max_d=MAX_DEPTH)
            for d, vals in bf_by_d.items():
                bf_agg[d].extend(vals)
        all_bf[stage] = dict(bf_agg)
        print(f"  processed {stage}: {sum(len(v) for v in bf_agg.values())} samples")
    return all_bf


def build_comparison(all_bf):
    """For each (stage, depth), run the 3-way comparison."""
    results = {}
    for stage in STAGES:
        stage_out = {}
        for d in range(MAX_DEPTH + 1):
            vals = all_bf.get(stage, {}).get(d, [])
            if not vals:
                continue
            cmp = compare_one(vals)
            if cmp is None:
                continue
            stage_out[str(d)] = cmp
        results[stage] = stage_out
    return results


def best_by_aic(entry):
    scores = {k: entry[k]["aic"] for k in ("poisson", "geometric", "negbin")
              if entry.get(k) and entry[k]["aic"] is not None}
    if not scores:
        return None
    return min(scores, key=scores.get)


def plot_comparison(results, all_bf, out_path):
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 4, hspace=0.45, wspace=0.35)

    # Row 0: AIC per depth, 4 stages
    for col, stage in enumerate(STAGES):
        ax = fig.add_subplot(gs[0, col])
        stage_r = results.get(stage, {})
        depths = sorted(int(d) for d in stage_r.keys())
        for dist in ("poisson", "geometric", "negbin"):
            aics = []
            dd = []
            for d in depths:
                e = stage_r[str(d)]
                val = e.get(dist, {}).get("aic")
                if val is not None:
                    aics.append(val)
                    dd.append(d)
            ax.plot(dd, aics, "o-", label=dist, color=DIST_COLORS[dist], markersize=4)
        ax.set_title(f"{STAGE_LABELS[stage]}\nAIC per depth (lower=better)", fontsize=10)
        ax.set_xlabel("Depth")
        ax.set_ylabel("AIC")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3)
        if col == 0:
            ax.legend(fontsize=8)

    # Row 1: ΔAIC from best (log1p) — shows how much worse each dist is than the winner
    for col, stage in enumerate(STAGES):
        ax = fig.add_subplot(gs[1, col])
        stage_r = results.get(stage, {})
        depths = sorted(int(d) for d in stage_r.keys())
        for dist in ("poisson", "geometric", "negbin"):
            gaps = []
            dd = []
            for d in depths:
                e = stage_r[str(d)]
                best = min(v["aic"] for k, v in e.items()
                           if k in ("poisson", "geometric", "negbin")
                           and isinstance(v, dict) and v.get("aic") is not None)
                val = e.get(dist, {}).get("aic")
                if val is not None:
                    gaps.append(val - best)
                    dd.append(d)
            ax.plot(dd, gaps, "o-", label=dist, color=DIST_COLORS[dist], markersize=4)
        ax.axhline(0, color="black", linestyle="--", alpha=0.5, linewidth=0.8)
        ax.set_title(f"{STAGE_LABELS[stage]}\nΔAIC from best (0 = winner)", fontsize=10)
        ax.set_xlabel("Depth")
        ax.set_ylabel("ΔAIC")
        ax.set_yscale("symlog", linthresh=1)
        ax.grid(True, alpha=0.3)

    # Row 2: example histograms with all 3 fits overlaid — pick D0, D1, D3, D5 for step_0
    example_depths = [0, 1, 3, 5]
    for col, d in enumerate(example_depths):
        ax = fig.add_subplot(gs[2, col])
        vals = np.array(all_bf.get("step_0", {}).get(d, []), dtype=int)
        if len(vals) == 0:
            ax.set_title(f"step_0 D{d}\n(no data)")
            continue
        max_val = min(int(np.percentile(vals, 99)) + 2, 50)
        bins = np.arange(-0.5, max_val + 1.5)
        ax.hist(vals, bins=bins, density=True, color="#bdc3c7",
                edgecolor="black", linewidth=0.3, label="Data", alpha=0.7)
        x = np.arange(0, max_val + 1)
        entry = results["step_0"][str(d)]
        if entry["poisson"]["fit"]:
            ax.plot(x, sp_stats.poisson.pmf(x, mu=entry["poisson"]["fit"]["lambda"]),
                    "o-", color=DIST_COLORS["poisson"], markersize=3,
                    label=f"Poisson λ={entry['poisson']['fit']['lambda']:.2f}")
        if entry["geometric"]["fit"]:
            p = entry["geometric"]["fit"]["p"]
            ax.plot(x, sp_stats.geom.pmf(x + 1, p=p),
                    "s-", color=DIST_COLORS["geometric"], markersize=3,
                    label=f"Geom p={p:.2f}")
        if entry["negbin"]["fit"]:
            r = entry["negbin"]["fit"]["r"]
            pp = entry["negbin"]["fit"]["p"]
            ax.plot(x, sp_stats.nbinom.pmf(x, n=r, p=pp),
                    "^-", color=DIST_COLORS["negbin"], markersize=3,
                    label=f"NB r={r:.2f},p={pp:.2f}")
        winner = best_by_aic(entry)
        ax.set_title(f"step_0 D{d}\nAIC winner: {winner}", fontsize=10)
        ax.set_xlabel("Branching Factor")
        ax.set_ylabel("Density")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        "Branching-Factor Distribution Comparison — Poisson vs Geometric vs NegBin\n"
        "(ROADMAP item #5: 3rd candidate + AIC model selection)",
        fontsize=14,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved {out_path}")


def print_winner_table(results):
    print("\n" + "=" * 90)
    print("AIC WINNERS per (stage, depth)  [P=Poisson, G=Geometric, N=NegBin]")
    print("=" * 90)
    header = f"{'depth':>6} | " + " | ".join(f"{STAGE_LABELS[s]:>15}" for s in STAGES)
    print(header)
    print("-" * 90)
    # win counts
    counts = {s: defaultdict(int) for s in STAGES}
    for d in range(MAX_DEPTH + 1):
        row = [f"{d:>6}"]
        for s in STAGES:
            e = results.get(s, {}).get(str(d))
            if not e:
                row.append(f"{'-':>15}")
                continue
            w = best_by_aic(e)
            counts[s][w] += 1
            tag = {"poisson": "P", "geometric": "G", "negbin": "N"}[w]
            nb_ll = e["negbin"].get("loglik")
            po_ll = e["poisson"].get("loglik")
            dll = f"{nb_ll - po_ll:+.0f}" if (nb_ll is not None and po_ll is not None) else "  n/a"
            row.append(f"{tag}  ΔLL(N−P)={dll:>7}".rjust(15))
        print(" | ".join(row))
    print("-" * 90)
    for s in STAGES:
        c = counts[s]
        print(f"{STAGE_LABELS[s]} winner totals: "
              f"Poisson={c['poisson']}  Geom={c['geometric']}  NegBin={c['negbin']}")


def main():
    tree_base = "faithful_baseline/results/math500_full/train/trees_syntactic"
    rollout_dir = "faithful_baseline/results/math500_full/train"
    output_dir = "faithful_baseline/results/math500_full/train/poisson_beta_analysis"
    os.makedirs(output_dir, exist_ok=True)

    print("Collecting branching-factor data...")
    all_bf = collect_bf_data(tree_base, rollout_dir)

    print("Running 3-way comparisons...")
    results = build_comparison(all_bf)

    json_path = os.path.join(output_dir, "bf_distribution_comparison.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved {json_path}")

    print_winner_table(results)

    png_path = os.path.join(output_dir, "bf_distribution_comparison.png")
    plot_comparison(results, all_bf, png_path)

    for d in ("figures/branching_analysis", "figures/distribution_fitting"):
        os.makedirs(d, exist_ok=True)
        dst = os.path.join(d, "bf_distribution_comparison.png")
        with open(png_path, "rb") as fi, open(dst, "wb") as fo:
            fo.write(fi.read())
        print(f"Copied to {dst}")


if __name__ == "__main__":
    main()

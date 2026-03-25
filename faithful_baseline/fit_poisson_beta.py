#!/usr/bin/env python3
"""Fit Poisson (branching factor) and Beta (node accuracy) distributions
at each depth of post-hoc syntactic trees from flat rollouts.

For 400 problems × 4 training stages (1600 trees total):
- At each depth d, collect all branching factors → fit Poisson(λ_d)
- At each depth d, collect all node accuracies → fit Beta(α_d, β_d)

Outputs distribution parameter tables and comparison plots.
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
from mcts_inference.utils import is_correct


def collect_leaf_rollout_ids(node):
    if not node.get("children"):
        return set(node.get("rollout_ids", []))
    ids = set()
    for c in node["children"]:
        ids |= collect_leaf_rollout_ids(c)
    return ids


def collect_nodes_at_depth(tree, target_depth, current_depth=0):
    if current_depth == target_depth:
        return [tree]
    nodes = []
    for c in tree.get("children", []):
        nodes.extend(collect_nodes_at_depth(c, target_depth, current_depth + 1))
    return nodes


def max_depth(node, d=0):
    if not node.get("children"):
        return d
    return max(max_depth(c, d + 1) for c in node["children"])


def extract_depth_data(tree, correctness, max_d=15):
    """Extract branching factors and node accuracies at each depth."""
    md = max_depth(tree)
    bf_by_depth = defaultdict(list)
    acc_by_depth = defaultdict(list)

    for d in range(min(md, max_d) + 1):
        nodes = collect_nodes_at_depth(tree, d)
        for node in nodes:
            children = node.get("children", [])
            # Branching factor: only for non-leaf nodes
            if children:
                bf_by_depth[d].append(len(children))

            # Node accuracy
            rids = list(collect_leaf_rollout_ids(node))
            if rids:
                n_correct = sum(1 for r in rids if correctness.get(r, False))
                acc = n_correct / len(rids)
                acc_by_depth[d].append(acc)

    return bf_by_depth, acc_by_depth


def fit_poisson(data):
    """Fit Poisson distribution. Returns lambda parameter."""
    if not data:
        return None, None
    data = np.array(data, dtype=float)
    lam = np.mean(data)
    return lam, data


def fit_negbin(data):
    """Fit Negative Binomial distribution via method of moments.
    NB parameterization: mean = r(1-p)/p, var = r(1-p)/p^2
    Returns (r, p) or (None, None) if fit fails."""
    if not data or len(data) < 2:
        return None, None, None
    data = np.array(data, dtype=float)
    mean = np.mean(data)
    var = np.var(data)
    if var <= mean or mean <= 0:
        # Not overdispersed, NB not better than Poisson
        return None, None, data
    # Method of moments: var/mean = 1/p => p = mean/var, r = mean*p/(1-p)
    p = mean / var
    r = mean * p / (1 - p)
    return r, p, data


def negbin_loglik(data, r, p):
    """Compute NB log-likelihood. scipy uses n=r, p=p (success prob)."""
    if r is None or p is None:
        return -np.inf
    try:
        return np.sum(sp_stats.nbinom.logpmf(data.astype(int), n=r, p=p))
    except:
        return -np.inf


def poisson_loglik(data, lam):
    """Compute Poisson log-likelihood."""
    if lam is None or lam <= 0:
        return -np.inf
    return np.sum(sp_stats.poisson.logpmf(data.astype(int), mu=lam))


def fit_beta(data):
    """Fit Beta distribution. Returns (alpha, beta) parameters."""
    if not data or len(data) < 2:
        return None, None, None
    data = np.array(data, dtype=float)
    # Clamp to (0, 1) exclusive for Beta fitting
    data = np.clip(data, 1e-6, 1 - 1e-6)
    try:
        a, b, loc, scale = sp_stats.beta.fit(data, floc=0, fscale=1)
        return a, b, data
    except Exception:
        # Fallback: method of moments
        mean = np.mean(data)
        var = np.var(data)
        if var == 0 or mean == 0 or mean == 1:
            return 1.0, 1.0, data
        common = mean * (1 - mean) / var - 1
        a = mean * common
        b = (1 - mean) * common
        return max(a, 0.01), max(b, 0.01), data


def main():
    tree_base = "faithful_baseline/results/math500_full/train/trees_syntactic"
    rollout_dir = "faithful_baseline/results/math500_full/train"
    output_dir = "faithful_baseline/results/math500_full/train/poisson_beta_analysis"
    os.makedirs(output_dir, exist_ok=True)

    stages = ["step_0", "step_40", "step_80", "step_120"]
    stage_labels = {
        "step_0": "Base (step 0)",
        "step_40": "Step 40",
        "step_80": "Step 80",
        "step_120": "Step 120",
    }
    colors = {
        "step_0": "#e74c3c",
        "step_40": "#e67e22",
        "step_80": "#2ecc71",
        "step_120": "#3498db",
    }
    MAX_DEPTH = 12

    # Load correctness data
    stage_correctness = {}
    for stage in stages:
        rollout_path = os.path.join(rollout_dir, f"rollouts_{stage}.json")
        with open(rollout_path) as f:
            data = json.load(f)
        corr = {}
        for entry in data["results"]:
            pid = entry["problem_index"]
            gt = entry.get("ground_truth", entry.get("answer", ""))
            pid_corr = {}
            for r in entry["rollouts"]:
                rid = r["rollout_id"]
                text = r.get("full_text", "") or r.get("response", "")
                pid_corr[rid] = is_correct(text, gt)
            corr[pid] = pid_corr
        stage_correctness[stage] = corr
        print(f"Loaded correctness for {stage}")

    # Collect branching factors and accuracies across all trees
    all_bf = {}  # stage -> depth -> list of branching factors
    all_acc = {}  # stage -> depth -> list of accuracies

    for stage in stages:
        stage_dir = os.path.join(tree_base, stage)
        tree_files = sorted(f for f in os.listdir(stage_dir) if f.startswith("tree_"))
        print(f"\n{stage}: processing {len(tree_files)} trees...")

        bf_agg = defaultdict(list)
        acc_agg = defaultdict(list)

        for tf in tree_files:
            pid = int(tf.split("_p")[1].split(".")[0])
            with open(os.path.join(stage_dir, tf)) as f:
                tree = json.load(f)
            corr = stage_correctness[stage].get(pid, {})
            bf_by_d, acc_by_d = extract_depth_data(tree, corr, max_d=MAX_DEPTH)
            for d in bf_by_d:
                bf_agg[d].extend(bf_by_d[d])
            for d in acc_by_d:
                acc_agg[d].extend(acc_by_d[d])

        all_bf[stage] = dict(bf_agg)
        all_acc[stage] = dict(acc_agg)

    # =========================================================================
    # Fit distributions and save parameters
    # =========================================================================
    fit_results = {}
    for stage in stages:
        stage_fits = {}
        for d in range(MAX_DEPTH + 1):
            entry = {}
            # Branching factor: Poisson + Negative Binomial
            bf_data = all_bf[stage].get(d, [])
            if bf_data:
                bf_arr = np.array(bf_data, dtype=float)
                lam, _ = fit_poisson(bf_data)
                entry["poisson_lambda"] = float(lam)
                entry["bf_mean"] = float(np.mean(bf_arr))
                entry["bf_var"] = float(np.var(bf_arr))
                entry["bf_std"] = float(np.std(bf_arr))
                entry["bf_count"] = len(bf_data)
                entry["dispersion"] = float(np.var(bf_arr) / max(np.mean(bf_arr), 1e-9))

                # Negative Binomial fit
                nb_r, nb_p, _ = fit_negbin(bf_data)
                if nb_r is not None:
                    entry["negbin_r"] = float(nb_r)
                    entry["negbin_p"] = float(nb_p)

                    # Log-likelihood comparison
                    ll_pois = poisson_loglik(bf_arr, lam)
                    ll_nb = negbin_loglik(bf_arr, nb_r, nb_p)
                    entry["ll_poisson"] = float(ll_pois)
                    entry["ll_negbin"] = float(ll_nb)
                    entry["ll_ratio"] = float(ll_nb - ll_pois)  # positive = NB better

            # Beta for accuracy
            acc_data = all_acc[stage].get(d, [])
            if acc_data:
                a, b, _ = fit_beta(acc_data)
                if a is not None:
                    entry["beta_alpha"] = float(a)
                    entry["beta_beta"] = float(b)
                entry["acc_mean"] = float(np.mean(acc_data))
                entry["acc_std"] = float(np.std(acc_data))
                entry["acc_count"] = len(acc_data)

            if entry:
                stage_fits[str(d)] = entry
        fit_results[stage] = stage_fits

    with open(os.path.join(output_dir, "fitted_parameters.json"), "w") as f:
        json.dump(fit_results, f, indent=2)
    print("\nSaved fitted_parameters.json")

    # Print summary table
    print("\n" + "=" * 100)
    print("FITTED DISTRIBUTION PARAMETERS (Poisson vs Negative Binomial)")
    print("=" * 100)
    for stage in stages:
        print(f"\n{stage_labels[stage]}:")
        print(f"{'Depth':>5} | {'Pois λ':>7} | {'NB r':>7} | {'NB p':>7} | "
              f"{'var/mean':>8} | {'ΔLL(NB)':>8} | "
              f"{'Beta α':>7} | {'Beta β':>7} | {'E[acc]':>7} | {'#bf':>6}")
        print("-" * 100)
        for d in range(MAX_DEPTH + 1):
            e = fit_results[stage].get(str(d), {})
            if not e:
                continue
            lam = e.get("poisson_lambda", 0)
            nb_r = e.get("negbin_r", 0)
            nb_p = e.get("negbin_p", 0)
            disp = e.get("dispersion", 0)
            dll = e.get("ll_ratio", 0)
            ba = e.get("beta_alpha", 0)
            bb = e.get("beta_beta", 0)
            am = e.get("acc_mean", 0)
            n = e.get("bf_count", 0)
            nb_r_s = f"{nb_r:.3f}" if nb_r else "  n/a"
            nb_p_s = f"{nb_p:.3f}" if nb_p else "  n/a"
            dll_s = f"{dll:.1f}" if dll else "  n/a"
            print(f"{d:>5} | {lam:>7.2f} | {nb_r_s:>7} | {nb_p_s:>7} | "
                  f"{disp:>8.2f} | {dll_s:>8} | "
                  f"{ba:>7.2f} | {bb:>7.2f} | {am:>7.3f} | {n:>6}")

    # =========================================================================
    # Plot 1: Parameter curves — Poisson λ, NB r, dispersion, Beta params
    # =========================================================================
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Panel (0,0): Poisson λ(d) vs NB mean = r(1-p)/p
    for stage in stages:
        depths = sorted(int(d) for d in fit_results[stage] if "poisson_lambda" in fit_results[stage][d])
        depths = [d for d in depths if d <= MAX_DEPTH]
        lambdas = [fit_results[stage][str(d)]["poisson_lambda"] for d in depths]
        axes[0, 0].plot(depths, lambdas, "o-", color=colors[stage],
                        label=stage_labels[stage], markersize=5)
    axes[0, 0].set_title("Poisson λ(d) = E[branching factor]")
    axes[0, 0].set_xlabel("Depth")
    axes[0, 0].set_ylabel("λ")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Panel (0,1): Dispersion = var/mean (1 = Poisson, >1 = overdispersed)
    for stage in stages:
        depths = sorted(int(d) for d in fit_results[stage] if "dispersion" in fit_results[stage][d])
        depths = [d for d in depths if d <= MAX_DEPTH]
        disps = [fit_results[stage][str(d)]["dispersion"] for d in depths]
        axes[0, 1].plot(depths, disps, "o-", color=colors[stage],
                        label=stage_labels[stage], markersize=5)
    axes[0, 1].axhline(y=1.0, color="black", linestyle="--", alpha=0.5, label="Poisson (var=mean)")
    axes[0, 1].set_title("Dispersion = Var/Mean\n(>1 means Poisson underestimates variance)")
    axes[0, 1].set_xlabel("Depth")
    axes[0, 1].set_ylabel("Var / Mean")
    axes[0, 1].legend(fontsize=8)
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_yscale("log")

    # Panel (1,0): NB r parameter (shape/dispersion)
    for stage in stages:
        depths = sorted(int(d) for d in fit_results[stage] if "negbin_r" in fit_results[stage][d])
        depths = [d for d in depths if d <= MAX_DEPTH]
        rs = [fit_results[stage][str(d)]["negbin_r"] for d in depths]
        axes[1, 0].plot(depths, rs, "o-", color=colors[stage],
                        label=stage_labels[stage], markersize=5)
    axes[1, 0].set_title("NegBin r(d) — shape parameter\n(smaller r = heavier tail)")
    axes[1, 0].set_xlabel("Depth")
    axes[1, 0].set_ylabel("r")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Panel (1,1): Log-likelihood ratio (NB - Poisson), positive = NB better
    for stage in stages:
        depths = sorted(int(d) for d in fit_results[stage] if "ll_ratio" in fit_results[stage][d])
        depths = [d for d in depths if d <= MAX_DEPTH]
        lrs = [fit_results[stage][str(d)]["ll_ratio"] for d in depths]
        axes[1, 1].plot(depths, lrs, "o-", color=colors[stage],
                        label=stage_labels[stage], markersize=5)
    axes[1, 1].axhline(y=0, color="black", linestyle="--", alpha=0.5)
    axes[1, 1].set_title("ΔLL = LL(NegBin) − LL(Poisson)\n(positive = NB fits better)")
    axes[1, 1].set_xlabel("Depth")
    axes[1, 1].set_ylabel("ΔLL")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    fig.suptitle("Branching Factor: Poisson vs Negative Binomial Comparison", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "parameter_curves.png"), dpi=150, bbox_inches="tight")
    print("Saved parameter_curves.png")

    # =========================================================================
    # Plot 2: Poisson vs NegBin fit — histograms + both PMFs at selected depths
    # =========================================================================
    selected_depths = [0, 1, 2, 4]
    fig, axes = plt.subplots(len(selected_depths), len(stages), figsize=(16, 3.5 * len(selected_depths)))

    for row, d in enumerate(selected_depths):
        for col, stage in enumerate(stages):
            ax = axes[row, col]
            bf_data = all_bf[stage].get(d, [])
            if not bf_data:
                ax.set_title(f"{stage_labels[stage]}, D{d}\n(no data)", fontsize=9)
                continue

            bf_arr = np.array(bf_data, dtype=float)
            lam = np.mean(bf_arr)
            nb_r, nb_p, _ = fit_negbin(bf_data)

            # Histogram
            max_val = min(int(np.percentile(bf_arr, 99)) + 2, 50)
            bins = np.arange(-0.5, max_val + 1.5, 1)
            ax.hist(bf_arr, bins=bins, density=True, color=colors[stage],
                    alpha=0.5, edgecolor="black", linewidth=0.5, label="Data")

            # Poisson PMF overlay
            x_range = np.arange(0, max_val + 1)
            poisson_pmf = sp_stats.poisson.pmf(x_range, lam)
            ax.plot(x_range, poisson_pmf, "k.--", linewidth=1.2, markersize=4,
                    label=f"Poisson(λ={lam:.1f})", alpha=0.7)

            # NegBin PMF overlay
            if nb_r is not None:
                nb_pmf = sp_stats.nbinom.pmf(x_range, n=nb_r, p=nb_p)
                ax.plot(x_range, nb_pmf, "r.-", linewidth=1.5, markersize=4,
                        label=f"NB(r={nb_r:.2f},p={nb_p:.2f})")
                disp = np.var(bf_arr) / max(np.mean(bf_arr), 1e-9)
                ax.set_title(f"{stage_labels[stage]}, D{d}\nvar/mean={disp:.1f}, n={len(bf_data)}", fontsize=9)
            else:
                disp = np.var(bf_arr) / max(np.mean(bf_arr), 1e-9)
                ax.set_title(f"{stage_labels[stage]}, D{d}\nvar/mean={disp:.1f} (≤1), n={len(bf_data)}", fontsize=9)

            ax.set_xlabel("Branching Factor")
            ax.set_ylabel("Density")
            ax.legend(fontsize=6)
            ax.grid(True, alpha=0.2)

    fig.suptitle("Poisson vs Negative Binomial Fit to Branching Factor P(bf | depth, stage)",
                 fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "poisson_vs_negbin_histograms.png"), dpi=150, bbox_inches="tight")
    print("Saved poisson_vs_negbin_histograms.png")

    # =========================================================================
    # Plot 3: Beta fit quality — histograms + fitted PDF at selected depths
    # =========================================================================
    fig, axes = plt.subplots(len(selected_depths), len(stages), figsize=(16, 3.5 * len(selected_depths)))

    for row, d in enumerate(selected_depths):
        for col, stage in enumerate(stages):
            ax = axes[row, col]
            acc_data = all_acc[stage].get(d, [])
            if not acc_data:
                ax.set_title(f"{stage_labels[stage]}, D{d}\n(no data)", fontsize=9)
                continue

            acc_arr = np.array(acc_data)
            acc_clamped = np.clip(acc_arr, 1e-6, 1 - 1e-6)

            try:
                a, b, _, _ = sp_stats.beta.fit(acc_clamped, floc=0, fscale=1)
            except:
                mean = np.mean(acc_clamped)
                var = np.var(acc_clamped)
                common = mean * (1 - mean) / max(var, 1e-6) - 1
                a = max(mean * common, 0.01)
                b = max((1 - mean) * common, 0.01)

            # Histogram
            bins = np.linspace(0, 1, 30)
            ax.hist(acc_arr, bins=bins, density=True, color=colors[stage],
                    alpha=0.6, edgecolor="black", linewidth=0.5, label="Data")

            # Beta PDF overlay
            x_range = np.linspace(0.001, 0.999, 200)
            beta_pdf = sp_stats.beta.pdf(x_range, a, b)
            # Clip extreme values for display
            beta_pdf = np.clip(beta_pdf, 0, np.percentile(beta_pdf, 99) * 2)
            ax.plot(x_range, beta_pdf, "k-", linewidth=1.5,
                    label=f"Beta(α={a:.2f},β={b:.2f})")

            ax.set_title(f"{stage_labels[stage]}, D{d}\nα={a:.2f}, β={b:.2f}, n={len(acc_data)}",
                         fontsize=9)
            ax.set_xlabel("Accuracy")
            ax.set_ylabel("Density")
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.2)

    fig.suptitle("Beta Fit to Node Accuracy Distribution P(acc | depth, stage)",
                 fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "beta_fit_histograms.png"), dpi=150, bbox_inches="tight")
    print("Saved beta_fit_histograms.png")

    # =========================================================================
    # Plot 4: Combined parameter summary — all key params in one figure
    # =========================================================================
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # (0,0): Poisson λ(d)
    for stage in stages:
        depths = sorted(int(d) for d in fit_results[stage] if "poisson_lambda" in fit_results[stage][d])
        depths = [d for d in depths if d <= MAX_DEPTH]
        lambdas = [fit_results[stage][str(d)]["poisson_lambda"] for d in depths]
        axes[0, 0].plot(depths, lambdas, "o-", color=colors[stage],
                        label=stage_labels[stage], markersize=5, linewidth=2)
    axes[0, 0].set_title("Poisson λ(d)")
    axes[0, 0].set_xlabel("Depth d")
    axes[0, 0].set_ylabel("λ")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # (0,1): NegBin r(d)
    for stage in stages:
        depths = sorted(int(d) for d in fit_results[stage] if "negbin_r" in fit_results[stage][d])
        depths = [d for d in depths if d <= MAX_DEPTH]
        rs = [fit_results[stage][str(d)]["negbin_r"] for d in depths]
        axes[0, 1].plot(depths, rs, "o-", color=colors[stage],
                        label=stage_labels[stage], markersize=5, linewidth=2)
    axes[0, 1].set_title("NegBin r(d)\n(smaller = heavier tail)")
    axes[0, 1].set_xlabel("Depth d")
    axes[0, 1].set_ylabel("r")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # (0,2): Dispersion ratio
    for stage in stages:
        depths = sorted(int(d) for d in fit_results[stage] if "dispersion" in fit_results[stage][d])
        depths = [d for d in depths if d <= MAX_DEPTH]
        disps = [fit_results[stage][str(d)]["dispersion"] for d in depths]
        axes[0, 2].plot(depths, disps, "o-", color=colors[stage],
                        label=stage_labels[stage], markersize=5, linewidth=2)
    axes[0, 2].axhline(y=1.0, color="black", linestyle="--", alpha=0.5)
    axes[0, 2].set_title("Var/Mean (Dispersion)\n(=1 if Poisson holds)")
    axes[0, 2].set_xlabel("Depth d")
    axes[0, 2].set_ylabel("Var / Mean")
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    axes[0, 2].set_yscale("log")

    # (1,0): Beta mean = α/(α+β)
    for stage in stages:
        depths = sorted(int(d) for d in fit_results[stage] if "beta_alpha" in fit_results[stage][d])
        depths = [d for d in depths if d <= MAX_DEPTH]
        means = [fit_results[stage][str(d)]["beta_alpha"] /
                 (fit_results[stage][str(d)]["beta_alpha"] + fit_results[stage][str(d)]["beta_beta"])
                 for d in depths]
        axes[1, 0].plot(depths, means, "o-", color=colors[stage],
                        label=stage_labels[stage], markersize=5, linewidth=2)
    axes[1, 0].set_title("Beta Mean = α/(α+β)\n(Expected node accuracy)")
    axes[1, 0].set_xlabel("Depth d")
    axes[1, 0].set_ylabel("E[accuracy]")
    axes[1, 0].set_ylim(0, 1.05)
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # (1,1): Beta concentration = α+β
    for stage in stages:
        depths = sorted(int(d) for d in fit_results[stage] if "beta_alpha" in fit_results[stage][d])
        depths = [d for d in depths if d <= MAX_DEPTH]
        conc = [fit_results[stage][str(d)]["beta_alpha"] + fit_results[stage][str(d)]["beta_beta"]
                for d in depths]
        axes[1, 1].plot(depths, conc, "o-", color=colors[stage],
                        label=stage_labels[stage], markersize=5, linewidth=2)
    axes[1, 1].set_title("Beta Concentration α+β\n(higher = more peaked)")
    axes[1, 1].set_xlabel("Depth d")
    axes[1, 1].set_ylabel("α + β")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    # (1,2): LL ratio (NB - Poisson)
    for stage in stages:
        depths = sorted(int(d) for d in fit_results[stage] if "ll_ratio" in fit_results[stage][d])
        depths = [d for d in depths if d <= MAX_DEPTH]
        lrs = [fit_results[stage][str(d)]["ll_ratio"] for d in depths]
        axes[1, 2].plot(depths, lrs, "o-", color=colors[stage],
                        label=stage_labels[stage], markersize=5, linewidth=2)
    axes[1, 2].axhline(y=0, color="black", linestyle="--", alpha=0.5)
    axes[1, 2].set_title("ΔLL = LL(NB) − LL(Poisson)\n(>0 = NB better)")
    axes[1, 2].set_xlabel("Depth d")
    axes[1, 2].set_ylabel("ΔLL")
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)

    fig.suptitle("Flat Rollout Trees: Poisson/NegBin (branching) + Beta (accuracy) Parameters\n"
                 "400 problems × 4 training stages",
                 fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "parameter_summary.png"), dpi=150, bbox_inches="tight")
    print("Saved parameter_summary.png")

    print("\nDone!")


if __name__ == "__main__":
    main()

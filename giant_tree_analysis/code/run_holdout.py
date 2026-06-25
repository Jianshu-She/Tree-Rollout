"""
Leakage-free small-tree analysis: ground-truth V from a TRUTH half of leaves,
small tree samples only from the disjoint EVAL half. Reports |A|-weighted
sign-recovery (wsign) per schedule, averaged over in-band trees.
"""
import glob, numpy as np, pandas as pd
from tree_loader import Tree
from simulator import Simulator

corr = pd.read_parquet("correctness.parquet")
vroot = corr.groupby("prompt_idx").correct.mean()

sims = []
for p in sorted(glob.glob("prompt_*.json")):
    pid = int(p.split("_")[1].split(".")[0])
    if not (0.1 <= vroot.get(pid, 0) <= 0.9):
        continue
    t = Tree(p)
    t.apply_grades(corr[corr.prompt_idx == pid].sort_values("leaf_index")["correct"].to_numpy())
    s = Simulator(t); s.setup_holdout(seed=0)
    sims.append(s)
print("in-band trees:", len(sims))

def wsign_for(sim, schedule, n_seeds=12):
    ah, at = [], []
    for sd in range(n_seeds):
        E = sim.sample_ho(schedule, np.random.default_rng(100 + sd))
        ah += [e["A_hat"] for e in E]; at += [e["A_true"] for e in E]
    ah = np.array(ah); at = np.array(at)
    m = np.abs(at) > 1e-9
    if m.sum() < 5:
        return None
    w = np.abs(at[m])
    return np.sum(w * (np.sign(ah[m]) == np.sign(at[m]))) / w.sum()

SCHED = [
    [4, 16, 16, 16], [4, 8, 8, 8], [2, 2, 8, 8], [2, 2, 4, 8],
    [4, 4, 4, 4], [2, 4, 4, 4], [2, 2, 2, 4],
    [2, 2, 2, 2], [8, 8, 4, 1], [8, 4, 4, 1], [4, 4, 2, 1],
]
rows = []
print(f"\n{'schedule':16s}{'leaves':>7s}{'b3':>4s}{'wsign(held-out)':>17s}")
for S in SCHED:
    vals = [wsign_for(s, S) for s in sims]
    vals = [v for v in vals if v is not None]
    if not vals:
        continue
    m, sd = float(np.mean(vals)), float(np.std(vals))
    rows.append(dict(schedule=str(S), leaves=int(np.prod(S)), b3=S[3], wsign=m, sd=sd))
    print(f"{str(S):16s}{int(np.prod(S)):7d}{S[3]:4d}{m:12.3f}±{sd:.2f}")
pd.DataFrame(rows).to_csv("holdout_sweep.csv", index=False)
print("\nsaved holdout_sweep.csv")

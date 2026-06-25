"""
Two things per schedule, over in-band trees (drop unsolvable prompts already):
  yield   = P(a sampled small tree has signal = has BOTH a correct and a wrong leaf)
  wsign|S = |A|-weighted sign-recovery, computed ONLY on the with-signal trees
Uses held-out ground truth (truth half) + eval-half sampling (leakage-free).
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

def select_eval(sim, schedule, rng):
    chosen = []
    def descend(node, depth):
        if depth == len(schedule) or not node.children:
            ev = [li for li in node.leaf_idxs if not sim.is_truth[li]]
            if ev:
                chosen.append(int(rng.choice(ev)))
            return
        kids = list(node.children.values())
        b = min(schedule[depth], len(kids))
        for ci in rng.choice(len(kids), size=b, replace=False):
            descend(kids[ci], depth + 1)
    descend(sim.t.root, 0)
    return chosen

def edges_wsign(sim, chosen):
    cnt = {}; cor = {}
    for li in chosen:
        ok = int(sim.grades[li])
        for nid in sim.leaf_path[li]:
            cnt[nid] = cnt.get(nid, 0) + 1; cor[nid] = cor.get(nid, 0) + ok
    Vhat = {n: cor[n] / cnt[n] for n in cnt}
    seen = set(); ah = []; at = []
    for li in chosen:
        path = sim.leaf_path[li]
        for d in range(len(path) - 1):
            pk, ck = path[d], path[d + 1]
            if (pk, ck) in seen:
                continue
            seen.add((pk, ck))
            a = sim.Vtrue_ho[ck] - sim.Vtrue_ho[pk]
            if np.isnan(a):
                continue
            ah.append(Vhat[ck] - Vhat[pk]); at.append(a)
    ah = np.array(ah); at = np.array(at); m = np.abs(at) > 1e-9
    if m.sum() < 3:
        return None
    w = np.abs(at[m])
    return float(np.sum(w * (np.sign(ah[m]) == np.sign(at[m]))) / w.sum())

SCHED = [[4, 16, 16, 16], [2, 2, 8, 8], [4, 4, 4, 4], [2, 2, 2, 4],
         [2, 2, 2, 2], [8, 8, 4, 1], [4, 4, 2, 1]]
print(f"\n{'schedule':16s}{'leaves':>7s}{'b3':>4s}{'yield(has signal)':>18s}{'wsign|signal':>14s}")
for S in SCHED:
    n_have = 0; n_tot = 0; wsigns = []
    for sim in sims:
        for sd in range(20):
            rng = np.random.default_rng(7000 + sd)
            chosen = select_eval(sim, S, rng)
            n_tot += 1
            g = sim.grades[chosen]
            has = (g.sum() > 0) and (g.sum() < len(g))   # both correct & wrong present
            if has:
                n_have += 1
                w = edges_wsign(sim, chosen)
                if w is not None:
                    wsigns.append(w)
    print(f"{str(S):16s}{int(np.prod(S)):7d}{S[3]:4d}{n_have/n_tot:17.2f}{np.mean(wsigns):14.3f}")

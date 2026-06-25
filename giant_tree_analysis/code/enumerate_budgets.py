"""Budget-fixed shape sweep. Reads data from MCTS (read-only), writes here."""
import sys, glob, gc, os, numpy as np, pandas as pd
MCTS = "/home/jianshu.she/mcts"
sys.path.insert(0, MCTS)
from tree_loader import Tree
from simulator import Simulator

corr = pd.read_parquet(os.path.join(MCTS, "correctness.parquet"))
vroot = corr.groupby("prompt_idx").correct.mean()

FACT = [1, 2, 4, 8, 16]
BUDGETS = [64, 128, 256]
def shapes_for(B):
    out = []
    for b0 in [1, 2, 4]:
        for b1 in FACT:
            for b2 in FACT:
                for b3 in FACT:
                    if b0 * b1 * b2 * b3 == B:
                        out.append((b0, b1, b2, b3))
    return out
ALL = sorted({s for B in BUDGETS for s in shapes_for(B)})
print("shapes to test:", len(ALL), flush=True)

def select_eval(sim, S, rng):
    chosen = []
    def descend(node, depth):
        if depth == len(S) or not node.children:
            ev = [li for li in node.leaf_idxs if not sim.is_truth[li]]
            if ev:
                chosen.append(int(rng.choice(ev)))
            return
        kids = list(node.children.values())
        for ci in rng.choice(len(kids), size=min(S[depth], len(kids)), replace=False):
            descend(kids[ci], depth + 1)
    descend(sim.t.root, 0)
    return chosen

def tree_wsign(sim, chosen):
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
            if not np.isnan(a):
                ah.append(Vhat[ck] - Vhat[pk]); at.append(a)
    ah = np.array(ah); at = np.array(at); m = np.abs(at) > 1e-9
    if m.sum() < 3:
        return None
    w = np.abs(at[m])
    return float(np.sum(w * (np.sign(ah[m]) == np.sign(at[m]))) / w.sum())

paths = [(p, int(p.split("_")[1].split(".")[0])) for p in sorted(glob.glob(os.path.join(MCTS, "prompt_*.json")))]
paths = [(p, pid) for p, pid in paths if 0.1 <= vroot.get(pid, 0) <= 0.9]
print("in-band trees:", len(paths), flush=True)

acc = {s: {"wsign": [], "yield": []} for s in ALL}
N_SEEDS = 16
for k, (p, pid) in enumerate(paths):
    try:
        t = Tree(p)
        t.apply_grades(corr[corr.prompt_idx == pid].sort_values("leaf_index")["correct"].to_numpy())
        sim = Simulator(t); sim.setup_holdout(seed=0)
    except Exception as e:
        print("  skip", p, e, flush=True); continue
    for S in ALL:
        ws = []; have = 0; tot = 0
        for sd in range(N_SEEDS):
            chosen = select_eval(sim, S, np.random.default_rng(7000 + sd))
            tot += 1
            g = sim.grades[chosen]
            if 0 < g.sum() < len(g):
                have += 1
                w = tree_wsign(sim, chosen)
                if w is not None:
                    ws.append(w)
        if ws:
            acc[S]["wsign"].append(np.mean(ws))
        acc[S]["yield"].append(have / tot)
    del t, sim; gc.collect()
    print(f"  [{k+1}/{len(paths)}] done", flush=True)

rows = []
for s in ALL:
    a = acc[s]
    rows.append(dict(b0=s[0], b1=s[1], b2=s[2], b3=s[3], budget=int(np.prod(s)),
                     wsign=np.mean(a["wsign"]) if a["wsign"] else np.nan,
                     yield_=np.mean(a["yield"]), n=len(a["wsign"])))
df = pd.DataFrame(rows)
df.to_csv("budget_enum.csv", index=False)
for B in BUDGETS:
    sub = df[df.budget == B].sort_values("wsign", ascending=False)
    print(f"\n===== budget = {B} rollouts =====", flush=True)
    print(f"{'shape':18s}{'b3':>4s}{'wsign(quality)':>16s}{'yield':>9s}")
    for _, r in sub.iterrows():
        print(f"[{r.b0},{r.b1},{r.b2},{r.b3}]".ljust(18)
              + f"{int(r.b3):4d}{r.wsign:16.3f}{r.yield_:9.2f}", flush=True)
print(f"\nsaved budget_enum.csv (trees={len(paths)})", flush=True)

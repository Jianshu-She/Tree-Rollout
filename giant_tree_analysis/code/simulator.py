"""
Offline advantage-recovery simulator.

Treats each giant tree's full-leaf V as GROUND TRUTH, then asks: under a token
budget, how well does a NARROWER tree (the kind RL can actually afford) recover
the ground-truth per-step advantage field A(u->v) = V(v) - V(u)?

A budget tree is specified by a branching schedule S = [b0,b1,b2,b3] (children
kept at depths 0,1,2,3; must be <= the giant tree's [4,16,16,16]). It is
realized by randomly sub-selecting children of the full tree, so every sampled
node/edge has a matching ground-truth value. Token cost is prefix-shared
(chunk-level dedup) because reused prefixes are the whole point of a tree vs N
independent rollouts.

Only depths 0->1->2->3 carry non-trivial advantage (the giant tree is linear
past depth 4, so deeper steps have A==0 by construction -- the known blind spot
that needs deeper re-collection to study).
"""
from __future__ import annotations
import numpy as np
from tree_loader import Tree, CHUNK, BRANCH_DEPTH


class Simulator:
    def __init__(self, tree: Tree):
        assert hasattr(tree, "grades"), "call tree.apply_grades(...) first"
        self.t = tree
        self.leaves = tree.leaves
        self.grades = np.asarray(tree.grades, dtype=np.int8)
        # assign each node an int id; precompute per-leaf branch path (int ids)
        self.kid = {k: i for i, k in enumerate(tree.index.keys())}   # key->int
        self.Vtrue = np.array([tree.index[k].V() for k in self.kid])  # by int id
        self.leaf_path = []   # leaf_path[li] = [id at depth0..maxdepth]
        self.leaf_tail = []   # tokens generated beyond BRANCH_DEPTH (unique/leaf)
        self.node_inc = {}    # int id -> incremental tokens at that node
        for leaf in self.leaves:
            tok = leaf["token_ids"]; n = len(tok)
            path = [self.kid[()]]
            for depth in range(1, BRANCH_DEPTH + 1):
                key = tuple(tok[: depth * CHUNK])
                if key not in self.kid:
                    break
                nid = self.kid[key]
                path.append(nid)
                self.node_inc[nid] = min(CHUNK, n - (depth - 1) * CHUNK)
                if n <= depth * CHUNK:
                    break
            self.leaf_path.append(path)
            self.leaf_tail.append(max(0, n - BRANCH_DEPTH * CHUNK))

    # ---- realize a budget tree by sub-selecting children ----------------
    def sample(self, schedule, rng):
        """Return (edges, token_cost). edges: list of dicts with A_hat/A_true."""
        chosen = self._select_leaves(schedule, rng)
        if not chosen:
            return [], 0
        g = self.grades
        cnt = {}; cor = {}
        for li in chosen:
            ok = int(g[li])
            for nid in self.leaf_path[li]:
                cnt[nid] = cnt.get(nid, 0) + 1
                cor[nid] = cor.get(nid, 0) + ok
        Vhat = {nid: cor[nid] / cnt[nid] for nid in cnt}
        seen = set(); E = []
        for li in chosen:
            path = self.leaf_path[li]
            for d in range(len(path) - 1):
                pk, ck = path[d], path[d + 1]
                if (pk, ck) in seen:
                    continue
                seen.add((pk, ck))
                E.append(dict(depth=d,
                              A_hat=Vhat[ck] - Vhat[pk],
                              A_true=self.Vtrue[ck] - self.Vtrue[pk],
                              n_child=cnt[ck]))
        return E, self._token_cost(chosen)

    def _select_leaves(self, schedule, rng):
        out = []
        def descend(node, depth):
            if depth == len(schedule) or not node.children:
                # take one representative leaf per deepest selected node
                out.append(rng.choice(node.leaf_idxs))
                return
            kids = list(node.children.values())
            b = min(schedule[depth], len(kids))
            for ci in rng.choice(len(kids), size=b, replace=False):
                descend(kids[ci], depth + 1)
        descend(self.t.root, 0)
        return out

    def _token_cost(self, leaf_idxs):
        # prefix-shared: each distinct branch node counted once + per-leaf tail
        branch = set()
        tail = 0
        for li in leaf_idxs:
            branch.update(self.leaf_path[li][1:])
            tail += self.leaf_tail[li]
        return sum(self.node_inc[nid] for nid in branch) + tail

    # ---- HELD-OUT (leakage-free) variant -------------------------------
    def setup_holdout(self, seed):
        """Split leaves: truth half defines ground-truth V; eval half is the
        only pool the small tree may sample from (disjoint -> no leakage)."""
        rng = np.random.default_rng(seed)
        self.is_truth = rng.integers(0, 2, size=len(self.leaves)).astype(bool)
        cnt = np.zeros(len(self.kid)); cor = np.zeros(len(self.kid))
        for li in range(len(self.leaves)):
            if not self.is_truth[li]:
                continue
            ok = int(self.grades[li])
            for nid in self.leaf_path[li]:
                cnt[nid] += 1; cor[nid] += ok
        with np.errstate(invalid="ignore", divide="ignore"):
            self.Vtrue_ho = np.where(cnt > 0, cor / cnt, np.nan)

    def sample_ho(self, schedule, rng):
        chosen = []
        def descend(node, depth):
            if depth == len(schedule) or not node.children:
                ev = [li for li in node.leaf_idxs if not self.is_truth[li]]
                if ev:
                    chosen.append(int(rng.choice(ev)))
                return
            kids = list(node.children.values())
            b = min(schedule[depth], len(kids))
            for ci in rng.choice(len(kids), size=b, replace=False):
                descend(kids[ci], depth + 1)
        descend(self.t.root, 0)
        if not chosen:
            return []
        cnt = {}; cor = {}
        for li in chosen:
            ok = int(self.grades[li])
            for nid in self.leaf_path[li]:
                cnt[nid] = cnt.get(nid, 0) + 1; cor[nid] = cor.get(nid, 0) + ok
        Vhat = {n: cor[n] / cnt[n] for n in cnt}
        seen = set(); E = []
        for li in chosen:
            path = self.leaf_path[li]
            for d in range(len(path) - 1):
                pk, ck = path[d], path[d + 1]
                if (pk, ck) in seen:
                    continue
                seen.add((pk, ck))
                at = self.Vtrue_ho[ck] - self.Vtrue_ho[pk]
                if np.isnan(at):
                    continue
                E.append(dict(depth=d, A_hat=Vhat[ck] - Vhat[pk], A_true=at))
        return E

    # ---- evaluate a schedule over many seeds ---------------------------
    def evaluate(self, schedule, n_seeds=20, base_seed=0):
        corrs, signs, costs, nedges = [], [], [], []
        for s in range(n_seeds):
            rng = np.random.default_rng(base_seed + s)
            E, cost = self.sample(schedule, rng)
            if len(E) < 3:
                continue
            ah = np.array([e["A_hat"] for e in E])
            at = np.array([e["A_true"] for e in E])
            mask = np.abs(at) > 1e-9
            if mask.sum() >= 3 and ah.std() > 0 and at[mask].std() > 0:
                corrs.append(np.corrcoef(ah[mask], at[mask])[0, 1])
                signs.append(np.mean(np.sign(ah[mask]) == np.sign(at[mask])))
            costs.append(cost); nedges.append(len(E))
        f = lambda a: (round(float(np.mean(a)), 3), round(float(np.std(a)), 3)) if a else (None, None)
        return dict(schedule=schedule, n_leaves=int(np.prod(schedule)),
                    recovery_corr=f(corrs), sign_acc=f(signs),
                    tokens=f(costs), n_edges=f(nedges))


if __name__ == "__main__":
    import sys, pandas as pd
    corr = pd.read_parquet("correctness.parquet")
    path = sys.argv[1] if len(sys.argv) > 1 else "prompt_0003.json"
    pid = int(path.split("_")[1].split(".")[0])
    t = Tree(path)
    t.apply_grades(corr[corr.prompt_idx == pid].sort_values("leaf_index")["correct"].to_numpy())
    sim = Simulator(t)
    print(f"=== {path}  V(root)={t.root.V():.3f}  n_leaves={t.n_leaves} ===")
    # matched-budget schedules: same leaf product, different shape
    for S in [[4, 16, 16, 16], [4, 8, 8, 8], [2, 4, 4, 4], [4, 4, 4, 4],
              [2, 2, 8, 8], [8, 4, 4, 2], [2, 8, 8, 2]]:
        r = sim.evaluate(S, n_seeds=15)
        print(f"  S={str(S):16s} leaves~{r['n_leaves']:5d}  "
              f"corr={r['recovery_corr'][0]}±{r['recovery_corr'][1]}  "
              f"sign={r['sign_acc'][0]}  tok={r['tokens'][0]}  edges={r['n_edges'][0]}")

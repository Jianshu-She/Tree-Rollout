"""Radial render of one real giant tree, leaves colored by correctness."""
import sys, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from tree_loader import Tree, CHUNK

path = sys.argv[1] if len(sys.argv) > 1 else "prompt_0072.json"
pid = int(path.split("_")[1].split(".")[0])
corr = pd.read_parquet("correctness.parquet")
t = Tree(path)
g = corr[corr.prompt_idx == pid].sort_values("leaf_index")["correct"].to_numpy()
t.apply_grades(g)
eos = np.array([l["eos_depth"] for l in t.leaves])

# deepest branch node (key) per leaf
deepest = {}
for li, leaf in enumerate(t.leaves):
    tok = leaf["token_ids"]; key = ()
    for d in range(1, 5):
        k = tuple(tok[: d * CHUNK])
        if k in t.index:
            key = k
        else:
            break
        if len(tok) <= d * CHUNK:
            break
    deepest.setdefault(key, []).append(li)

node_angle = {}
leaf_angle = {}
def assign(node, lo, hi):
    node_angle[node.key] = (lo + hi) / 2
    term = deepest.get(node.key, [])
    children = list(node.children.values())
    total = sum(c.n_leaves for c in children) + len(term)
    if total == 0:
        return
    x = lo
    for li in term:
        w = (hi - lo) / total
        leaf_angle[li] = x + w / 2; x += w
    for c in children:
        w = (hi - lo) * c.n_leaves / total
        assign(c, x, x + w); x += w
assign(t.root, 0, 2 * np.pi)

def xy(theta, r):
    return r * np.cos(theta), r * np.sin(theta)

# branch edges (depth 0..4)
seg = []
for key, ang in node_angle.items():
    if not key:
        continue
    d = len(key) // CHUNK
    pkey = key[: (d - 1) * CHUNK]
    if pkey in node_angle:
        seg.append([xy(node_angle[pkey], d - 1), xy(ang, d)])
# leaf spokes + endpoints
lx, ly, lc = [], [], []
spokes = []
for li, ang in leaf_angle.items():
    dk = next(k for k in [tuple(t.leaves[li]["token_ids"][:d*CHUNK]) for d in range(4,-1,-1)] if k in node_angle)
    r0 = len(dk) // CHUNK
    r1 = max(eos[li], r0 + 0.3)
    spokes.append([xy(ang, r0), xy(ang, r1)])
    x, y = xy(ang, r1); lx.append(x); ly.append(y); lc.append(g[li])
lx, ly, lc = np.array(lx), np.array(ly), np.array(lc)

fig, ax = plt.subplots(figsize=(11, 11))
ax.add_collection(LineCollection(spokes, colors="#E8E8E8", linewidths=0.25, zorder=1))
ax.add_collection(LineCollection(seg, colors="#888", linewidths=0.6, zorder=2))
ax.scatter(lx[lc == 0], ly[lc == 0], s=2.0, c="#CCCCCC", zorder=3, label="wrong")
ax.scatter(lx[lc == 1], ly[lc == 1], s=6.0, c="#2CA02C", zorder=4, label="correct")
ax.scatter([0], [0], s=80, c="black", zorder=5)
ax.text(0, 0, "Q", fontsize=9, ha="center", va="center", color="white", zorder=6)
ax.set_aspect("equal"); ax.axis("off")
ax.set_title(f"One real giant tree (prompt {pid})\n"
             f"{t.n_leaves} leaves | center->out = reasoning steps | green = correct | "
             f"solve rate {g.mean():.0%}", fontsize=12)
ax.legend(loc="upper right", markerscale=3, fontsize=11)
plt.tight_layout(); plt.savefig("giant_tree.png", dpi=130)
print("saved giant_tree.png  leaves=", t.n_leaves, "correct=", int(g.sum()))

"""Clean schematic of the giant-tree shape: branch 4x16x16x16 then go straight."""
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(13, 8))

# visual branching (small, for readability) -- REAL numbers go in labels
vis = [3, 3, 3]          # drawn children per level for steps 1,2,3
real_lbl = ["x4", "x16", "x16", "x16"]
real_cum = ["4", "64", "1024", "16384"]

# build positions level by level
levels = [[0.0]]         # root x positions
for b in vis:
    prev = levels[-1]
    span = 1.0
    new = []
    for px in prev:
        width = span / len(prev)
        for j in range(b):
            new.append(px + (j - (b - 1) / 2) * width / b)
    levels.append(new)

ystep = 1.0
# draw branch levels (y from 0 down)
for d in range(len(levels) - 1):
    y0 = -d * ystep; y1 = -(d + 1) * ystep
    par = levels[d]; chi = levels[d + 1]
    k = len(chi) // len(par)
    for i, px in enumerate(par):
        for j in range(k):
            cx = chi[i * k + j]
            ax.plot([px, cx], [y0, y1], color="#4C72B0", lw=1.1, zorder=1)
for d, xs in enumerate(levels):
    ax.scatter(xs, [-d * ystep] * len(xs), s=40, c="#33548A", zorder=2)

# linear tails: from each深度3 node go straight down, end in a leaf (green/grey)
rng = np.random.default_rng(1)
leaf_y = -(len(levels) - 1) * ystep
for k, cx in enumerate(levels[-1]):
    depth_extra = rng.integers(2, 6)            # how far it runs straight
    yend = leaf_y - depth_extra * ystep
    ax.plot([cx, cx], [leaf_y, yend], color="#BBBBBB", lw=1.0, zorder=1)
    correct = rng.random() < 0.3
    ax.scatter([cx], [yend], s=55,
               c=("#2CA02C" if correct else "#CCCCCC"), zorder=3)

# annotations on the right
xann = 0.62
labels = [
    (0, "step 0: the question (root)"),
    (-1, "step 1: branch  x4   -> 4 nodes"),
    (-2, "step 2: each x16  -> 64 nodes"),
    (-3, "step 3: each x16  -> 1024 nodes"),
    (-4, "step 4: each x16  -> 16384 nodes  (almost all leaves start here)"),
]
for y, txt in labels:
    ax.annotate(txt, (xann, y * ystep), fontsize=12, va="center",
                bbox=dict(boxstyle="round", fc="#FFF3CD", ec="#999"))
ax.annotate("step 5 on: NO more branching.\neach path runs straight to the end\n(avg ends step 10, longest step 32)",
            (xann, -6.2 * ystep), fontsize=12, va="center",
            bbox=dict(boxstyle="round", fc="#E8F4E8", ec="#2CA02C"))
ax.annotate("green dot = this path ends CORRECT\ngrey dot = wrong",
            (xann, -8.2 * ystep), fontsize=12, va="center",
            bbox=dict(boxstyle="round", fc="#F0F0F0", ec="#999"))

ax.set_title("Shape of a giant tree: branch hard for 4 steps, then go straight to the end",
             fontsize=14)
ax.axis("off"); ax.set_xlim(-0.6, 1.4)
plt.tight_layout(); plt.savefig("tree_schematic.png", dpi=130)
print("saved tree_schematic.png")

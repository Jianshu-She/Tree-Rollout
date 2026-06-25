"""
Offline giant-tree loader + trie reconstruction for DAPO-Math giant trees.

The HF files store only LEAVES; tree topology is implicit in the shared
128-token chunk prefixes (branching schedule [4,16,16,16] over the first 4
chunks, then linear to depth 32). This module rebuilds the trie so we can
compute Monte-Carlo node values V(node) and per-step advantages A(u->v).

Correctness is NOT in the dataset; pass a `grades` array (len == n_leaves,
1/0 per leaf in file order) to populate V. Until then V is left as None and
only topology/shape stats are available.
"""
from __future__ import annotations
import json
from dataclasses import dataclass, field

CHUNK = 128          # tokens per step/chunk
BRANCH_DEPTH = 4     # branching happens at chunks 0..3; depth>=4 is linear


@dataclass
class Node:
    depth: int
    key: tuple              # chunk-prefix key identifying this node
    leaf_idxs: list = field(default_factory=list)   # all leaves under this node
    children: dict = field(default_factory=dict)    # child_key -> Node
    # filled once grades are supplied:
    n_correct: int | None = None

    @property
    def n_leaves(self) -> int:
        return len(self.leaf_idxs)

    def V(self):
        if self.n_correct is None:
            return None
        return self.n_correct / self.n_leaves if self.n_leaves else None


class Tree:
    def __init__(self, path: str):
        with open(path) as f:
            self.d = json.load(f)
        self.prompt_idx = self.d["prompt_idx"]
        self.leaves = self.d["leaves"]
        self.n_leaves = len(self.leaves)
        self.root = Node(depth=0, key=())
        self._build()

    def _leaf_path_keys(self, tok):
        """Chunk-prefix keys for depths 1..BRANCH_DEPTH (topology lives here)."""
        keys = []
        for depth in range(1, BRANCH_DEPTH + 1):
            pref = tuple(tok[: depth * CHUNK])
            keys.append(pref)
            if len(tok) <= depth * CHUNK:   # leaf ended before this depth
                break
        return keys

    def _build(self):
        self.nodes_by_depth = {0: [self.root]}
        index = {(): self.root}
        for li, leaf in enumerate(self.leaves):
            tok = leaf["token_ids"]
            self.root.leaf_idxs.append(li)
            parent = self.root
            for depth, key in enumerate(self._leaf_path_keys(tok), start=1):
                node = index.get(key)
                if node is None:
                    node = Node(depth=depth, key=key)
                    index[key] = node
                    parent.children[key] = node
                    self.nodes_by_depth.setdefault(depth, []).append(node)
                node.leaf_idxs.append(li)
                parent = node
        self.index = index

    def apply_grades(self, grades):
        """grades: list/array of 0/1 per leaf in file order."""
        assert len(grades) == self.n_leaves, (len(grades), self.n_leaves)
        self.grades = grades
        for node in self.index.values():
            node.n_correct = sum(grades[i] for i in node.leaf_idxs)

    def branch_nodes(self, depth):
        return self.nodes_by_depth.get(depth, [])

    def shape_stats(self):
        eos = [l["eos_depth"] for l in self.leaves]
        nat = sum(1 for l in self.leaves if l["natural_eos"])
        ntok = [l["n_tokens"] for l in self.leaves]
        return {
            "prompt_idx": self.prompt_idx,
            "n_leaves": self.n_leaves,
            "nodes_per_depth": {d: len(v) for d, v in sorted(self.nodes_by_depth.items())},
            "natural_eos_frac": round(nat / self.n_leaves, 3),
            "eos_depth_mean": round(sum(eos) / len(eos), 2),
            "eos_depth_le1_frac": round(sum(1 for e in eos if e <= 1) / len(eos), 3),
            "tok_mean": round(sum(ntok) / len(ntok), 1),
            "tok_total": sum(ntok),
        }


if __name__ == "__main__":
    import sys
    for p in sys.argv[1:]:
        t = Tree(p)
        s = t.shape_stats()
        print(f"\n=== {p} (prompt {s['prompt_idx']}) ===")
        print(f"  n_leaves={s['n_leaves']}  tok_total={s['tok_total']:,}  tok_mean={s['tok_mean']}")
        print(f"  natural_eos={s['natural_eos_frac']}  eos_depth_mean={s['eos_depth_mean']}  eos<=1_frac={s['eos_depth_le1_frac']}")
        print(f"  branch nodes per depth: {s['nodes_per_depth']}")

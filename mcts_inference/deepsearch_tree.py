"""DeepSearch-style MCTS with global frontier selection and entropy guidance.

Key differences from standard MCTS:
1. Global frontier selection: instead of root-to-leaf UCT, score ALL non-terminal
   nodes globally and expand the best one.
2. Entropy-guided selection: use token-level entropy as uncertainty bonus.
3. Dynamic expansion width: children per expansion decays with depth.

Reference: DeepSearch (arxiv 2509.25454, ICLR 2026)
"""

import math
from typing import List, Optional

from config import MCTSConfig
from mcts_node import MCTSNode
from policy_model import PolicyModel
from reward_model import RewardModel, LogprobRewardModel
from utils import is_terminal_text

import numpy as np


class DeepSearchTree:
    """DeepSearch-style MCTS tree.

    Instead of UCT root-to-leaf selection, uses global frontier selection
    with entropy-guided scoring to pick the most promising node to expand.
    """

    def __init__(self, question: str, config: MCTSConfig):
        self.question = question
        self.config = config
        self.root = MCTSNode(text="", parent=None)
        self.root.visit_count = 1
        self.total_nodes = 1

        # DeepSearch hyperparameters — defaults match the official repo
        # (github.com/smiles724/DeepSearch, deepsearch/rollout/sglang_mcts_wrapper.py)
        self.lambda_quality = getattr(config, "ds_lambda_quality", 0.4)  # global_lambda1
        self.lambda_entropy = getattr(config, "ds_lambda_entropy", 0.4)  # global_lambda2
        self.lambda_depth = getattr(config, "ds_lambda_depth", 0.01)     # global_lambda3
        self.lambda_self_q = getattr(config, "ds_lambda_self_q", 0.0)    # global_lambda4
        self.depth_bonus_type = getattr(config, "ds_depth_bonus_type", "sqrt")
        self.base_expansion_width = config.num_children  # starting width
        self.width_decay = getattr(config, "ds_width_decay", 1)  # reduce by this per depth step
        self.min_width = getattr(config, "ds_min_width", 1)

        # Store entropy per node
        self._node_entropy = {}  # node id -> entropy value

    # ------------------------------------------------------------------
    # Entropy computation
    # ------------------------------------------------------------------

    @staticmethod
    def compute_entropy(logprobs: List[float]) -> float:
        """Compute average token-level entropy from logprobs.

        Uses -mean(logprob) as a proxy for entropy.
        Higher value = more uncertain.
        """
        if not logprobs:
            return 0.0
        return -np.mean(logprobs)

    # ------------------------------------------------------------------
    # Dynamic expansion width
    # ------------------------------------------------------------------

    def get_expansion_width(self, depth: int) -> int:
        """Get number of children to generate at this depth.

        Width decreases with depth: wider exploration at shallow levels,
        narrower at deeper levels (matching flat rollout observations).
        """
        step_size = max(1, self.config.max_depth // max(self.base_expansion_width, 1))
        step_number = depth // step_size
        width = self.base_expansion_width - step_number * self.width_decay
        return max(self.min_width, int(width))

    # ------------------------------------------------------------------
    # GLOBAL FRONTIER SELECTION
    # ------------------------------------------------------------------

    def _collect_all_nodes(self, node: MCTSNode) -> List[MCTSNode]:
        """Collect all non-terminal nodes in the tree."""
        result = []
        if not node.is_terminal:
            result.append(node)
        for child in node.children:
            result.extend(self._collect_all_nodes(child))
        return result

    def global_select(self) -> MCTSNode:
        """Select the best node to expand using global frontier scoring.

        Score = quality_potential + uncertainty_bonus + depth_bonus

        - quality_potential: tanh(parent_q) — prefer children of high-value parents
        - uncertainty_bonus: entropy * lambda — prefer uncertain nodes
        - depth_bonus: encourage deeper exploration
        """
        candidates = self._collect_all_nodes(self.root)

        # Filter to expandable nodes (leaves or nodes that can have more children)
        expandable = []
        for node in candidates:
            if node.is_leaf:
                expandable.append(node)
            elif len(node.children) < self.get_expansion_width(node.depth):
                expandable.append(node)

        if not expandable:
            # Fallback: return any non-terminal leaf
            leaves = [n for n in candidates if n.is_leaf]
            if leaves:
                return leaves[0]
            return self.root

        # Score each candidate
        best_node = None
        best_score = float("-inf")

        for node in expandable:
            # Quality potential: uses parent_q, optionally self_q via lambda_self_q
            if node.parent is not None:
                quality = self.lambda_quality * math.tanh(node.parent.q_value)
            else:
                quality = 0.0
            if self.lambda_self_q > 0 and node.visit_count > 0:
                quality += self.lambda_self_q * math.tanh(node.q_value)

            # Uncertainty bonus: token-level entropy proxy (-mean(logprob))
            entropy = self._node_entropy.get(id(node), 0.0)
            uncertainty = self.lambda_entropy * entropy

            # Depth bonus: official default is sqrt(d / max_depth)
            if self.depth_bonus_type == "sqrt":
                depth_bonus = self.lambda_depth * math.sqrt(
                    node.depth / max(1, self.config.max_depth)
                )
            elif self.depth_bonus_type == "log":
                depth_bonus = self.lambda_depth * math.log(node.depth + 1)
            else:  # "constant"
                depth_bonus = self.lambda_depth * node.depth

            score = quality + uncertainty + depth_bonus

            if score > best_score:
                best_score = score
                best_node = node

        return best_node or self.root

    # ------------------------------------------------------------------
    # EXPAND
    # ------------------------------------------------------------------

    def expand(
        self,
        node: MCTSNode,
        policy_model: PolicyModel,
        reward_model: RewardModel,
    ) -> List[MCTSNode]:
        """Generate children for the selected node."""
        if node.is_terminal:
            return []
        if node.depth >= self.config.max_depth:
            node.is_terminal = True
            return []

        partial = node.get_full_reasoning()
        n_children = self.get_expansion_width(node.depth)

        # If node already has some children, only generate the remaining
        remaining = n_children - len(node.children)
        if remaining <= 0:
            return []

        continuations = policy_model.generate_continuations(
            question=self.question,
            partial_reasoning=partial,
            n=remaining,
        )

        new_children: List[MCTSNode] = []
        for text, logprobs, hit_stop in continuations:
            if not text.strip():
                continue

            full_reasoning = partial + text
            terminal = hit_stop or is_terminal_text(full_reasoning)

            # Score with reward model
            if isinstance(reward_model, LogprobRewardModel):
                score = reward_model.score_from_logprobs(logprobs)
            else:
                chunks = node.get_path_texts()[1:]
                chunks.append(text)
                scores = reward_model.score(self.question, chunks)
                score = scores[-1]

            child = MCTSNode(
                text=text,
                parent=None,
                prm_score=score,
                token_count=len(logprobs) if logprobs else len(text.split()),
                is_terminal=terminal,
            )
            node.add_child(child)
            new_children.append(child)
            self.total_nodes += 1

            # Store entropy for the child
            entropy = self.compute_entropy(logprobs)
            self._node_entropy[id(child)] = entropy

        if not new_children and not node.children:
            node.is_terminal = True

        return new_children

    # ------------------------------------------------------------------
    # BACKPROPAGATION
    # ------------------------------------------------------------------

    @staticmethod
    def backpropagate(node: MCTSNode, value: float) -> None:
        """Walk from node to root, updating visit counts and values."""
        current = node
        while current is not None:
            current.visit_count += 1
            current.total_value += value
            current = current.parent

    # ------------------------------------------------------------------
    # Best path extraction
    # ------------------------------------------------------------------

    def best_terminal_node(self) -> Optional[MCTSNode]:
        terminals = self._collect_terminals(self.root)
        if not terminals:
            return None
        return max(terminals, key=lambda n: n.visit_count)

    def best_q_terminal(self) -> Optional[MCTSNode]:
        terminals = self._collect_terminals(self.root)
        if not terminals:
            return None
        return max(terminals, key=lambda n: n.q_value)

    def all_terminal_nodes(self) -> List[MCTSNode]:
        return self._collect_terminals(self.root)

    def _collect_terminals(self, node: MCTSNode) -> List[MCTSNode]:
        if node.is_terminal:
            return [node]
        result = []
        for child in node.children:
            result.extend(self._collect_terminals(child))
        return result

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        """Serialize the entire tree into a nested dict."""
        best = self.best_terminal_node()
        best_path_ids = set()
        if best is not None:
            for node in best.get_path_nodes():
                best_path_ids.add(id(node))

        counter = [0]

        def _serialize(node: MCTSNode) -> dict:
            node_id = counter[0]
            counter[0] += 1
            preview = node.text[:50].replace("\n", " ") if node.text else "(root)"
            return {
                "id": node_id,
                "depth": node.depth,
                "text_preview": preview,
                "visit_count": node.visit_count,
                "q_value": round(node.q_value, 4),
                "prm_score": round(node.prm_score, 4),
                "is_terminal": node.is_terminal,
                "token_count": node.token_count,
                "entropy": round(self._node_entropy.get(id(node), 0.0), 4),
                "on_best_path": id(node) in best_path_ids,
                "children": [_serialize(c) for c in node.children],
            }

        return _serialize(self.root)

    def stats(self) -> dict:
        terminals = self.all_terminal_nodes()
        return {
            "total_nodes": self.total_nodes,
            "terminal_nodes": len(terminals),
            "max_depth_reached": max((t.depth for t in terminals), default=0),
            "root_visits": self.root.visit_count,
        }

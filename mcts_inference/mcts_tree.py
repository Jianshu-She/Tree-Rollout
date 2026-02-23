from typing import List, Optional

from config import MCTSConfig
from mcts_node import MCTSNode
from policy_model import PolicyModel
from reward_model import RewardModel, LogprobRewardModel
from utils import is_terminal_text


class MCTSTree:
    """Monte Carlo Tree Search over reasoning chunks.

    Each node is a 256-token chunk.  The tree is grown by:
      1. SELECT  — UCB1 walk from root to an expandable leaf
      2. EXPAND  — generate children with the policy model, score with PRM
      3. BACKPROP — propagate the reward back to root
    """

    def __init__(self, question: str, config: MCTSConfig):
        self.question = question
        self.config = config
        self.root = MCTSNode(text="", parent=None)
        self.root.visit_count = 1  # avoid log(0) in UCB1
        self.total_nodes = 1

    # ------------------------------------------------------------------
    # SELECT
    # ------------------------------------------------------------------

    def select(self, node: Optional[MCTSNode] = None) -> MCTSNode:
        """Walk from *node* (default root) to the best expandable leaf via UCB1."""
        if node is None:
            node = self.root

        while not node.is_leaf:
            # Pick child with highest UCB1
            best_child = max(
                node.children,
                key=lambda c: c.ucb1(self.config.exploration_constant),
            )
            node = best_child

        return node

    # ------------------------------------------------------------------
    # EXPAND
    # ------------------------------------------------------------------

    def expand(
        self,
        node: MCTSNode,
        policy_model: PolicyModel,
        reward_model: RewardModel,
    ) -> List[MCTSNode]:
        """Generate children for *node* and score them with the reward model.

        Returns the list of newly created child nodes.
        """
        if node.is_terminal:
            return []
        if node.depth >= self.config.max_depth:
            node.is_terminal = True
            return []

        partial = node.get_full_reasoning()

        # Generate continuations
        continuations = policy_model.generate_continuations(
            question=self.question,
            partial_reasoning=partial,
            n=self.config.num_children,
        )

        new_children: List[MCTSNode] = []
        for text, logprobs, hit_stop in continuations:
            if not text.strip():
                continue

            # Determine if the cumulative reasoning contains a completed answer
            full_reasoning = partial + text
            terminal = hit_stop or is_terminal_text(full_reasoning)

            # Score with reward model
            if isinstance(reward_model, LogprobRewardModel):
                score = reward_model.score_from_logprobs(logprobs)
            else:
                # Full-path scoring: score all chunks from root to this new node
                chunks = node.get_path_texts()[1:]  # skip root empty text
                chunks.append(text)
                scores = reward_model.score(self.question, chunks)
                score = scores[-1]  # score of the newest chunk

            child = MCTSNode(
                text=text,
                parent=None,  # set by add_child
                prm_score=score,
                token_count=len(logprobs) if logprobs else len(text.split()),
                is_terminal=terminal,
            )
            node.add_child(child)
            new_children.append(child)
            self.total_nodes += 1

        # If no valid children were produced, mark node terminal
        if not new_children:
            node.is_terminal = True

        return new_children

    # ------------------------------------------------------------------
    # BACKPROPAGATION
    # ------------------------------------------------------------------

    @staticmethod
    def backpropagate(node: MCTSNode, value: float) -> None:
        """Walk from *node* to root, updating visit counts and values."""
        current = node
        while current is not None:
            current.visit_count += 1
            current.total_value += value
            current = current.parent

    # ------------------------------------------------------------------
    # Best path extraction
    # ------------------------------------------------------------------

    def best_terminal_node(self) -> Optional[MCTSNode]:
        """Return the terminal node with the highest visit count."""
        terminals = self._collect_terminals(self.root)
        if not terminals:
            return None
        return max(terminals, key=lambda n: n.visit_count)

    def best_q_terminal(self) -> Optional[MCTSNode]:
        """Return the terminal node with the highest Q value."""
        terminals = self._collect_terminals(self.root)
        if not terminals:
            return None
        return max(terminals, key=lambda n: n.q_value)

    def all_terminal_nodes(self) -> List[MCTSNode]:
        """Return all terminal nodes in the tree."""
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
        """Serialize the entire tree into a nested dict for visualization."""
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
                "on_best_path": id(node) in best_path_ids,
                "children": [_serialize(c) for c in node.children],
            }

        return _serialize(self.root)

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def stats(self) -> dict:
        terminals = self.all_terminal_nodes()
        return {
            "total_nodes": self.total_nodes,
            "terminal_nodes": len(terminals),
            "max_depth_reached": max((t.depth for t in terminals), default=0),
            "root_visits": self.root.visit_count,
        }

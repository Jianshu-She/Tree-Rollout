import math
from typing import Optional, List


class MCTSNode:
    """A single node in the MCTS tree.

    Each node represents a 256-token chunk of reasoning.
    The full solution is the concatenation of chunks from root to this node.
    """

    def __init__(
        self,
        text: str = "",
        parent: Optional["MCTSNode"] = None,
        prm_score: float = 0.0,
        token_count: int = 0,
        is_terminal: bool = False,
    ):
        self.text = text
        self.parent = parent
        self.children: List["MCTSNode"] = []
        self.prm_score = prm_score
        self.token_count = token_count
        self.is_terminal = is_terminal

        # MCTS statistics
        self.visit_count: int = 0
        self.total_value: float = 0.0

        # Depth
        self.depth: int = 0 if parent is None else parent.depth + 1

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def q_value(self) -> float:
        """Mean value (exploitation term)."""
        if self.visit_count == 0:
            return 0.0
        return self.total_value / self.visit_count

    @property
    def is_leaf(self) -> bool:
        """True if the node has no children."""
        return len(self.children) == 0

    @property
    def is_expandable(self) -> bool:
        """True if the node can still be expanded (not terminal, not at max depth checked externally)."""
        return not self.is_terminal and self.is_leaf

    # ------------------------------------------------------------------
    # UCB1
    # ------------------------------------------------------------------

    def ucb1(self, exploration_constant: float = 1.414) -> float:
        """Upper Confidence Bound for this node.

        UCB1 = Q(node) + C * sqrt( ln(N_parent) / N(node) )
        """
        if self.visit_count == 0:
            return float("inf")  # unvisited nodes have highest priority
        if self.parent is None:
            return self.q_value
        exploitation = self.q_value
        exploration = exploration_constant * math.sqrt(
            math.log(self.parent.visit_count) / self.visit_count
        )
        return exploitation + exploration

    # ------------------------------------------------------------------
    # Path helpers
    # ------------------------------------------------------------------

    def get_path_nodes(self) -> List["MCTSNode"]:
        """Return list of nodes from root to this node (inclusive)."""
        path = []
        node = self
        while node is not None:
            path.append(node)
            node = node.parent
        path.reverse()
        return path

    def get_path_texts(self) -> List[str]:
        """Return list of text chunks from root to this node."""
        return [n.text for n in self.get_path_nodes()]

    def get_full_reasoning(self) -> str:
        """Concatenate all text from root to this node."""
        return "".join(self.get_path_texts())

    def get_total_tokens(self) -> int:
        """Sum of token counts from root to this node."""
        return sum(n.token_count for n in self.get_path_nodes())

    # ------------------------------------------------------------------
    # Tree modification
    # ------------------------------------------------------------------

    def add_child(self, child: "MCTSNode") -> None:
        child.parent = self
        child.depth = self.depth + 1
        self.children.append(child)

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        status = "T" if self.is_terminal else "N"
        return (
            f"MCTSNode(depth={self.depth}, visits={self.visit_count}, "
            f"Q={self.q_value:.3f}, PRM={self.prm_score:.3f}, "
            f"tokens={self.token_count}, {status})"
        )
